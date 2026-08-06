# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""v1-mean corrector trainer: CF x PDD, clean-history teacher on the mean velocity.

Crosses :mod:`train_v1`'s counterfactual clean-history teacher (CF) with the
renoise-loop integration probed by :mod:`gate_mean_velocity` (the PDD axis).
The LoRA corrector r_phi learns to jump straight to the frozen model's
*integrated* prediction: its single forward at ``timesteps[start]`` under the
drifted history must match the mean velocity of the frozen model's full
renoise loop over solver steps ``[start, end)`` under the lap-aligned clean
history::

    L = ||u_{theta+LoRA}(z_start, h_gen) - u_theta^loop(h_clean)||^2
        / ||u_theta^loop(h_clean) - u_theta^loop(h_gen)||^2

``u^loop`` replays the host loop exactly as
:func:`gate_mean_velocity.probe_u_mean`: per probe draw, one pre-drawn eps
stack anchors ``z_start = (1 - sigma_start) * x0 + sigma_start * eps[0]``
from the rollout's own x0 at ``sigmas[start]`` and feeds the between-step
renoises; the stack is shared by the clean-teacher, drifted-base, and
student passes. The student needs no loop: its one-jump mean velocity from
``sigmas[start]`` is its flow prediction itself,
``u = (z - xhat0) / sigma = flow``. The drift-gap denominator is required
(raw MSE diverges; reference finding); ``R^2 = 1 - L``.

``WINDOW="start:end"`` (default ``"0:4"``) selects the integration window as
in :mod:`gate_mean_velocity`; the default checkpoint is
``outputs/lora_v1_mean_w{start}{end}.pt``. Everything else -- three history
rebuilds per sampled cell (clean teacher, drifted base, LoRA-scaled student)
amortized over ``PROBES`` draws, functional attention + per-block gradient
checkpointing on the student pass, val holdout, resumable checkpoints --
matches :mod:`train_v1`; see its docstring for the host adaptations.

Run from the repo root (resumable: re-run the same command)::

    STEPS=800 .venv/bin/python integrations/lingbot/drift_correction/train_v1_mean.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from _lora import apply_lora, lora_parameters, set_lora_scale, unwrap_compiled
from _rollout import (
    _encode_ctrl,
    build_pipeline,
    camera_stream,
    chunk_camctrl,
    load_scene,
    rebuild_cache,
)
from _train_attn import functional_attention, patch_functional_attention
from gate_faithful import clean_counterpart_history, lap_of
from gate_mean_velocity import _parse_window, probe_u_mean
from torch import Tensor

## Training configuration

BASE = Path("integrations/lingbot/drift_correction")

PAIRS_DIR = Path(os.environ.get("PAIRS_DIR", str(BASE / "outputs/pairs")))
"""Clip files from ``build_pairs.py``."""

WINDOW = os.environ.get("WINDOW", "0:4")
"""Solver-step window ``"start:end"`` into
``scheduler.denoising_step_list``/``denoising_sigmas``, as in
:mod:`gate_mean_velocity`: ``z_start`` is built at ``sigmas[start]`` and the
teacher renoise loop covers steps ``[start, end)``."""

W_START, W_END = _parse_window(WINDOW)

CKPT = Path(
    os.environ.get("CKPT", str(BASE / f"outputs/lora_v1_mean_w{W_START}{W_END}.pt"))
)
"""Output checkpoint (LoRA + optimizer + step); saved every ``SAVE_EVERY``."""

VAL_CLIP = os.environ.get("VAL_CLIP", "clip_05")
"""Held-out clip stem for the val split."""

STEPS = int(os.environ.get("STEPS", "800"))
LR = float(os.environ.get("LR", "1e-4"))
WARMUP = 60
"""Linear LR warmup steps; required for stability (reference finding)."""

GRAD_CLIP = 1.0
RANK = 16
EVAL_EVERY = 50
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "200"))
"""Owner requirement: partial checkpoints every <= 200 steps (box wedges)."""

PROBES = int(os.environ.get("PROBES", "4"))
"""Eps-stack draws per rebuilt cell. Rebuilding the three history caches
costs ~3 * k network forwards per cell while a probe costs at most
``W_END - W_START``, so several probes per cell amortize the rebuild; each
draw gets its own renoise stack (fresh CPU generator)."""

MIN_LAP = 3
"""Training cells live in laps >= 3 (the drifted side is well past lap 1,
matching the faithful gate's probe laps); lap 1 is the clean reference and
lap 2 the warmup."""

SEED = int(os.environ.get("SEED", "0"))


def checkpoint_blocks(network) -> None:
    """Route every DiT block ``forward`` through gradient checkpointing.

    Per-instance overrides (not wrapper modules) so the network loop's
    ``isinstance(block, Block)`` assertion keeps passing. Requires the
    functional-attention toggle on grad passes: forward recomputation must
    be side-effect free and must retake the same code path (backward runs
    inside ``functional_attention()``). No-op under ``no_grad`` passes
    (history rebuilds, teacher/base probes).
    """
    from torch.utils.checkpoint import checkpoint

    def wrap(fn):
        def ckpt_fn(*args, _inner=fn, **kwargs):
            if not torch.is_grad_enabled():
                return _inner(*args, **kwargs)
            return checkpoint(_inner, *args, use_reentrant=False, **kwargs)

        return ckpt_fn

    for block in network.blocks:
        block.forward = wrap(block.forward)


def main() -> None:
    clips = sorted(PAIRS_DIR.glob("clip_*.pt"))
    assert clips, f"no clips under {PAIRS_DIR}; run build_pairs.py first"
    rng = np.random.default_rng(SEED)

    pipe = build_pipeline()
    dm = pipe.diffusion_model
    device = pipe.device
    dtype = dm.dtype
    transformer = dm.transformer
    scheduler = dm.scheduler
    timesteps = scheduler.denoising_step_list
    sigmas = scheduler.denoising_sigmas
    n_steps = int(timesteps.shape[0])
    assert W_END <= n_steps, f"WINDOW {WINDOW!r} out of range for {n_steps} steps"
    t_hi = int(timesteps[W_START])

    # Pin the one-shot text/image encoders: every rebuild goes through
    # ``pipe.initialize_cache``, whose default release/reload cycle would
    # re-load UMT5 from checkpoint three times per training step.
    pipe._ensure_oneshot_encoders_loaded()
    encoders = (pipe.text_encoder, pipe.image_encoder)

    datas = [torch.load(p, map_location="cpu", weights_only=False) for p in clips]
    val_ids = [i for i, p in enumerate(clips) if p.stem == VAL_CLIP]
    assert val_ids, f"VAL_CLIP={VAL_CLIP} not among {[p.stem for p in clips]}"
    train_ids = [i for i in range(len(datas)) if i not in val_ids]
    # Per-clip lap geometry: build_pairs varies the leg length across clips
    # (the repeat-prior fix), so nothing may assume a shared lap_chunks.
    ks_by_clip = [
        [
            k
            for k in range(int(d["num_chunk"]))
            if lap_of(k, int(d["lap_chunks"])) >= MIN_LAP
        ]
        for d in datas
    ]
    print(
        f"{len(datas)} clips (val {VAL_CLIP}) | window [{W_START}:{W_END})"
        f" u_mean {t_hi}->0 | lap_chunks "
        f"{[int(d['lap_chunks']) for d in datas]} | cells/clip "
        f"{[len(ks) for ks in ks_by_clip]} (laps >= {MIN_LAP}, clean lap 1)",
        flush=True,
    )

    network = unwrap_compiled(transformer.network)
    network.requires_grad_(False)  # frozen base; only the LoRA A/B path trains
    wrapped = apply_lora(network, rank=RANK)
    params = lora_parameters(network)
    print(
        f"LoRA on {len(wrapped)} projections | "
        f"{sum(p.numel() for p in params) / 1e6:.2f}M params",
        flush=True,
    )
    patch_functional_attention()
    checkpoint_blocks(network)
    opt = torch.optim.Adam(params, lr=LR)

    start_step = 0
    if CKPT.exists():
        state = torch.load(CKPT, map_location="cpu", weights_only=False)
        for i, p in enumerate(params):
            p.data.copy_(state["lora"][i].to(p.device, p.dtype))
        opt.load_state_dict(state["opt"])
        start_step = state["step"]
        rng = np.random.default_rng(state["rng_seed"])
        print(f"RESUMED from {CKPT} at step {start_step}", flush=True)

    def save(step: int) -> None:
        CKPT.parent.mkdir(parents=True, exist_ok=True)
        tmp = CKPT.with_suffix(".tmp")
        torch.save(
            {
                "lora": {i: p.detach().cpu() for i, p in enumerate(params)},
                "opt": opt.state_dict(),
                "step": step,
                "rng_seed": SEED + step,  # fresh stream on resume
            },
            tmp,
        )
        tmp.replace(CKPT)

    _runtimes: dict[int, tuple] = {}

    def clip_runtime(c: int) -> tuple:
        """Scene + per-chunk camera payloads of clip ``c`` (memoized)."""
        if c not in _runtimes:
            d = datas[c]
            scene = load_scene(int(d["scene_idx"]))
            poses_t, intr_t, ws = camera_stream(scene, d["poses"])
            camctrls = chunk_camctrl(
                poses_t, intr_t, ws, pipe, n_chunks=int(d["num_chunk"])
            )
            _runtimes[c] = (scene, camctrls)
        return _runtimes[c]

    def rebuild(scene, latents, camctrls):
        """``rebuild_cache`` with the pinned encoders re-attached."""
        pipe.text_encoder, pipe.image_encoder = encoders
        return rebuild_cache(pipe, scene, latents, camctrls)

    def open_probe(cache, k: int, camctrls):
        """Encode chunk ``k``'s control and open its probe bracket."""
        enc_p = _encode_ctrl(pipe, cache, k, camctrls[k])
        cache.transformer_cache.start(k)
        return enc_p

    def predict_u_student(cache, enc_p, z_first: Tensor) -> Tensor:
        """One functional-attention forward at ``timesteps[W_START]`` -> fp32 flow.

        The student's one-jump mean velocity from ``sigmas[W_START]`` is its
        flow prediction itself: ``u = (z - xhat0) / sigma = flow``.
        """
        t = timesteps[W_START].to(dtype=dtype)
        z_p = transformer.patchify_and_maybe_split_cp(z_first)
        with functional_attention():
            flow = transformer.predict_flow(
                noisy_latent=z_p,
                timestep=t,
                cache=cache.transformer_cache,
                input=enc_p,
            )
        return transformer.unpatchify_and_maybe_gather_cp(flow).float()

    def sample_losses(
        c: int, k: int, grad: bool, rng_: np.random.Generator
    ) -> tuple[float, float]:
        """One rebuilt cell, ``PROBES`` draws -> (mean loss, mean |r_u|^2).

        Teacher and base integrate the full renoise loop over the window
        (:func:`gate_mean_velocity.probe_u_mean`, no grad); the student does
        one forward on the shared ``z_start``. Grad-mode backwards happen
        inside (per probe, scaled ``1/PROBES``): the student cache and its
        open bracket must stay alive during checkpoint recomputation, and
        the recompute must retake the functional attention path.
        """
        scene, camctrls = clip_runtime(c)
        lat = datas[c]["latents"]
        lap_chunks = int(datas[c]["lap_chunks"])
        x0 = lat[k]  # cpu fp32; probe_u_mean anchors it on device itself
        stacks = []
        for _ in range(PROBES):
            g = torch.Generator().manual_seed(int(rng_.integers(2**31)))
            stacks.append(
                [torch.randn(x0.shape, generator=g) for _ in range(W_END - W_START)]
            )

        def probe_all(history: list[Tensor]) -> list[Tensor]:
            """Rebuild one history and integrate every draw (one cache alive)."""
            cache = rebuild(scene, history, camctrls[:k])
            out = [
                probe_u_mean(
                    pipe, cache, k, camctrls[k], x0, eps, start=W_START, end=W_END
                )
                for eps in stacks
            ]
            del cache
            torch.cuda.empty_cache()
            return out

        with torch.no_grad():
            set_lora_scale(network, 0.0)
            u_cleans = probe_all(clean_counterpart_history(lat, k, lap_chunks))
            u_bases = probe_all(lat[:k])
        r_sqs = [float((cl - ba).square().sum()) for cl, ba in zip(u_cleans, u_bases)]

        set_lora_scale(network, 1.0)
        with torch.no_grad():  # LoRA-scaled replay, no grad
            cache = rebuild(scene, lat[:k], camctrls[:k])
            enc_p = open_probe(cache, k, camctrls)
        x0_dev = x0.to(device, torch.float32)
        sigma_start = sigmas[W_START]
        losses = []
        for eps, u_clean, r_sq in zip(stacks, u_cleans, r_sqs):
            z_first = (
                (1.0 - sigma_start) * x0_dev
                + sigma_start * eps[0].to(device, torch.float32)
            ).to(dtype)
            with torch.enable_grad() if grad else torch.no_grad():
                u_student = predict_u_student(cache, enc_p, z_first)
                loss = (u_student - u_clean.to(device)).square().sum() / (r_sq + 1e-8)
            if grad:
                with functional_attention():
                    (loss / PROBES).backward()
            losses.append(float(loss))
        del cache, enc_p
        torch.cuda.empty_cache()
        return sum(losses) / PROBES, sum(r_sqs) / PROBES

    @torch.no_grad()
    def val_r2(n: int = 2) -> float:
        vrng = np.random.default_rng(1234)  # fixed cells/noise across evals
        s = 0.0
        for _ in range(n):
            c = int(vrng.choice(val_ids))
            loss, _ = sample_losses(c, int(vrng.choice(ks_by_clip[c])), False, vrng)
            s += loss
        return 1 - s / n

    torch.set_grad_enabled(True)
    for step in range(start_step + 1, STEPS + 1):
        for pg in opt.param_groups:
            pg["lr"] = LR * min(1.0, step / WARMUP)
        opt.zero_grad()
        c = int(rng.choice(train_ids))
        k = int(rng.choice(ks_by_clip[c]))
        loss, ru_sq = sample_losses(c, k, grad=True, rng_=rng)  # backwards inside
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        opt.step()
        line = (
            f"step {step:5d} | loss {loss:.4f} (train R^2 {1 - loss:+.3f})"
            f" | |r_u|^2 {ru_sq:.1f}"
        )
        if step % EVAL_EVERY == 0 or step == 1:
            line += f" | val R^2 {val_r2():+.3f}"
        print(line, flush=True)
        if step % SAVE_EVERY == 0 or step == STEPS:
            save(step)

    print(
        f"TRAIN-V1-MEAN-DONE | window [{W_START}:{W_END}) u_mean {t_hi}->0"
        f" | final val R^2 {val_r2(4):+.3f} | saved {CKPT}",
        flush=True,
    )


if __name__ == "__main__":
    main()

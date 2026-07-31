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

"""v1 corrector trainer: counterfactual clean-history teacher in x0 space.

Trains the LoRA corrector r_phi on the frozen distilled LingBot-World model
so its x0 prediction under drifted history matches the frozen model's
prediction under the lap-aligned clean history at the same ``z_t``::

    L = ||x0_{theta+LoRA}(z_t, h_gen, t) - x0_theta(z_t, h_clean, t)||^2
        / ||x0_theta(z_t, h_clean, t) - x0_theta(z_t, h_gen, t)||^2

The drift-gap denominator is required (raw MSE diverges; reference
finding); ``R^2 = 1 - L``. All passes share the exact ``z_t`` re-noised
from the rollout's own x0 (native anchoring) at the solver's warped
``(t, sigma)``; ``t`` is sampled uniformly from the 4 distillation steps.

Host adaptations vs the HY / Omnidreams references: LingBot keeps history
only in the rolling ``BlockKVCache`` (no clean-latent buffer, no memory
prefill), so each sampled cell teacher-forces the full history into a fresh
cache via :func:`_rollout.rebuild_cache` -- three rebuilds per cell (clean
teacher, drifted base, LoRA-scaled student), one cache alive at a time (the
21-chunk window KV is ~78 GB on the 14B host). Rebuilds are amortized:
each rebuilt cell is probed at ``PROBES`` independent ``(t, eps)`` draws
before moving on. Probes run through the functional self-attention toggle
(:mod:`_train_attn` -- restores k/v gradients, enables checkpointing) with
per-block gradient checkpointing on the student pass (40 blocks at
dim 5120; the unchunked bf16 tape does not fit next to the window KV).

Run from the repo root (resumable: re-run the same command)::

    STEPS=800 .venv/bin/python integrations/lingbot/drift_correction/train_v1.py
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
from torch import Tensor

## Training configuration

BASE = Path("integrations/lingbot/drift_correction")

PAIRS_DIR = Path(os.environ.get("PAIRS_DIR", str(BASE / "outputs/pairs")))
"""Clip files from ``build_pairs.py``."""

CKPT = Path(os.environ.get("CKPT", str(BASE / "outputs/lora_v1.pt")))
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
"""``(t, eps)`` draws per rebuilt cell. Rebuilding the three history caches
costs ~3 * k network forwards per cell while a probe forward costs one, so
several probes per cell amortize the rebuild; each draw gets its own
timestep and noise."""

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
        f"{len(datas)} clips (val {VAL_CLIP}) | lap_chunks "
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

    def predict_xhat0(cache, enc_p, z_t: Tensor, t_idx: int) -> Tensor:
        """One functional-attention denoise-step forward -> fp32 ``xhat0``."""
        t = timesteps[t_idx].to(dtype=dtype)
        z_p = transformer.patchify_and_maybe_split_cp(z_t)
        with functional_attention():
            flow = transformer.predict_flow(
                noisy_latent=z_p,
                timestep=t,
                cache=cache.transformer_cache,
                input=enc_p,
            )
        flow_u = transformer.unpatchify_and_maybe_gather_cp(flow)
        return z_t.float() - float(sigmas[t_idx]) * flow_u.float()

    def sample_losses(
        c: int, k: int, grad: bool, rng_: np.random.Generator
    ) -> tuple[float, float]:
        """One rebuilt cell, ``PROBES`` draws -> (mean loss, mean |r_t|^2).

        Grad-mode backwards happen inside (per probe, scaled ``1/PROBES``):
        the student cache and its open bracket must stay alive during
        checkpoint recomputation, and the recompute must retake the
        functional attention path.
        """
        scene, camctrls = clip_runtime(c)
        lat = datas[c]["latents"]
        lap_chunks = int(datas[c]["lap_chunks"])
        x0 = lat[k].to(device, torch.float32)
        probes = []
        for _ in range(PROBES):
            t_idx = int(rng_.integers(n_steps))
            g = torch.Generator(device=device).manual_seed(int(rng_.integers(2**31)))
            eps = torch.randn(x0.shape, device=device, dtype=torch.float32, generator=g)
            sig = float(sigmas[t_idx])
            probes.append((t_idx, ((1.0 - sig) * x0 + sig * eps).to(dtype)))

        def probe_all(history: list[Tensor]) -> list[Tensor]:
            """Rebuild one history and probe every draw (one cache alive)."""
            cache = rebuild(scene, history, camctrls[:k])
            enc_p = open_probe(cache, k, camctrls)
            out = [predict_xhat0(cache, enc_p, z_t, t_idx) for t_idx, z_t in probes]
            del cache, enc_p
            torch.cuda.empty_cache()
            return out

        with torch.no_grad():
            set_lora_scale(network, 0.0)
            cleans = probe_all(clean_counterpart_history(lat, k, lap_chunks))
            bases = probe_all(lat[:k])
        r_sqs = [(cl - ba).square().sum() for cl, ba in zip(cleans, bases)]

        set_lora_scale(network, 1.0)
        with torch.no_grad():  # LoRA-scaled replay, no grad
            cache = rebuild(scene, lat[:k], camctrls[:k])
            enc_p = open_probe(cache, k, camctrls)
        losses = []
        for (t_idx, z_t), x0_clean, r_sq in zip(probes, cleans, r_sqs):
            with torch.enable_grad() if grad else torch.no_grad():
                x0_corr = predict_xhat0(cache, enc_p, z_t, t_idx)
                loss = (x0_corr - x0_clean).square().sum() / (r_sq + 1e-8)
            if grad:
                with functional_attention():
                    (loss / PROBES).backward()
            losses.append(float(loss))
        del cache, enc_p
        torch.cuda.empty_cache()
        return sum(losses) / PROBES, sum(float(r) for r in r_sqs) / PROBES

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
        loss, rt_sq = sample_losses(c, k, grad=True, rng_=rng)  # backwards inside
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        opt.step()
        line = (
            f"step {step:5d} | loss {loss:.4f} (train R^2 {1 - loss:+.3f})"
            f" | |r_t|^2 {rt_sq:.1f}"
        )
        if step % EVAL_EVERY == 0 or step == 1:
            line += f" | val R^2 {val_r2():+.3f}"
        print(line, flush=True)
        if step % SAVE_EVERY == 0 or step == STEPS:
            save(step)

    print(f"TRAIN-V1-DONE | final val R^2 {val_r2(4):+.3f} | saved {CKPT}", flush=True)


if __name__ == "__main__":
    main()

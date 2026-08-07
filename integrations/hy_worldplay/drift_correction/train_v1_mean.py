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
window integration probed by :mod:`gate_mean_velocity` (the PDD axis). The
LoRA corrector r_phi learns to jump straight to the frozen model's
*integrated* prediction: its single flow forward at ``timesteps[W_START]``
under the drifted history must match the mean velocity of the frozen model's
Euler roll over solver steps ``[W_START, W_END)`` under the lap-aligned clean
history::

    L = ||u_{theta+LoRA}(z_first, h_gen) - u_theta^euler(h_clean)||^2
        / ||u_theta^euler(h_clean) - u_theta^euler(h_gen)||^2

``u^euler`` is :func:`gate_mean_velocity.integrate_u_mean`. The host solver
is a deterministic Euler ODE (no between-step renoise, unlike the reference
host's renoise loop), so one ``eps`` draw per sample anchors
``z_first = (1 - sigma_start) * x0 + sigma_start * eps`` from the rollout's
own x0 at ``sigmas[W_START]``, and the same ``z_first`` feeds the
clean-teacher, drifted-base, and student passes. The student needs no roll:
a single Euler jump across the window with its flow at ``sigmas[W_START]``
lands on ``z_end = z_first + (sigmas[W_END] - sigmas[W_START]) * flow``, so
``u_mean = (z_first - z_end) / (sigmas[W_START] - sigmas[W_END]) = flow`` --
its flow prediction *is* its one-jump mean velocity. The drift-gap
denominator is required (raw MSE diverges); ``R^2 = 1 - L``.

``WINDOW="start:end"`` (default ``"0:2"``: solver steps t=1000, 960, jump
target at t=888.9) selects the integration window as in
:mod:`gate_mean_velocity`; the default checkpoint is
``outputs/lora_v1_mean_w{start}{end}.pt``. Everything else -- probe
brackets, no-grad LoRA-scaled memory prefill, functional attention +
per-block gradient checkpointing on the student pass (backward before the
bracket close), val split, ``INIT`` resume -- matches :mod:`train_v1`; see
its docstring for the host adaptations.

Run from the repo root (resumable via ``INIT``)::

    STEPS=1500 .venv/bin/python integrations/hy_worldplay/drift_correction/train_v1_mean.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import numpy as np
import torch
from _lora import (
    apply_lora,
    load_lora,
    lora_parameters,
    save_lora,
    set_lora_scale,
    unwrap_compiled,
)
from _pairs import chunk_x0, clean_counterfactual, history_of, load_clip, make_ctrl
from _rollout import build_runner, finish_probe_chunk, start_probe_chunk
from _train_attn import patch_functional_attention
from gate_mean_velocity import _parse_window, integrate_u_mean
from hy_worldplay._action import (
    HyWorldPlayWan21Transformer,
    HyWorldPlayWan21TransformerCache,
)
from hy_worldplay.runner import _resolve_prompt, preprocess_first_frame
from torch import Tensor

## Training configuration

PAIRS_DIR = Path(
    os.environ.get(
        "PAIRS_DIR", "integrations/hy_worldplay/drift_correction/outputs/pairs"
    )
)
"""Clip files from ``build_pairs.py``."""

WINDOW = os.environ.get("WINDOW", "0:2")
"""Solver-step window ``"start:end"`` into ``scheduler.timesteps``/``sigmas``,
as in :mod:`gate_mean_velocity`: ``z_first`` is built at ``sigmas[start]``
and the teacher's Euler roll covers steps ``[start, end)``."""

W_START, W_END = _parse_window(WINDOW)

CKPT = Path(
    os.environ.get(
        "CKPT",
        "integrations/hy_worldplay/drift_correction/outputs/"
        f"lora_v1_mean_w{W_START}{W_END}.pt",
    )
)
"""Output LoRA checkpoint; saved every ``SAVE_EVERY`` steps."""

INIT = os.environ.get("INIT", "")
"""Optional LoRA checkpoint to resume/init from."""

STEPS = int(os.environ.get("STEPS", "1500"))
LR = float(os.environ.get("LR", "5e-4"))
WARMUP = 60
"""Linear LR warmup steps; required for stability (reference finding)."""

GRAD_CLIP = 1.0
RANK = 16
EVAL_EVERY = 100
SAVE_EVERY = 250
CLEAN_LAP = 1
SEED = int(os.environ.get("SEED", "0"))


def checkpoint_blocks(network) -> None:
    """Route block ``forward`` and ``prefill_memory_kv`` through checkpointing.

    Per-instance overrides (not wrapper modules) so the network loop's
    ``isinstance(block, Block)`` assertion keeps passing. Requires the
    functional-attention patch: forward recomputation must be side-effect
    free. The prefill's ``write_rope`` / ``write_prope`` side effects are
    plain re-assignments of deterministically recomputed tensors, so its
    recompute is value-stable. No-op under ``no_grad`` passes (teacher/base
    window integrations, prefills).
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
        block.prefill_memory_kv = wrap(block.prefill_memory_kv)


def main() -> None:
    clips = sorted(PAIRS_DIR.glob("clip_*.pt"))
    assert clips, f"no clips under {PAIRS_DIR}; run build_pairs.py first"
    rng = np.random.default_rng(SEED)

    meta = torch.load(clips[0], map_location="cpu", weights_only=False)
    runner = build_runner(
        num_chunk=meta["num_chunk"],
        pose=meta["pose"],
        output_dir=CKPT.parent,
        compile_network=False,
    )
    pipe = runner.pipeline
    device = next(pipe.parameters()).device
    dtype = next(pipe.parameters()).dtype
    transformer = cast(HyWorldPlayWan21Transformer, pipe.diffusion_model.transformer)
    scheduler = pipe.diffusion_model.scheduler
    timesteps = cast(Tensor, scheduler.timesteps)
    sigmas = cast(Tensor, scheduler.sigmas)
    assert W_END < len(sigmas), (
        f"WINDOW {WINDOW!r} out of range for {len(sigmas)} sigmas"
    )
    sigma_start = sigmas[W_START].to(dtype)
    t_hi, t_lo = int(timesteps[W_START]), int(timesteps[W_END])

    cfg = runner.config
    assert cfg.image_path is not None  # build_runner resolves the sample image
    image = preprocess_first_frame(
        cfg.image_path, cfg.pixel_height, cfg.pixel_width
    ).to(device=device, dtype=dtype)
    cache = pipe.initialize_cache(text=[_resolve_prompt(cfg.prompt)], image=image)
    tc = cache.transformer_cache
    assert isinstance(tc, HyWorldPlayWan21TransformerCache)

    network = unwrap_compiled(transformer.network)
    wrapped = apply_lora(network, rank=RANK)
    if INIT:
        load_lora(network, INIT)
        print(f"LoRA init from {INIT}", flush=True)
    # Functional attention: restores k/v gradients (the stock path severs
    # them at the rolling-cache write) and makes per-block checkpointing
    # sound (no mutable storage in the tape -- the fp32 residual/dual-branch
    # tape is ~40 GiB unchunked).
    patch_functional_attention()
    checkpoint_blocks(network)
    params = lora_parameters(network)
    n_params = sum(p.numel() for p in params)
    print(
        f"LoRA on {len(wrapped)} projections | {n_params / 1e6:.2f}M params", flush=True
    )
    opt = torch.optim.AdamW(params, lr=LR)

    # Clip latents stay on CPU; a draw moves one clip's tensors to device.
    datas = [load_clip(p, "cpu", dtype) for p in clips]
    nval = max(2, len(datas) // 8)
    train_ids = list(range(len(datas) - nval))
    val_ids = list(range(len(datas) - nval, len(datas)))
    lap_chunks = datas[0]["lap_latents"] // 4
    num_chunk = datas[0]["num_chunk"]
    # Windows fully past the clean lap: selected recent-16 frames all drifted.
    ks = list(range((CLEAN_LAP + 2) * lap_chunks, num_chunk))
    print(
        f"{len(datas)} clips ({len(val_ids)} val) | k in {ks[0]}..{ks[-1]}"
        f" | window [{W_START}:{W_END}) u_mean {t_hi}->{t_lo}",
        flush=True,
    )

    def predict_u_student(ctrl, z_first: Tensor) -> Tensor:
        """One flow forward at ``timesteps[W_START]`` -> fp32 mean velocity.

        The student's one-jump mean velocity from ``sigmas[W_START]`` is its
        flow prediction itself (see the module docstring).
        """
        flow = transformer.predict_flow(
            noisy_latent=z_first,
            timestep=timesteps[W_START].to(dtype),
            cache=tc,
            input=ctrl,
        )
        return flow.float()

    def prefill_only(ctrl, history: Tensor, k: int) -> None:
        """Open the chunk bracket and run the memory prefill without grad."""
        start_probe_chunk(tc, ar_idx=k, history=history)
        with torch.no_grad():
            if ctrl.memory_frame_indices:
                transformer.prefill_memory_kv_cache(
                    cache=tc, input=ctrl, timestep=timesteps[0].to(dtype)
                )
                tc.prefill_completed_for_chunk = k

    def sample_losses(c: int, grad: bool) -> tuple[Tensor, Tensor]:
        """One drift-pair sample -> (normalized loss, drift-gap sq-norm).

        Teacher and base integrate the frozen model's Euler roll over the
        window (:func:`gate_mean_velocity.integrate_u_mean`, no grad, LoRA
        scale 0); the student does one graded forward on the shared
        ``z_first``. In grad mode the student's bracket stays open for the
        caller: backward must precede ``finish_probe_chunk`` (checkpoint
        recomputation needs the chunk's KV state alive).
        """
        d = {
            key: (v.to(device) if isinstance(v, Tensor) else v)
            for key, v in datas[c].items()
        }
        k = int(rng.choice(ks))
        selected = d["memory_frame_indices"][k]
        ctrl = make_ctrl(d, k, device=device, dtype=dtype)
        h_gen = history_of(d, k)
        h_clean = clean_counterfactual(
            h_gen, selected=selected, lap_latents=d["lap_latents"], clean_lap=CLEAN_LAP
        )
        x0 = chunk_x0(d, k)
        z_first = (1 - sigma_start) * x0 + sigma_start * torch.randn(
            x0.shape, device=device, dtype=dtype
        )

        def u_window(history: Tensor) -> Tensor:
            """Frozen-model window integration under ``history`` (one bracket)."""
            prefill_only(ctrl, history, k)
            u = integrate_u_mean(
                transformer,
                tc,
                ctrl=ctrl,
                z_t=z_first,
                timesteps=timesteps,
                sigmas=sigmas,
                dtype=dtype,
                start=W_START,
                end=W_END,
            )
            finish_probe_chunk(tc, ar_idx=k)
            return u

        with torch.no_grad():
            set_lora_scale(network, 0.0)
            u_clean = u_window(h_clean)
            u_base = u_window(h_gen)
        r_u_sq = (u_clean - u_base).square().sum()

        set_lora_scale(network, 1.0)
        prefill_only(ctrl, h_gen, k)  # LoRA-scaled prefill, no grad
        with torch.enable_grad() if grad else torch.no_grad():
            u_student = predict_u_student(ctrl, z_first)
            loss = (u_student - u_clean).square().sum() / (r_u_sq + 1e-8)
        if not grad:
            finish_probe_chunk(tc, ar_idx=k)
        return loss, r_u_sq

    @torch.no_grad()
    def val_r2(n: int = 8) -> float:
        s = 0.0
        for _ in range(n):
            loss, _ = sample_losses(int(rng.choice(val_ids)), grad=False)
            s += loss.item()
        return 1 - s / n

    torch.set_grad_enabled(True)
    for step in range(1, STEPS + 1):
        for pg in opt.param_groups:
            pg["lr"] = LR * min(1.0, step / WARMUP)
        opt.zero_grad()
        loss, ru_sq = sample_losses(int(rng.choice(train_ids)), grad=True)
        loss.backward()
        # Backward precedes the bracket close: checkpoint recomputation
        # needs the chunk's KV state alive.
        finish_probe_chunk(tc, ar_idx=tc.autoregressive_index)
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        opt.step()
        if step % EVAL_EVERY == 0 or step == 1:
            print(
                f"step {step:5d} | loss {loss.item():.4f} (train R^2 {1 - loss.item():+.3f})"
                f" | val R^2 {val_r2():+.3f} | |r_u|^2 {ru_sq.item():.1f}",
                flush=True,
            )
        if step % SAVE_EVERY == 0 or step == STEPS:
            save_lora(network, CKPT)

    print(
        f"TRAIN-V1-MEAN-DONE | window [{W_START}:{W_END}) u_mean {t_hi}->{t_lo}"
        f" | final val R^2 {val_r2(16):+.3f} | saved {CKPT}",
        flush=True,
    )


if __name__ == "__main__":
    main()

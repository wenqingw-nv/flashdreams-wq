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

"""Integrated mean-velocity gate: alpha* on the full-window Euler prediction.

:mod:`gate_faithful` decomposes the *single-step* x0 gap at each inference
timestep. This gate asks the same question about the quantity a mean-velocity
corrector would actually match: the average velocity of the full 1000->0
Euler integration. Each probe builds the noisy state at the top timestep
only, rolls it through every scheduler step under the drifted vs lap-aligned
clean history (same protocol as :mod:`gate_faithful`, same clips), and the
``u_mean = (z_t - x0_hat) / sigma_0`` gap is decomposed over noise seeds into
systematic bias vs variance -- one integrated "timestep row" instead of one
per step. The summary prints the per-step t=1000 reference from
``gate_faithful.json`` (when present) next to the integrated alpha* for
direct comparison.

Run after ``build_pairs.py`` from the repo root::

    .venv/bin/python integrations/hy_worldplay/drift_correction/gate_mean_velocity.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import torch
from _pairs import (
    chunk_x0,
    clean_counterfactual,
    history_of,
    load_clip,
    make_ctrl,
)
from _rollout import build_runner, finish_probe_chunk, start_probe_chunk
from hy_worldplay._action import HyWorldPlayCtrl, HyWorldPlayWan21TransformerCache
from hy_worldplay.runner import _resolve_prompt, preprocess_first_frame
from torch import Tensor

## Gate configuration

PAIRS_DIR = Path("integrations/hy_worldplay/drift_correction/outputs/pairs")
"""Clip files from ``build_pairs.py``."""

OUT_PATH = Path(
    "integrations/hy_worldplay/drift_correction/outputs/gate/gate_mean_velocity.json"
)
"""Aggregated integrated-window result."""

FAITHFUL_PATH = Path(
    "integrations/hy_worldplay/drift_correction/outputs/gate/gate_faithful.json"
)
"""Per-step reference from :mod:`gate_faithful`, printed alongside when present."""

PROBE_CHUNKS = (12, 16, 20, 23)
"""Late chunks probed per clip (laps 3+ of a 6-lap / 24-chunk rollout)."""

M_NOISE = 8
"""Noise seeds per (clip, chunk) cell."""

CLEAN_LAP = 1
"""Lap supplying the clean teacher content. Lap 0 is even cleaner but its
first frame is the stamped real-image latent (different distribution)."""

PER_STEP_T1000_REF = 0.81
"""Unbiased alpha* at t=1000 from the faithful per-step gate; the GO bar
(the integrated gap should be at least as systematic as the single-step
reference)."""


def integrate_u_mean(
    transformer: Any,
    tc: HyWorldPlayWan21TransformerCache,
    *,
    ctrl: HyWorldPlayCtrl,
    z_t: Tensor,
    timesteps: Tensor,
    sigmas: Tensor,
    dtype: torch.dtype,
) -> Tensor:
    """Mean velocity of the full Euler integration from ``z_t`` down to x0.

    Runs every scheduler step under the cache's current history
    (``sigmas[-1] == 0``, so ``z`` lands on the integrated ``x0_hat``). All
    forwards stay inside the caller's probe bracket -- the bracket supports
    repeated ``predict_flow`` calls at the same chunk.

    Returns:
        fp32 ``u_mean = (z_t - x0_hat) / sigma_0``.
    """
    z = z_t
    for i in range(len(timesteps) - 1):
        flow = transformer.predict_flow(
            noisy_latent=z, timestep=timesteps[i].to(dtype), cache=tc, input=ctrl
        )
        z = z + float(sigmas[i + 1] - sigmas[i]) * flow
    return (z_t.float() - z.float()) / float(sigmas[0])


def main() -> None:
    torch.set_grad_enabled(False)
    clips = sorted(PAIRS_DIR.glob("clip_*.pt"))
    assert clips, f"no clips under {PAIRS_DIR}; run build_pairs.py first"

    meta = torch.load(clips[0], map_location="cpu", weights_only=False)
    runner = build_runner(
        num_chunk=meta["num_chunk"], pose=meta["pose"], output_dir=PAIRS_DIR
    )
    pipe = runner.pipeline
    device = next(pipe.parameters()).device
    dtype = next(pipe.parameters()).dtype
    cfg = runner.config
    assert cfg.image_path is not None  # build_runner resolves the sample image
    image = preprocess_first_frame(
        cfg.image_path, cfg.pixel_height, cfg.pixel_width
    ).to(device=device, dtype=dtype)
    cache = pipe.initialize_cache(text=[_resolve_prompt(cfg.prompt)], image=image)
    tc = cache.transformer_cache
    assert isinstance(tc, HyWorldPlayWan21TransformerCache)
    transformer = pipe.diffusion_model.transformer
    scheduler = pipe.diffusion_model.scheduler
    timesteps = cast(Tensor, scheduler.timesteps)
    sigmas = cast(Tensor, scheduler.sigmas)
    sigma_0 = sigmas[0].to(dtype)

    # Accumulate squared-bias / variance / norms for the single integrated
    # window across all (clip, chunk) cells.
    agg: dict[str, list[float]] = {"alphas": [], "alphas_ub": [], "rels": []}
    for clip_path in clips:
        d = load_clip(clip_path, device, dtype)
        lap = d["lap_latents"]
        for k in PROBE_CHUNKS:
            if k >= d["num_chunk"]:
                continue
            selected = d["memory_frame_indices"][k]
            if not selected:
                continue
            ctrl = make_ctrl(d, k, device=device, dtype=dtype)
            h_gen = history_of(d, k)
            h_clean = clean_counterfactual(
                h_gen, selected=selected, lap_latents=lap, clean_lap=CLEAN_LAP
            )
            x0 = chunk_x0(d, k)

            z_ts: list[Tensor] = []
            for m in range(M_NOISE):
                g = torch.Generator(device=device).manual_seed(700_000 + 10_000 * k + m)
                eps = torch.randn(x0.shape, device=device, dtype=dtype, generator=g)
                z_ts.append((1 - sigma_0) * x0 + sigma_0 * eps)

            preds: dict[str, list[Tensor]] = {}
            for name, h in (("gen", h_gen), ("clean", h_clean)):
                start_probe_chunk(tc, ar_idx=k, history=h)
                preds[name] = [
                    integrate_u_mean(
                        transformer,
                        tc,
                        ctrl=ctrl,
                        z_t=z,
                        timesteps=timesteps,
                        sigmas=sigmas,
                        dtype=dtype,
                    )
                    for z in z_ts
                ]
                finish_probe_chunk(tc, ar_idx=k)

            deltas = torch.stack([a - b for a, b in zip(preds["gen"], preds["clean"])])
            m = deltas.shape[0]
            bias = deltas.mean(dim=0)
            bias_sq = bias.square().sum().item()
            sq_dev = (deltas - bias).square().sum(dim=tuple(range(1, deltas.ndim)))
            var = sq_dev.mean().item()
            var_ub = sq_dev.sum().item() / (m - 1)
            bias_sq_ub = max(0.0, bias_sq - var_ub / m)
            gen_norm = torch.stack(preds["gen"]).flatten(1).norm(dim=1).mean()
            rel = (deltas.flatten(1).norm(dim=1).mean() / (gen_norm + 1e-12)).item()
            a = bias_sq / (bias_sq + var + 1e-12)
            a_ub = bias_sq_ub / (bias_sq_ub + var_ub + 1e-12)
            agg["alphas"].append(a)
            agg["alphas_ub"].append(a_ub)
            agg["rels"].append(rel)
            print(
                f"{clip_path.stem} k={k:2d} | u_mean 1000->0"
                f" a*={a:.3f}/{a_ub:.3f} rel={rel:.3f}",
                flush=True,
            )

    assert agg["alphas"], f"no probe cells among {PROBE_CHUNKS} produced predictions"
    n_cells = len(agg["alphas"])
    alpha = sum(agg["alphas"]) / n_cells
    alpha_ub = sum(agg["alphas_ub"]) / n_cells
    mean_rel = sum(agg["rels"]) / n_cells
    summary = {
        "alpha_star": alpha,
        "alpha_star_unbiased": alpha_ub,
        "rel": mean_rel,
        "cells": n_cells,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))

    print("\n====== HY-WorldPlay MEAN-VELOCITY gate (integrated 1000->0) ======")
    print(
        f"integrated: alpha* {alpha:.3f} (unbiased {alpha_ub:.3f})"
        f" | rel drift gap {mean_rel:.3f} | {n_cells} cells"
    )
    ref = None
    if FAITHFUL_PATH.exists():
        ref = json.loads(FAITHFUL_PATH.read_text()).get("1000")
    if ref is not None:
        print(
            f"per-step t=1000: alpha* {ref['alpha_star']:.3f}"
            f" (unbiased {ref['alpha_star_unbiased']:.3f})"
            f" vs integrated {alpha:.3f} (unbiased {alpha_ub:.3f})"
        )
    else:
        print(
            f"per-step reference {FAITHFUL_PATH} missing;"
            f" comparing against the {PER_STEP_T1000_REF} constant"
        )
    if mean_rel < 0.01:
        print("RESULT: STOP: no integrated drift signal")
    elif alpha_ub >= PER_STEP_T1000_REF:
        print(
            "RESULT: GO: integrated gap at least as systematic as per-step"
            " (hypothesis: >=0.95)"
        )
    else:
        print("RESULT: below the per-step t=1000 reference -- report before training.")
    print(f"saved {OUT_PATH}")
    print("MEAN-GATE-DONE", flush=True)


if __name__ == "__main__":
    main()

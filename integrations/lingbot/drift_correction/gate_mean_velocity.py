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

"""Integrated mean-velocity gate: alpha* on the full renoise-loop prediction.

:mod:`gate_faithful` decomposes the *single-step* x0 gap at each solver
timestep. This gate asks the same question about the quantity a mean-velocity
corrector would actually match: the average velocity of the host's full
denoising loop. LingBot's ``FlowMatchScheduler.sample`` does not Euler-step
between timesteps -- it re-noises the running x0 estimate at every step
(``z = (1 - sigma_next) * x0_hat + sigma_next * fresh_noise``) -- so the
probe replays exactly that loop: build ``z`` at ``sigmas[start]``, roll it
through solver steps ``[start, end)`` under the drifted vs lap-aligned clean
history (same clips + protocol as :mod:`gate_faithful`; the per-step renoise
draws are fixed per noise seed and shared by both histories), and decompose
the gap of ``u_mean = (z_start - x0_hat) / sigmas[start]`` over noise seeds
into systematic bias vs variance -- one integrated "timestep row" instead of
one per step. The summary prints the per-step t=999 reference from
``gate_faithful.json`` (when present) next to the integrated alpha* for
direct comparison.

``WINDOW="start:end"`` (default ``"0:4"``, solver-step indices into
``scheduler.denoising_step_list``/``denoising_sigmas``) narrows the probe:
``z`` is built at ``sigmas[start]`` and the renoise loop covers steps
``[start, end)``. The final ``x0_hat`` is always an x0 estimate, so
``u_mean`` always measures the mean velocity from ``sigmas[start]`` down to
0. Non-default windows write ``gate_mean_velocity_w{start}{end}.json``
instead.

Run after ``build_pairs.py`` from the repo root::

    .venv/bin/python integrations/lingbot/drift_correction/gate_mean_velocity.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from _rollout import (
    _encode_ctrl,
    build_pipeline,
    camera_stream,
    chunk_camctrl,
    load_scene,
    rebuild_cache,
)
from gate_faithful import clean_counterpart_history, lap_of
from lingbot.encoder.camctrl import CamCtrlInput
from lingbot.pipeline import LingbotWorldInferencePipeline
from torch import Tensor

## Gate configuration

PAIRS_DIR = Path("integrations/lingbot/drift_correction/outputs/pairs")
"""Clip files from ``build_pairs.py``."""

OUT_PATH = Path(
    "integrations/lingbot/drift_correction/outputs/gate/gate_mean_velocity.json"
)
"""Aggregated integrated-window result (default window; other windows write
``gate_mean_velocity_w{start}{end}.json`` alongside)."""

WINDOW = os.environ.get("WINDOW", "0:4")
"""Solver-step window ``"start:end"`` into
``scheduler.denoising_step_list``/``denoising_sigmas``. ``z`` is built at
``sigmas[start]`` and the renoise loop covers steps ``[start, end)``; the
default covers the full 4-step loop."""

FAITHFUL_PATH = Path(
    "integrations/lingbot/drift_correction/outputs/gate/gate_faithful.json"
)
"""Per-step reference from :mod:`gate_faithful`, printed alongside when present."""

M_NOISE = int(os.environ.get("M_NOISE", "8"))
"""Noise seeds per (clip, chunk) cell."""

PROBE_LAPS = tuple(int(s) for s in os.environ.get("PROBE_LAPS", "3,4,5,6").split(","))
"""Laps whose first chunk is probed (lap 1 = clean reference, lap 2 warmup)."""

PER_STEP_T999_REF = 0.878
"""Unbiased alpha* at t=999 from the faithful per-step gate; the GO bar
(the integrated gap should be at least as systematic as the single-step
reference)."""


def _parse_window(spec: str) -> tuple[int, int]:
    """Parse ``WINDOW`` into (start, end) solver-step indices; end exclusive."""
    start_s, _, end_s = spec.partition(":")
    start, end = int(start_s), int(end_s)
    assert 0 <= start < end, f"bad WINDOW {spec!r}: need 0 <= start < end"
    return start, end


def draw_eps(x0: Tensor, k: int, w_start: int, n: int, m: int) -> list[Tensor]:
    """The ``n``-deep renoise stack for noise seed ``m`` at chunk ``k``.

    Seeded per (chunk, window, seed) so the gen and clean histories see
    identical draws: ``eps[0]`` builds the initial ``z`` at
    ``sigmas[w_start]``, ``eps[1:]`` feed the between-step renoises.
    """
    g = torch.Generator().manual_seed(700_000 + 10_000 * k + 1_000 * w_start + m)
    return [torch.randn(x0.shape, generator=g) for _ in range(n)]


def probe_u_mean(
    pipe: LingbotWorldInferencePipeline,
    cache,
    ar_index: int,
    ctrl: CamCtrlInput,
    x0: Tensor,
    eps: list[Tensor],
    start: int,
    end: int,
) -> Tensor:
    """Mean velocity of the renoise loop over solver steps ``[start, end)``.

    Modeled on :func:`_rollout.probe_xhat0` (same encoder reuse via
    ``cache._probe_enc``, same anchoring math), but instead of one forward it
    replays the host ``FlowMatchScheduler.sample`` loop on a rebuilt cache:
    each step converts the predicted flow to ``x0_hat = z - sigma_i * flow``,
    then re-noises to the next step's sigma with the pre-drawn ``eps``
    (``z = (1 - sigma_next) * x0_hat + sigma_next * eps[j + 1]``). The probe
    never finalizes ``ar_index`` -- call repeatedly with different ``eps`` on
    the same cache; discard the cache afterwards.

    Args:
        pipe: Pipeline from :func:`_rollout.build_pipeline`.
        cache: Cache from :func:`_rollout.rebuild_cache` (histories written).
        ar_index: Probe chunk position (== number of teacher-forced chunks).
        ctrl: Camera payload of the probe chunk.
        x0: Unpatchified clean latent of the probe chunk (any device).
        eps: ``end - start`` unpatchified ``x0``-shaped noise tensors from
            :func:`draw_eps`.
        start: First solver step of the window (inclusive).
        end: Last solver step of the window (exclusive).

    Returns:
        Unpatchified fp32 ``u_mean = (z_start - x0_hat) / sigmas[start]``
        (cpu).
    """
    dm = pipe.diffusion_model
    transformer = dm.transformer
    scheduler = dm.scheduler
    sigmas = scheduler.denoising_sigmas
    timesteps = scheduler.denoising_step_list
    assert len(eps) == end - start, f"need {end - start} noises, got {len(eps)}"

    if not hasattr(cache, "_probe_enc"):
        cache._probe_enc = {}
    if ar_index not in cache._probe_enc:
        cache._probe_enc[ar_index] = _encode_ctrl(pipe, cache, ar_index, ctrl)
        cache.transformer_cache.start(ar_index)
    enc_p = cache._probe_enc[ar_index]

    x0 = x0.to(dm.device, torch.float32)
    sigma_start = sigmas[start]
    z = (
        (1.0 - sigma_start) * x0 + sigma_start * eps[0].to(dm.device, torch.float32)
    ).to(dm.dtype)
    z_first = z
    x0_hat = z.float()  # overwritten on step 0; start < end is guaranteed
    for j, i in enumerate(range(start, end)):
        z_p = transformer.patchify_and_maybe_split_cp(z)
        flow = transformer.predict_flow(
            noisy_latent=z_p,
            timestep=timesteps[i].to(dtype=dm.dtype),
            cache=cache.transformer_cache,
            input=enc_p,
        )
        flow_u = transformer.unpatchify_and_maybe_gather_cp(flow)
        x0_hat = z.float() - float(sigmas[i]) * flow_u.float()
        if i + 1 < end:
            sigma_next = sigmas[i + 1]
            z = (
                (1.0 - sigma_next) * x0_hat
                + sigma_next * eps[j + 1].to(dm.device, torch.float32)
            ).to(dm.dtype)
    u_mean = (z_first.float() - x0_hat) / float(sigma_start)
    return u_mean.detach().cpu()


def main() -> None:
    torch.set_grad_enabled(False)
    w_start, w_end = _parse_window(WINDOW)
    out_path = OUT_PATH
    if (w_start, w_end) != (0, 4):
        out_path = OUT_PATH.with_name(f"gate_mean_velocity_w{w_start}{w_end}.json")
    clips = sorted(PAIRS_DIR.glob("clip_*.pt"))
    assert clips, f"no clips under {PAIRS_DIR}; run build_pairs.py first"
    pipe = build_pipeline()
    scheduler = pipe.diffusion_model.scheduler
    timesteps = scheduler.denoising_step_list
    n_steps = timesteps.shape[0]
    assert w_end <= n_steps, (
        f"WINDOW {WINDOW!r} out of range for {n_steps} solver steps"
    )
    t_hi = int(timesteps[w_start])

    # Accumulate squared-bias / variance / norms for the single integrated
    # window across all (clip, chunk) cells.
    agg: dict[str, list[float]] = {"alphas": [], "alphas_ub": [], "rels": []}
    for clip_path in clips:
        d = torch.load(clip_path, map_location="cpu", weights_only=False)
        scene = load_scene(d["scene_idx"])
        poses_t, intr_t, ws = camera_stream(scene, d["poses"])
        camctrls = chunk_camctrl(poses_t, intr_t, ws, pipe, n_chunks=d["num_chunk"])
        lat = d["latents"]
        lap_chunks = d["lap_chunks"]

        probe_chunks = [
            1 + (j - 1) * lap_chunks
            for j in PROBE_LAPS
            if 1 + (j - 1) * lap_chunks < d["num_chunk"]
        ]
        for k in probe_chunks:
            x0 = lat[k]
            eps_lists = [
                draw_eps(x0, k, w_start, w_end - w_start, m) for m in range(M_NOISE)
            ]

            preds: dict[str, list[Tensor]] = {}
            for name, hist in (
                ("gen", lat[:k]),
                ("clean", clean_counterpart_history(lat, k, lap_chunks)),
            ):
                cache = rebuild_cache(pipe, scene, hist, camctrls[:k])
                preds[name] = [
                    probe_u_mean(
                        pipe,
                        cache,
                        k,
                        camctrls[k],
                        x0,
                        eps_lists[m],
                        start=w_start,
                        end=w_end,
                    )
                    for m in range(M_NOISE)
                ]
                del cache
                torch.cuda.empty_cache()

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
                f"{clip_path.stem} k={k:2d} lap={lap_of(k, lap_chunks)}"
                f" | u_mean {t_hi}->0 a*={a:.3f}/{a_ub:.3f} rel={rel:.3f}",
                flush=True,
            )

    assert agg["alphas"], f"no probe cells among laps {PROBE_LAPS} produced predictions"
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\n====== LingBot MEAN-VELOCITY gate (integrated {t_hi}->0) ======")
    print(
        f"integrated: alpha* {alpha:.3f} (unbiased {alpha_ub:.3f})"
        f" | rel drift gap {mean_rel:.3f} | {n_cells} cells"
    )
    ref = None
    if FAITHFUL_PATH.exists():
        ref = json.loads(FAITHFUL_PATH.read_text()).get("999")
    if ref is not None:
        print(
            f"per-step t=999: alpha* {ref['alpha_star']:.3f}"
            f" (unbiased {ref['alpha_star_unbiased']:.3f})"
            f" vs integrated {alpha:.3f} (unbiased {alpha_ub:.3f})"
        )
    else:
        print(
            f"per-step reference {FAITHFUL_PATH} missing;"
            f" comparing against the {PER_STEP_T999_REF} constant"
        )
    if mean_rel < 0.01:
        print("RESULT: STOP: no integrated drift signal")
    elif alpha_ub >= PER_STEP_T999_REF:
        print(
            "RESULT: GO: integrated gap at least as systematic as per-step"
            " (hypothesis: >=0.95)"
        )
    else:
        print("RESULT: below the per-step t=999 reference -- report before training.")
    print(f"saved {out_path}")
    print("MEAN-GATE-DONE", flush=True)


if __name__ == "__main__":
    main()

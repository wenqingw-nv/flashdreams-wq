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

"""Faithful step-0 gate: alpha*(t) on LingBot strafe-loop drift pairs.

At a late-lap chunk of a :mod:`build_pairs` rollout, the history is
teacher-forced into a fresh KV cache twice — once with the rollout's own
(drifted) chunk latents, once with every lap >= 2 chunk swapped for its
lap-1 counterpart (same camera poses, same RoPE positions, clean content) —
and the x0-prediction gap at matched ``z_t`` is decomposed over noise seeds
into systematic bias vs variance (``alpha* = |bias|^2 / (|bias|^2 + var)``).
Also reports the relative drift-gap magnitude ``rel`` (the training-loss
denominator); near-zero rel means there is nothing to correct.

Kill bars (pre-stated, per the cross-host playbook): mean rel < 0.01 ->
STOP; fewer than 70% of cells with unbiased alpha* >= 0.7 -> report to the
owner before any training.

Run after ``build_pairs.py`` from the repo root::

    .venv/bin/python integrations/lingbot/drift_correction/gate_faithful.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from _rollout import (
    build_pipeline,
    camera_stream,
    chunk_camctrl,
    load_scene,
    probe_xhat0,
    rebuild_cache,
)

PAIRS_DIR = Path("integrations/lingbot/drift_correction/outputs/pairs")
OUT_PATH = Path(
    "integrations/lingbot/drift_correction/outputs/gate/gate_faithful.json"
)

M_NOISE = int(os.environ.get("M_NOISE", "8"))
"""Noise seeds per (clip, chunk, timestep) cell."""

PROBE_LAPS = tuple(int(s) for s in os.environ.get("PROBE_LAPS", "3,4,5,6").split(","))
"""Laps whose first chunk is probed (lap 1 = clean reference, lap 2 warmup)."""


def lap_of(chunk: int, lap_chunks: int) -> int:
    """1-based lap index of ``chunk`` (chunk 0 is the lead-in, lap 0)."""
    return 0 if chunk == 0 else 1 + (chunk - 1) // lap_chunks


def clean_counterpart_history(
    latents: list[torch.Tensor], upto: int, lap_chunks: int
) -> list[torch.Tensor]:
    """History ``0..upto-1`` with every lap >= 2 chunk swapped to lap 1."""
    out = []
    for i in range(upto):
        if lap_of(i, lap_chunks) >= 2:
            out.append(latents[1 + (i - 1) % lap_chunks])
        else:
            out.append(latents[i])
    return out


def main() -> None:
    torch.set_grad_enabled(False)
    clips = sorted(PAIRS_DIR.glob("clip_*.pt"))
    assert clips, f"no clips under {PAIRS_DIR}; run build_pairs.py first"
    pipe = build_pipeline()
    scheduler = pipe.diffusion_model.scheduler
    timesteps = scheduler.denoising_step_list
    n_steps = timesteps.shape[0]

    agg = {i: {"alphas": [], "alphas_ub": [], "rels": []} for i in range(n_steps)}
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
            eps = [
                torch.randn(
                    x0.shape,
                    generator=torch.Generator().manual_seed(
                        900_000 + 10_000 * k + m
                    ),
                )
                for m in range(M_NOISE)
            ]
            preds: dict[str, dict[int, list[torch.Tensor]]] = {}
            for name, hist in (
                ("gen", lat[:k]),
                ("clean", clean_counterpart_history(lat, k, lap_chunks)),
            ):
                cache = rebuild_cache(pipe, scene, hist, camctrls[:k])
                preds[name] = {
                    t_idx: [
                        probe_xhat0(
                            pipe, cache, k, camctrls[k], x0, t_idx, eps[m]
                        )
                        for m in range(M_NOISE)
                    ]
                    for t_idx in range(n_steps)
                }
                del cache
                torch.cuda.empty_cache()

            line = [f"{clip_path.stem} k={k:2d} lap={lap_of(k, lap_chunks)}"]
            for t_idx in range(n_steps):
                deltas = torch.stack(
                    [a - b for a, b in zip(preds["gen"][t_idx], preds["clean"][t_idx])]
                )
                m = deltas.shape[0]
                bias = deltas.mean(dim=0)
                bias_sq = bias.square().sum().item()
                sq_dev = (deltas - bias).square().sum(dim=tuple(range(1, deltas.ndim)))
                var = sq_dev.mean().item()
                var_ub = sq_dev.sum().item() / (m - 1)
                bias_sq_ub = max(0.0, bias_sq - var_ub / m)
                gen_norm = (
                    torch.stack(preds["gen"][t_idx]).flatten(1).norm(dim=1).mean()
                )
                rel = (deltas.flatten(1).norm(dim=1).mean() / (gen_norm + 1e-12)).item()
                a = bias_sq / (bias_sq + var + 1e-12)
                a_ub = bias_sq_ub / (bias_sq_ub + var_ub + 1e-12)
                agg[t_idx]["alphas"].append(a)
                agg[t_idx]["alphas_ub"].append(a_ub)
                agg[t_idx]["rels"].append(rel)
                line.append(
                    f"t={int(timesteps[t_idx]):4d} a*={a:.3f}/{a_ub:.3f} rel={rel:.3f}"
                )
            print(" | ".join(line), flush=True)

    summary = {
        str(int(timesteps[i])): {
            "alpha_star": sum(v["alphas"]) / len(v["alphas"]),
            "alpha_star_unbiased": sum(v["alphas_ub"]) / len(v["alphas_ub"]),
            "rel": sum(v["rels"]) / len(v["rels"]),
            "cells": len(v["alphas"]),
        }
        for i, v in agg.items()
        if v["alphas"]
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))

    print("\n========== LingBot FAITHFUL gate (strafe-loop drift pairs) ==========")
    for t, v in summary.items():
        print(
            f"t={t:>4s}: alpha* {v['alpha_star']:.3f}"
            f" (unbiased {v['alpha_star_unbiased']:.3f})"
            f" | rel drift gap {v['rel']:.3f} | {v['cells']} cells"
        )
    all_ub = [a for v in agg.values() for a in v["alphas_ub"]]
    mean_rel = sum(r for v in agg.values() for r in v["rels"]) / max(
        1, sum(len(v["rels"]) for v in agg.values())
    )
    frac = sum(a >= 0.7 for a in all_ub) / len(all_ub)
    print(
        f"cells with unbiased alpha* >= 0.7: {frac:.0%} | mean rel gap {mean_rel:.3f}"
    )
    if mean_rel < 0.01:
        print("RESULT: drift gap ~zero -> nothing to correct. STOP.")
    elif frac >= 0.7:
        print("RESULT: real drift gap is systematic. GO for v1 training.")
    else:
        print(
            "RESULT: below the reference bar -- report before committing to training."
        )
    print(f"saved {OUT_PATH}")
    print("GATE-DONE", flush=True)


if __name__ == "__main__":
    main()

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

"""Oracle-teacher gate: does a future-informed teacher carry new signal here?

Paper-host result (2026-07-19): swapping the corrector's teacher for a
future-informed one broke the drift--progression frontier (Δ +0.12 at
anchoring 0.643 vs ~0.78 predicted). This gate measures the analog on
HY-WorldPlay before any training.

Host-native oracle (no bidirectional pass needed): the memory prefill is
pose-indexed, so the teacher becomes future-informed by ADDING memory
indices at the lap-``CLEAN_LAP`` frames whose poses match the chunks the
camera is ABOUT to visit. Those are past positions with clean content --
fully in-distribution; the only change is which poses the memory covers.

Per (clip, chunk, t), over ``M_NOISE`` seeds, three histories:
  gen     -- drifted memory (what the host does)
  clean   -- past teacher (current v1/v2 target)
  oracle  -- past teacher + future-pose lap-1 memory indices

GATE metrics (paper-host thresholds: alpha* >= ~0.7, info ~0.3--1):
  alpha*_clean / alpha*_oracle -- systematic fraction of each teacher gap
  info = ||mean(oracle) - mean(clean)|| / ||mean(clean - gen)||
  new-pose frac -- of the added future poses, how many were NOT already
     covered (mod lap) by the FOV selection; info ~ 0 with low new-pose
     frac means loop pairs already leak the future -> oracle needs
     non-loop pairs, not a teacher swap.

Run after ``build_pairs.py`` from the repo root::

    .venv/bin/python integrations/hy_worldplay/drift_correction/gate_oracle.py
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import torch
from torch import Tensor

from _pairs import chunk_x0, clean_counterfactual, history_of, load_clip, make_ctrl
from _rollout import build_runner, finish_probe_chunk, predict_x0, start_probe_chunk
from hy_worldplay._action import HyWorldPlayWan21TransformerCache
from hy_worldplay.runner import _resolve_prompt, preprocess_first_frame

## Gate configuration

PAIRS_DIR = Path("integrations/hy_worldplay/drift_correction/outputs/pairs")
OUT_PATH = Path(
    "integrations/hy_worldplay/drift_correction/outputs/gate/gate_oracle.json"
)

PROBE_CHUNKS = (12, 16, 20)
"""Late chunks probed per clip; the last chunk (23) has no future and is skipped."""

FUTURE_CHUNKS = 2
"""Chunks of lookahead (8 latents) whose lap-1 pose counterparts join the memory."""

M_NOISE = 8
CLEAN_LAP = 1


def oracle_indices(selected: list[int], k: int, lap: int) -> tuple[list[int], float]:
    """Past selection + lap-1 counterparts of the next ``FUTURE_CHUNKS`` chunks' poses.

    Returns the merged index list and the fraction of added future poses not
    already covered (mod lap) by the FOV selection.
    """
    future = range(4 * (k + 1), 4 * (k + 1 + FUTURE_CHUNKS))
    srcs = sorted({(f % lap) + CLEAN_LAP * lap for f in future})
    covered = {idx % lap for idx in selected}
    new_pose = [s for s in srcs if s % lap not in covered]
    merged = sorted(set(selected) | set(srcs))
    return merged, len(new_pose) / max(1, len(srcs))


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
    image = preprocess_first_frame(
        cfg.image_path, cfg.pixel_height, cfg.pixel_width
    ).to(device=device, dtype=dtype)
    cache = pipe.initialize_cache(text=[_resolve_prompt(cfg.prompt)], image=image)
    tc = cache.transformer_cache
    assert isinstance(tc, HyWorldPlayWan21TransformerCache)
    transformer = pipe.diffusion_model.transformer
    scheduler = pipe.diffusion_model.scheduler
    timesteps, sigmas = scheduler.timesteps, scheduler.sigmas
    n_steps = len(timesteps) - 1

    agg = {
        i: {"a_clean": [], "a_or": [], "info": [], "rel": []} for i in range(n_steps)
    }
    new_pose_fracs = []
    for clip_path in clips:
        d = load_clip(clip_path, device, dtype)
        lap = d["lap_latents"]
        for k in PROBE_CHUNKS:
            if k + FUTURE_CHUNKS >= d["num_chunk"]:
                continue
            selected = d["memory_frame_indices"][k]
            if not selected:
                continue
            merged, np_frac = oracle_indices(selected, k, lap)
            new_pose_fracs.append(np_frac)
            ctrl = make_ctrl(d, k, device=device, dtype=dtype)
            ctrl_or = dataclasses.replace(ctrl, memory_frame_indices=merged)
            h_gen = history_of(d, k)
            h_clean = clean_counterfactual(
                h_gen, selected=selected, lap_latents=lap, clean_lap=CLEAN_LAP
            )
            h_or = clean_counterfactual(
                h_gen, selected=merged, lap_latents=lap, clean_lap=CLEAN_LAP
            )
            x0 = chunk_x0(d, k)

            z_ts: dict[int, list[Tensor]] = {}
            for t_idx in range(n_steps):
                sig = sigmas[t_idx].to(dtype)
                z_ts[t_idx] = []
                for m in range(M_NOISE):
                    g = torch.Generator(device=device).manual_seed(
                        700_000 + 10_000 * k + 100 * t_idx + m
                    )
                    eps = torch.randn(x0.shape, device=device, dtype=dtype, generator=g)
                    z_ts[t_idx].append((1 - sig) * x0 + sig * eps)

            preds: dict[str, dict[int, list[Tensor]]] = {}
            for name, h, c in (
                ("gen", h_gen, ctrl),
                ("clean", h_clean, ctrl),
                ("oracle", h_or, ctrl_or),
            ):
                start_probe_chunk(tc, ar_idx=k, history=h)
                preds[name] = {
                    t_idx: [
                        predict_x0(
                            transformer,
                            tc,
                            ctrl=c,
                            z_t=z,
                            timestep=timesteps[t_idx].to(dtype),
                            sigma=float(sigmas[t_idx]),
                        )
                        for z in z_list
                    ]
                    for t_idx, z_list in z_ts.items()
                }
                finish_probe_chunk(tc, ar_idx=k)

            line = [f"{clip_path.stem} k={k:2d} newpose={np_frac:.2f}"]
            for t_idx in range(n_steps):

                def alpha(a: list[Tensor], b: list[Tensor]) -> float:
                    deltas = torch.stack([x - y for x, y in zip(a, b)])
                    bias_sq = deltas.mean(dim=0).square().sum().item()
                    var = (
                        (deltas - deltas.mean(dim=0))
                        .square()
                        .sum(dim=tuple(range(1, deltas.ndim)))
                        .mean()
                        .item()
                    )
                    return bias_sq / (bias_sq + var + 1e-12)

                p = preds
                gap = torch.stack(
                    [a - b for a, b in zip(p["clean"][t_idx], p["gen"][t_idx])]
                ).mean(dim=0)
                gnorm = gap.norm().item() + 1e-12
                info = (
                    torch.stack(p["oracle"][t_idx]).mean(dim=0)
                    - torch.stack(p["clean"][t_idx]).mean(dim=0)
                ).norm().item() / gnorm
                a_c = alpha(p["clean"][t_idx], p["gen"][t_idx])
                a_o = alpha(p["oracle"][t_idx], p["gen"][t_idx])
                agg[t_idx]["a_clean"].append(a_c)
                agg[t_idx]["a_or"].append(a_o)
                agg[t_idx]["info"].append(info)
                gen_norm = (
                    torch.stack(p["gen"][t_idx]).flatten(1).norm(dim=1).mean().item()
                )
                agg[t_idx]["rel"].append(gnorm / (gen_norm + 1e-12))
                line.append(
                    f"t={int(timesteps[t_idx]):4d} a_cl={a_c:.3f} a_or={a_o:.3f} info={info:.3f}"
                )
            print(" | ".join(line), flush=True)

    summary = {
        str(int(timesteps[i])): {
            key: sum(v[key]) / len(v[key]) for key in ("a_clean", "a_or", "info", "rel")
        }
        for i, v in agg.items()
        if v["a_clean"]
    }
    summary["new_pose_frac"] = sum(new_pose_fracs) / max(1, len(new_pose_fracs))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))

    print("\n========== HY-WorldPlay ORACLE-TEACHER gate ==========")
    for t, v in summary.items():
        if t == "new_pose_frac":
            continue
        print(
            f"t={t:>4s}: a*_clean {v['a_clean']:.3f} | a*_oracle {v['a_or']:.3f}"
            f" | info {v['info']:.3f} | rel {v['rel']:.3f}"
        )
    print(f"GATE new-pose frac (future poses not already FOV-covered): "
          f"{summary['new_pose_frac']:.2f}")
    infos = [x for v in agg.values() for x in v["info"]]
    a_ors = [x for v in agg.values() for x in v["a_or"]]
    mean_info = sum(infos) / max(1, len(infos))
    mean_a = sum(a_ors) / max(1, len(a_ors))
    if mean_info < 0.05:
        print("RESULT: oracle adds ~nothing over the past teacher on loop pairs "
              "(future poses already leak via FOV memory) -> teacher swap alone "
              "will not move the bridge; needs non-loop pairs. STOP.")
    elif mean_a >= 0.6:
        print("RESULT: future-informed gap is systematic and informative. "
              "GO for v3b-HY training.")
    else:
        print("RESULT: oracle target too noisy on this host -- report before training.")
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()

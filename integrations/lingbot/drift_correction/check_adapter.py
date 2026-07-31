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

"""Adapter mechanics check: replay determinism + history sensitivity.

Three assertions before any gate/pair spend:
1. Rebuilding the same history twice and probing with the same ``(t, eps)``
   is deterministic (bitwise-equal ``xhat0``).
2. Substituting one history chunk changes the probe output (history flows
   through the teacher-forced KV replay).
3. The probe reconstructs the rollout's own chunk reasonably at the lowest
   solver noise level (loose sanity bound, not a kill bar).

Run from the flashdreams repo root::

    .venv/bin/python integrations/lingbot/drift_correction/check_adapter.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from _rollout import (
    build_pipeline,
    camera_stream,
    capture_rollout,
    chunk_camctrl,
    load_scene,
    probe_xhat0,
    rebuild_cache,
)
from _traj import grammar_poses, strafe_loop_grammar

OUT = Path("integrations/lingbot/drift_correction/outputs")
K = 8
"""Probe chunk position (teacher-forced history = chunks 0..K-1)."""


def main() -> None:
    torch.set_grad_enabled(False)
    pipe = build_pipeline()
    scene = load_scene(0)
    poses = grammar_poses(strafe_loop_grammar((24, 36, 24)))
    poses_t, intr_t, ws = camera_stream(scene, poses)
    camctrls = chunk_camctrl(poses_t, intr_t, ws, pipe, n_chunks=K + 1)
    assert len(camctrls) == K + 1, len(camctrls)

    snaps, _ = capture_rollout(pipe, scene, camctrls, noise_seed=7001)
    lat = [s.clean_latent for s in snaps]
    print(f"rollout captured: {len(snaps)} chunks, latent {tuple(lat[0].shape)}",
          flush=True)

    gen = torch.Generator().manual_seed(4242)
    eps = torch.randn(lat[K].shape, generator=gen)

    # 1. Replay determinism.
    cache = rebuild_cache(pipe, scene, lat[:K], camctrls[:K])
    a = probe_xhat0(pipe, cache, K, camctrls[K], lat[K], step_idx=3, noise=eps)
    del cache
    torch.cuda.empty_cache()
    cache = rebuild_cache(pipe, scene, lat[:K], camctrls[:K])
    b = probe_xhat0(pipe, cache, K, camctrls[K], lat[K], step_idx=3, noise=eps)
    det_max = float((a - b).abs().max())
    print(f"determinism: max |a-b| = {det_max}", flush=True)

    # 2. History sensitivity: replace the last history chunk with chunk 0's
    # latent (same shapes, wrong content) and probe identically.
    del cache
    torch.cuda.empty_cache()
    swapped = lat[: K - 1] + [lat[0]]
    cache = rebuild_cache(pipe, scene, swapped, camctrls[:K])
    c = probe_xhat0(pipe, cache, K, camctrls[K], lat[K], step_idx=3, noise=eps)
    sens = float((a - c).abs().mean()) / float(a.abs().mean())
    print(f"history sensitivity: mean |a-c| / mean |a| = {sens:.4f}", flush=True)

    # 3. Reconstruction sanity at the lowest-noise solver step.
    rec = float((a - lat[K]).abs().mean()) / float(lat[K].abs().mean())
    print(f"reconstruction: mean |xhat0-x0| / mean |x0| = {rec:.4f}", flush=True)

    report = {
        "probe_chunk": K,
        "determinism_max_abs": det_max,
        "history_sensitivity_rel": sens,
        "reconstruction_rel_err": rec,
        "pass": det_max == 0.0 and sens > 0.001,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "check_adapter.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("ADAPTER-CHECK-DONE", flush=True)


if __name__ == "__main__":
    main()

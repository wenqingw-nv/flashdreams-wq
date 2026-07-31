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

"""Strafe-loop drift-pair clips for the LingBot host (zero real videos).

Each clip is one seeded rollout on a chunk-aligned strafe loop: after a
9-frame lead-in (chunk 0 decodes 9 pixel frames, later chunks 12), every
lap is ``e-N, q-N`` with ``2N`` a multiple of 12, so lap boundaries land on
chunk boundaries and chunk ``k`` of lap ``j >= 2`` revisits the exact camera
poses of its lap-1 counterpart ``k - (j-1) * lap_chunks``. Lap-1 chunks are
the clean teacher content; later laps are the drifted student states.
Geometry (leg length) is constant within a clip and varied across clips —
uniform loops across the whole pair set taught HY's corrector a repeat
prior (owner-confirmed failure mode).

Run from the flashdreams repo root::

    .venv/bin/python integrations/lingbot/drift_correction/build_pairs.py
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
    capture_rollout,
    chunk_camctrl,
    load_scene,
)
from _traj import grammar_poses

OUT_DIR = Path("integrations/lingbot/drift_correction/outputs/pairs")

SCENES = tuple(
    int(s) for s in os.environ.get("SCENES", "0,1,2,3,4,5").split(",")
)
"""Example scenes to roll out (each becomes one clip)."""

LEGS = {0: 24, 1: 36, 2: 48, 3: 24, 4: 36, 5: 48}
"""Strafe leg length in frames per scene (2*leg % 12 == 0 keeps laps
chunk-aligned); varied across clips for geometry diversity."""

NUM_CHUNK = int(os.environ.get("NUM_CHUNK", "29"))
"""Rollout length in chunks (29 chunks = lead-in + 28 lap chunks;
covers 7/4.6/3.5 laps at legs 24/36/48 — deep enough for late-lap drift)."""

SEED0 = int(os.environ.get("SEED0", "31000"))
"""Base rollout seed; clip i uses ``SEED0 + i``."""


def clip_grammar(leg: int, num_chunk: int) -> tuple[str, int]:
    """Lead-in + repeated constant-leg laps covering ``num_chunk`` chunks.

    Returns the grammar and ``lap_chunks`` (chunks per lap). The lead-in
    ``w-9`` absorbs chunk 0's 9-frame decode so lap boundaries align to the
    12-frame cadence of later chunks.
    """
    lap_frames = 2 * leg
    assert lap_frames % 12 == 0, leg
    lap_chunks = lap_frames // 12
    n_laps = -(-(num_chunk - 1) // lap_chunks)  # ceil: cover the tail
    lap = f"e-{leg}, q-{leg}"
    return ", ".join(["w-9"] + [lap] * n_laps), lap_chunks


def main() -> None:
    torch.set_grad_enabled(False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pipe = build_pipeline()
    for i, scene_idx in enumerate(SCENES):
        out = OUT_DIR / f"clip_{scene_idx:02d}.pt"
        if out.exists():
            print(f"{out.name}: exists, skip", flush=True)
            continue
        scene = load_scene(scene_idx)
        leg = LEGS[scene_idx]
        grammar, lap_chunks = clip_grammar(leg, NUM_CHUNK)
        poses = grammar_poses(grammar)
        poses_t, intr_t, ws = camera_stream(scene, poses)
        camctrls = chunk_camctrl(poses_t, intr_t, ws, pipe, n_chunks=NUM_CHUNK)
        snaps, _ = capture_rollout(
            pipe, scene, camctrls, noise_seed=SEED0 + i
        )
        torch.save(
            {
                "scene_idx": scene_idx,
                "grammar": grammar,
                "leg": leg,
                "lap_chunks": lap_chunks,
                "num_chunk": len(snaps),
                "seed": SEED0 + i,
                "poses": poses,
                "latents": [s.clean_latent for s in snaps],
            },
            out,
        )
        print(
            f"{out.name}: {len(snaps)} chunks, leg={leg} "
            f"(lap={lap_chunks} chunks), seed={SEED0 + i}",
            flush=True,
        )
    (OUT_DIR / "meta.json").write_text(
        json.dumps({"scenes": SCENES, "legs": LEGS, "num_chunk": NUM_CHUNK}, indent=2)
    )
    print("PAIRS-DONE", flush=True)


if __name__ == "__main__":
    main()

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

"""Tiled-HDMap loop rollouts: drifted-vs-clean pair clips for Omnidreams.

The Omnidreams action grammar is the HDMap conditioning video, so the
strafe-loop trick ports as *HDMap tiling*: the rollout conditioning is
``clip[:5] + tile(clip[5 : 5 + 8 * LAP_CHUNKS], LAPS)`` — every lap's chunk
``i`` consumes pixel-identical HDMap frames, so a late (drifted) chunk's
KV-window content can be swapped for its lap-1 (clean) counterparts at the
same conditioning and the same RoPE positions. The lap boundary is a
conditioning teleport (a smooth loop would need re-rendering a loop
trajectory through the ludus renderer — noted, not built); the gate only
probes chunks whose 3-chunk window lies inside one lap.

Rollouts run ~21.5 s (81 chunks) because this host's drift shows late
(OmniDreams' own segmented FVD degrades over 20 s despite Self-Forcing
training + sinks); the gate reports the gap as a function of depth.

Assets: local benchmarking corpus (48 scenarios, authentic ludus-rendered
HDMap @ 704x1280x48 frames + first frame). The HF sample dataset is gated
and this box has no token.

Run from the flashdreams repo root::

    .venv/bin/python integrations/omnidreams/drift_correction/build_pairs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from _host import build_pipeline, capture_rollout, save_clip
from omnidreams.runner import (
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    _load_first_frame,
    _load_video,
    _write_video,
)

## Pair-build configuration

OUT_DIR = Path("integrations/omnidreams/drift_correction/outputs/pairs")
"""Clip files consumed by ``gate_faithful.py``."""

SCENARIO_DIR = Path(
    "/localhome/local-wenqingw/projs/benchmarking/corpus_v0/scenarios"
)
"""Local scenario corpus: ``clip_cXX_sY/{hdmap_*.mp4, first_frame_*.png}``."""

CAMERA = "camera_front_wide_120fov"
"""Single-view camera the chunk2 checkpoint was trained on."""

SCENARIOS = (
    "clip_c00_s0",
    "clip_c02_s1",
    "clip_c04_s2",
    "clip_c08_s1",
    "clip_c10_s1",
    "clip_c13_s1",
)
"""Six scenarios spread across the corpus's clip families."""

PROMPT = (
    "Driving scene from a front-facing car camera. Urban environment with "
    "roads, vehicles, pedestrians, traffic signs, and buildings. Clear "
    "visibility, realistic lighting, photorealistic quality. High resolution "
    "dashcam footage of city driving."
)
"""Shipped single-view default prompt (matches the runner literal)."""

LAP_CHUNKS = 5
"""AR chunks per lap (40 decoded frames = 1.33 s; the 48-frame source
clips leave 43 frames after chunk 0's 5, so 5 chunks is the max)."""

LAPS = 16
"""Laps per rollout: 81 chunks = 645 frames = 21.5 s at 30 fps."""

NUM_CHUNK = 1 + LAP_CHUNKS * LAPS
"""Chunk 0 (image-anchored, 5 frames) + the tiled laps."""

NOISE_SEED = 5042
"""Diffusion RNG seed per rollout (offset by clip index)."""


def tile_hdmap(hdmap: torch.Tensor) -> torch.Tensor:
    """Tile a ``[T, C, H, W]`` HDMap clip into the loop conditioning.

    Layout: the first 5 frames (chunk 0) then ``LAPS`` copies of the next
    ``8 * LAP_CHUNKS`` frames, so chunks ``1 + j * LAP_CHUNKS + i`` consume
    identical pixels for every lap ``j``.
    """
    lap_frames = 8 * LAP_CHUNKS
    assert hdmap.shape[0] >= 5 + lap_frames, (
        f"clip has {hdmap.shape[0]} frames; need {5 + lap_frames}."
    )
    lap = hdmap[5 : 5 + lap_frames]
    return torch.cat([hdmap[:5], lap.repeat(LAPS, 1, 1, 1)], dim=0)


def main() -> None:
    torch.set_grad_enabled(False)
    pipe = build_pipeline(with_oneshot_encoders=True)
    device, dtype = pipe.device, torch.bfloat16

    for c, scenario in enumerate(SCENARIOS):
        out = OUT_DIR / f"clip_{c:02d}.pt"
        if out.exists():
            print(f"SKIP clip {c} ({scenario}): {out} exists", flush=True)
            continue
        sdir = SCENARIO_DIR / scenario
        hdmap = _load_video(
            sdir / f"hdmap_{CAMERA}.mp4",
            pixel_height=DEFAULT_VIDEO_HEIGHT,
            pixel_width=DEFAULT_VIDEO_WIDTH,
            device=device,
            dtype=dtype,
        )
        hdmap = tile_hdmap(hdmap)[None, None]  # [1, 1, T, C, H, W]
        first = _load_first_frame(
            sdir / f"first_frame_{CAMERA}.png",
            pixel_height=DEFAULT_VIDEO_HEIGHT,
            pixel_width=DEFAULT_VIDEO_WIDTH,
            device=device,
            dtype=dtype,
        )[None, :, None]  # [1, V=1, 1, C, H, W]

        embeddings = pipe.precompute_embeddings(text=[[PROMPT]], image=first)
        cache = pipe.initialize_cache_from_embeddings(
            text_embeddings=embeddings["text_embeddings"],
            image_embeddings=embeddings["image_embeddings"],
        )
        snaps, video = capture_rollout(
            pipe,
            cache,
            hdmap_video=hdmap,
            num_chunk=NUM_CHUNK,
            noise_seed=NOISE_SEED + c,
        )
        save_clip(
            out,
            snaps=snaps,
            embeddings=embeddings,
            meta={
                "scenario": scenario,
                "prompt": PROMPT,
                "num_chunk": NUM_CHUNK,
                "lap_chunks": LAP_CHUNKS,
                "laps": LAPS,
                "noise_seed": NOISE_SEED + c,
            },
        )
        mp4 = OUT_DIR / f"clip_{c:02d}_loop.mp4"
        _write_video(video[0, 0].permute(0, 2, 3, 1), mp4, fps=30)
        print(f"clip {c} ({scenario}): saved {out} + {mp4}", flush=True)

    print("PAIRS-DONE", flush=True)


if __name__ == "__main__":
    main()

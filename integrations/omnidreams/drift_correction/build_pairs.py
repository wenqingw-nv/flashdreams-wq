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

"""Tiled-HDMap loop rollouts: drifted-vs-clean pair clips for Omnidreams (v2).

The Omnidreams action grammar is the HDMap conditioning video, so the
strafe-loop trick ports as *HDMap tiling*: the rollout conditioning is
``clip[:5] + tile(clip[5 : 5 + 8 * LAP_CHUNKS], LAPS)`` — every lap's chunk
``i`` consumes pixel-identical HDMap frames, so a late (drifted) chunk's
KV-window content can be swapped for its lap-aligned clean counterparts at
the same conditioning and the same RoPE positions. The lap boundary is a
conditioning teleport; probes/training only use chunks whose surviving
window lies inside one lap (gate) or map counterfactuals by lap position
(training). Rollouts run ~21.5 s (81 chunks) because this host's residual
drift shows late.

v2 (owner flag 2026-07-22): pairs v1 seeded rollouts with the local
benchmarking corpus, whose "first frames" are HDMap renders — the rollouts
stayed in a render-adjacent régime and collapsed to one scene. v2 sources
the gated HF sample set (``nvidia/omni-dreams-samples``): 32 clips with
REAL dashcam first frames, per-clip prompts, and 80 s authentic HDMaps.
Requires an authenticated HF token.

Run from the flashdreams repo root::

    HF_TOKEN=$(cat ~/.cache/huggingface/token) \
        .venv/bin/python integrations/omnidreams/drift_correction/build_pairs.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from _host import build_pipeline, capture_rollout, save_clip
from omnidreams.runner import (
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    _ensure_hf_single_view_example_data_synced,
    _load_first_frame,
    _load_video,
    _write_video,
)

## Pair-build configuration

OUT_DIR = Path(
    os.environ.get(
        "PAIRS_DIR", "integrations/omnidreams/drift_correction/outputs/pairs_v2"
    )
)
"""Clip files consumed by ``gate_faithful.py`` / ``train_v1.py``."""

N_CLIPS = int(os.environ.get("N_CLIPS", "6"))
"""Sample clips to roll out (first ``N_CLIPS`` of the dataset, sorted)."""

LAP_CHUNKS = 5
"""AR chunks per lap (40 decoded frames = 1.33 s; kept identical to pairs
v1 so gate numbers compare across régimes)."""

LAPS = 16
"""Laps per rollout: 81 chunks = 645 frames = 21.5 s at 30 fps."""

NUM_CHUNK = 1 + LAP_CHUNKS * LAPS
"""Chunk 0 (image-anchored, 5 frames) + the tiled laps."""

NOISE_SEED = 5042
"""Diffusion RNG seed per rollout (offset by clip index)."""


def _list_sample_uuids(n: int) -> list[str]:
    """Return the first ``n`` single-view sample UUIDs, alphabetically."""
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFolder

    entries = HfApi().list_repo_tree(
        repo_id="nvidia/omni-dreams-samples",
        repo_type="dataset",
        path_in_repo="data/single_view",
        recursive=False,
    )
    uuids = sorted(
        e.path.rsplit("/", 1)[-1] for e in entries if isinstance(e, RepoFolder)
    )
    assert len(uuids) >= n, f"dataset lists only {len(uuids)} single-view clips"
    return uuids[:n]


def _clip_prompt(uuid: str) -> str:
    """Fetch the clip's own prompt from the dataset."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="nvidia/omni-dreams-samples",
        repo_type="dataset",
        filename=f"data/single_view/{uuid}/prompt.txt",
    )
    return Path(path).read_text().strip()


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
    dtype = torch.bfloat16

    # Load every clip's inputs BEFORE any model/rollout work: video decode
    # forks ffmpeg, which fails silently (empty read) once this process has
    # grown to rollout size -- observed 3x on this box, always on the
    # second decode of a job.
    inputs: list[tuple[str, str, torch.Tensor, torch.Tensor] | None] = []
    for c, uuid in enumerate(_list_sample_uuids(N_CLIPS)):
        if (OUT_DIR / f"clip_{c:02d}.pt").exists():
            inputs.append(None)
            continue
        (hdmap_path,), (frame_path,) = _ensure_hf_single_view_example_data_synced(uuid)
        hdmap = _load_video(
            hdmap_path,
            pixel_height=DEFAULT_VIDEO_HEIGHT,
            pixel_width=DEFAULT_VIDEO_WIDTH,
            device="cpu",
            dtype=dtype,
        )
        hdmap = tile_hdmap(hdmap)[None, None]  # [1, 1, T, C, H, W]
        first = _load_first_frame(
            frame_path,
            pixel_height=DEFAULT_VIDEO_HEIGHT,
            pixel_width=DEFAULT_VIDEO_WIDTH,
            device="cpu",
            dtype=dtype,
        )[None, :, None]  # [1, V=1, 1, C, H, W]
        inputs.append((uuid, _clip_prompt(uuid), hdmap, first))
        print(f"loaded inputs for clip {c} ({uuid})", flush=True)

    pipe = build_pipeline(with_oneshot_encoders=True)
    device = pipe.device

    if CORRECTOR_LORA:
        from _lora import apply_lora, load_lora, unwrap_compiled
        from eval_rollouts import install_alpha_gate

        transformer = pipe.diffusion_model.transformer
        network = unwrap_compiled(transformer.network)
        apply_lora(network)
        load_lora(network, CORRECTOR_LORA)
        install_alpha_gate(transformer, network, {"gain": ("gate", 0.5)})
        print(f"corrector active at gate x 0.5: {CORRECTOR_LORA}", flush=True)

    for c, item in enumerate(inputs):
        out = OUT_DIR / f"clip_{c:02d}.pt"
        if item is None:
            print(f"SKIP clip {c}: {out} exists", flush=True)
            continue
        uuid, prompt, hdmap, first = item
        hdmap = hdmap.to(device)
        first = first.to(device)

        embeddings = pipe.precompute_embeddings(text=[[prompt]], image=first)
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
                "uuid": uuid,
                "prompt": prompt,
                "num_chunk": NUM_CHUNK,
                "lap_chunks": LAP_CHUNKS,
                "laps": LAPS,
                "noise_seed": NOISE_SEED + c,
            },
        )
        mp4 = OUT_DIR / f"clip_{c:02d}_loop.mp4"
        _write_video(video[0, 0].permute(0, 2, 3, 1), mp4, fps=30)
        print(f"clip {c} ({uuid}): saved {out} + {mp4}", flush=True)

    print("PAIRS-DONE", flush=True)


if __name__ == "__main__":
    main()

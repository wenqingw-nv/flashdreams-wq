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

"""Pre-merged vs unfused corrector equivalence on a full rollout.

Runs the shipped deploy hook (``omnidreams/_drift_corrector.py``) in one
mode per process (``MODE=premerged`` / ``MODE=unfused``; the hooks mutate
the network irreversibly), same protocol as ``bench_latency.py``
(scenario 7, 81 chunks, seed 5042). Saves per-chunk clean latents + the
decoded rollout; once both modes exist, compares latents chunk-by-chunk
and writes an sbs frame stack.

Run from the flashdreams repo root::

    MODE=premerged .venv/bin/python integrations/omnidreams/drift_correction/check_premerge_equiv.py
    MODE=unfused   .venv/bin/python integrations/omnidreams/drift_correction/check_premerge_equiv.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from _host import build_pipeline, capture_rollout
from bench_latency import GATE_SCALE, LORA, NUM_CHUNK, SCENARIO, SEED
from build_pairs import _clip_prompt, _list_sample_uuids, _sample_files
from omnidreams._drift_corrector import apply_drift_corrector
from omnidreams.runner import (
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    _load_video,
)

try:  # upstream main
    from flashdreams.infra.runner_io import load_first_frame_tensor
except ModuleNotFoundError:  # older branches
    from omnidreams.runner import _load_first_frame as load_first_frame_tensor

MODE = os.environ.get("MODE", "premerged")
OUT_DIR = Path("integrations/omnidreams/drift_correction/outputs/premerge_equiv")


def compare() -> None:
    """Compare the two saved latent stacks; write JSON + sbs frame stack."""
    import mediapy as media

    lat = {
        m: torch.load(OUT_DIR / f"latents_{m}.pt", weights_only=True)
        for m in ("premerged", "unfused")
    }
    per_chunk = []
    for a, b in zip(lat["premerged"], lat["unfused"]):
        d = (a - b).abs()
        per_chunk.append(
            {
                "max_abs_diff": float(d.max()),
                "mean_abs_diff": float(d.mean()),
                "latent_abs_mean": float(b.abs().mean()),
            }
        )
    frames = {m: np.load(OUT_DIR / f"frames_{m}.npy") for m in ("premerged", "unfused")}
    pix_diff = int(
        np.abs(
            frames["premerged"].astype(np.int16) - frames["unfused"].astype(np.int16)
        ).max()
    )
    n = len(frames["unfused"])
    t_idx = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    rows = [
        np.concatenate([frames[m][t] for t in t_idx], axis=1)
        for m in ("premerged", "unfused")
    ]
    media.write_image(str(OUT_DIR / "sbs_stack.png"), np.concatenate(rows, axis=0))
    report = {
        "protocol": {"scenario": SCENARIO, "num_chunk": NUM_CHUNK, "seed": SEED},
        "max_abs_latent_diff": max(c["max_abs_diff"] for c in per_chunk),
        "mean_abs_latent_diff": float(np.mean([c["mean_abs_diff"] for c in per_chunk])),
        "latent_abs_mean": float(np.mean([c["latent_abs_mean"] for c in per_chunk])),
        "max_abs_pixel_diff_uint8": pix_diff,
        "per_chunk": per_chunk,
    }
    (OUT_DIR / "compare.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_chunk"}, indent=2))
    print("COMPARE-DONE", flush=True)


def main() -> None:
    torch.set_grad_enabled(False)
    assert MODE in ("premerged", "unfused"), MODE
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16
    uuid = _list_sample_uuids(SCENARIO + 1)[SCENARIO]
    (hdmap_path,), (frame_path,) = _sample_files(uuid)
    n_frames = 5 + (NUM_CHUNK - 1) * 8
    hdmap = _load_video(
        hdmap_path,
        pixel_height=DEFAULT_VIDEO_HEIGHT,
        pixel_width=DEFAULT_VIDEO_WIDTH,
        device="cpu",  # ty: ignore[invalid-argument-type]
        dtype=dtype,
    )[:n_frames][None, None]
    first = load_first_frame_tensor(
        frame_path,
        pixel_height=DEFAULT_VIDEO_HEIGHT,
        pixel_width=DEFAULT_VIDEO_WIDTH,
        device="cpu",  # ty: ignore[invalid-argument-type]
        dtype=dtype,
    )[None, :, None]
    prompt = _clip_prompt(uuid)

    pipe = build_pipeline(with_oneshot_encoders=True)
    device = pipe.device
    emb = pipe.precompute_embeddings(text=[[prompt]], image=first.to(device))

    mem0 = torch.cuda.memory_allocated()
    mode_str = apply_drift_corrector(
        SimpleNamespace(pipeline=pipe),
        Path(LORA),
        GATE_SCALE[1],
        unfused=MODE == "unfused",
    )
    added_mb = (torch.cuda.memory_allocated() - mem0) / 2**20
    print(f"{MODE}: {mode_str} (measured +{added_mb:.1f} MiB)", flush=True)

    cache = pipe.initialize_cache_from_embeddings(
        text_embeddings=emb["text_embeddings"],
        image_embeddings=emb["image_embeddings"],
    )
    snaps, video = capture_rollout(
        pipe, cache, hdmap_video=hdmap.to(device), num_chunk=NUM_CHUNK, noise_seed=SEED
    )
    torch.save(
        [s.clean_latent.float().cpu() for s in snaps], OUT_DIR / f"latents_{MODE}.pt"
    )
    import mediapy as media

    arr = ((video[0, 0].permute(0, 2, 3, 1).float().numpy() + 1) / 2 * 255).clip(0, 255)
    arr = arr.astype(np.uint8)
    media.write_video(str(OUT_DIR / f"video_{MODE}.mp4"), arr, fps=30)
    # Raw (pre-encode) frame subset for the pixel-level comparison.
    keep = sorted(set(range(0, len(arr), 16)) | {len(arr) - 1})
    np.save(OUT_DIR / f"frames_{MODE}.npy", arr[keep])
    (OUT_DIR / f"apply_{MODE}.json").write_text(
        json.dumps({"mode_str": mode_str, "measured_added_mib": added_mb}, indent=2)
    )
    print("ROLLOUT-DONE", flush=True)

    if all((OUT_DIR / f"latents_{m}.pt").exists() for m in ("premerged", "unfused")):
        compare()


if __name__ == "__main__":
    main()

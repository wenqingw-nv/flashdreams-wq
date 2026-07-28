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

Runs the shipped deploy hook (``hy_worldplay/_drift_corrector.py``) in one
mode per process (``MODE=premerged`` / ``MODE=unfused``; the hooks mutate
the network irreversibly), same protocol as ``bench_latency.py`` (bridge
pose, 24 chunks, seed 5042). Saves per-chunk clean latents + the decoded
rollout; once both modes exist, compares latents chunk-by-chunk and
writes an sbs frame stack.

Run from the flashdreams repo root::

    MODE=premerged .venv/bin/python integrations/hy_worldplay/drift_correction/check_premerge_equiv.py
    MODE=unfused   .venv/bin/python integrations/hy_worldplay/drift_correction/check_premerge_equiv.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from _rollout import (
    _POINT_CLOUD_SEED,
    _resolve_prompt,
    build_runner,
    preprocess_first_frame,
)
from bench_latency import GATE_SCALE, LORA, NUM_CHUNK, POSE, SEED
from hy_worldplay._drift_corrector import maybe_apply_drift_corrector

MODE = os.environ.get("MODE", "premerged")
OUT_DIR = Path("integrations/hy_worldplay/drift_correction/outputs/premerge_equiv")


def rollout(runner) -> tuple[list[torch.Tensor], torch.Tensor]:
    """One seeded rollout; returns per-chunk clean latents + decoded video."""
    pipe = runner.pipeline
    cfg = runner.config
    device = next(pipe.parameters()).device
    dtype = next(pipe.parameters()).dtype
    image = preprocess_first_frame(
        cfg.image_path, cfg.pixel_height, cfg.pixel_width
    ).to(device=device, dtype=dtype)
    cache = pipe.initialize_cache(text=[_resolve_prompt(cfg.prompt)], image=image)
    runner._bind_action_labels()
    runner._bind_camera_data()
    torch.manual_seed(_POINT_CLOUD_SEED)
    runner._bind_memory_config(device=device)
    pipe.diffusion_model._rng = torch.Generator(device=device).manual_seed(SEED)

    latents: list[torch.Tensor] = []
    chunks: list[torch.Tensor] = []
    for ar_idx in range(cfg.num_chunk):
        chunk = pipe.generate(ar_idx, cache)
        fs = cache.final_state
        assert fs is not None
        latents.append(fs.clean_latent.detach().float().cpu())
        pipe.finalize(ar_idx, cache)
        chunks.append(chunk.cpu())
    video = torch.cat(chunks, dim=-4)
    del cache
    torch.cuda.empty_cache()
    return latents, video


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
        "protocol": {"pose": POSE, "num_chunk": NUM_CHUNK, "seed": SEED},
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

    runner = build_runner(
        num_chunk=NUM_CHUNK,
        pose=POSE,
        output_dir=OUT_DIR / "runner_tmp",
        image_path=None,
    )
    mem0 = torch.cuda.memory_allocated()
    mode_str = maybe_apply_drift_corrector(
        runner, Path(LORA), GATE_SCALE[1], unfused=MODE == "unfused"
    )
    added_mb = (torch.cuda.memory_allocated() - mem0) / 2**20
    print(f"{MODE}: {mode_str} (measured +{added_mb:.1f} MiB)", flush=True)

    latents, video = rollout(runner)
    torch.save(latents, OUT_DIR / f"latents_{MODE}.pt")
    import mediapy as media

    while video.dim() > 4:
        assert video.shape[0] == 1
        video = video[0]
    arr = ((video.permute(0, 2, 3, 1).float().numpy() + 1) / 2 * 255).clip(0, 255)
    arr = arr.astype(np.uint8)
    media.write_video(str(OUT_DIR / f"video_{MODE}.mp4"), arr, fps=runner.config.fps)
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

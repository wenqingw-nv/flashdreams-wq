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

"""Inference-latency benchmark: base vs pre-merged corrector deploy hook.

Same protocol as ``bench_latency.py`` (scenario 7, 81 chunks, 3 runs,
steady-state stats past WARM_CHUNKS), but the corrected config is the
shipped deploy hook's default pre-merged path
(``omnidreams/_drift_corrector.py``): per-step cached weight sets, no
LoRA matmuls in the hot path. Base is measured on the untouched network
before the hook is applied. JSON to
``outputs/bench_latency_premerged.json``.

Run from the flashdreams repo root::

    HF_TOKEN=... .venv/bin/python integrations/omnidreams/drift_correction/bench_latency_premerged.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from _host import build_pipeline
from bench_latency import (
    GATE_SCALE,
    LORA,
    NUM_CHUNK,
    RUNS,
    SCENARIO,
    WARM_CHUNKS,
    rollout_timed,
)
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

_BASE = Path("integrations/omnidreams/drift_correction")
OUT_JSON = _BASE / "outputs/bench_latency_premerged.json"


def main() -> None:
    torch.set_grad_enabled(False)
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
    hdmap = hdmap.to(device)
    emb = pipe.precompute_embeddings(text=[[prompt]], image=first.to(device))

    results: dict = {
        "protocol": {
            "scenario": SCENARIO,
            "num_chunk": NUM_CHUNK,
            "runs": RUNS,
            "warm_chunks": WARM_CHUNKS,
            "resolution": f"{DEFAULT_VIDEO_WIDTH}x{DEFAULT_VIDEO_HEIGHT}",
        },
        "configs": {},
    }

    def bench(config: str) -> None:
        # Full warmup rollout (compile/caches), not counted.
        cache = pipe.initialize_cache_from_embeddings(
            text_embeddings=emb["text_embeddings"],
            image_embeddings=emb["image_embeddings"],
        )
        rollout_timed(pipe, cache, hdmap)
        del cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        runs = []
        for r in range(RUNS):
            cache = pipe.initialize_cache_from_embeddings(
                text_embeddings=emb["text_embeddings"],
                image_embeddings=emb["image_embeddings"],
            )
            runs.append(rollout_timed(pipe, cache, hdmap))
            del cache
            torch.cuda.empty_cache()
            print(f"{config} run {r}: e2e {sum(runs[-1]):.2f}s", flush=True)
        peak_gb = torch.cuda.max_memory_allocated() / 2**30
        steady = np.array([t[WARM_CHUNKS:] for t in runs])
        e2e = np.array([sum(t) for t in runs])
        frames = 5 + (NUM_CHUNK - 1) * 8
        results["configs"][config] = {
            "per_chunk_ms_mean": float(steady.mean() * 1e3),
            "per_chunk_ms_std": float(steady.mean(axis=1).std() * 1e3),
            "e2e_s_mean": float(e2e.mean()),
            "e2e_s_std": float(e2e.std()),
            "peak_vram_gb": peak_gb,
            "effective_fps": frames / float(e2e.mean()),
            "runs_e2e_s": [float(x) for x in e2e],
        }

    bench("base")

    mem0 = torch.cuda.memory_allocated()
    mode_str = apply_drift_corrector(
        SimpleNamespace(pipeline=pipe), Path(LORA), GATE_SCALE[1]
    )
    added_mib = (torch.cuda.memory_allocated() - mem0) / 2**20
    print(f"applied: {mode_str} (measured +{added_mib:.1f} MiB)", flush=True)
    results["deploy_mode"] = mode_str
    results["premerge_added_mib"] = added_mib

    bench("corrgate025_premerged")

    b = results["configs"]["base"]
    c = results["configs"]["corrgate025_premerged"]
    results["overhead_ms_per_chunk"] = c["per_chunk_ms_mean"] - b["per_chunk_ms_mean"]
    results["overhead_pct_per_chunk"] = 100.0 * (
        c["per_chunk_ms_mean"] / b["per_chunk_ms_mean"] - 1.0
    )
    results["overhead_pct_e2e"] = 100.0 * (c["e2e_s_mean"] / b["e2e_s_mean"] - 1.0)

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print("BENCH-DONE", flush=True)


if __name__ == "__main__":
    main()

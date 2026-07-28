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

Same protocol as ``bench_latency.py`` (bridge pose, 24 chunks, 3 runs,
steady-state stats past WARM_CHUNKS), but the corrected config is the
shipped deploy hook's default pre-merged path
(``hy_worldplay/_drift_corrector.py``): per-step cached weight sets, no
LoRA matmuls in the hot path. Base is measured on the untouched network
before the hook is applied. JSON to
``outputs/bench_latency_premerged.json``.

Run from the flashdreams repo root::

    .venv/bin/python integrations/hy_worldplay/drift_correction/bench_latency_premerged.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from _rollout import build_runner
from bench_latency import (
    GATE_SCALE,
    LORA,
    NUM_CHUNK,
    POSE,
    RUNS,
    WARM_CHUNKS,
    rollout_timed,
)
from hy_worldplay._drift_corrector import maybe_apply_drift_corrector

_BASE = Path("integrations/hy_worldplay/drift_correction")
OUT_JSON = _BASE / "outputs/bench_latency_premerged.json"


def main() -> None:
    torch.set_grad_enabled(False)
    runner = build_runner(
        num_chunk=NUM_CHUNK,
        pose=POSE,
        output_dir=_BASE / "outputs/bench_tmp",
        image_path=None,
    )

    results: dict = {
        "protocol": {
            "pose": POSE,
            "num_chunk": NUM_CHUNK,
            "runs": RUNS,
            "warm_chunks": WARM_CHUNKS,
            "resolution": "1280x704",
        },
        "configs": {},
    }

    def bench(config: str) -> None:
        rollout_timed(runner)  # warmup rollout, not counted
        torch.cuda.reset_peak_memory_stats()
        runs = []
        for r in range(RUNS):
            runs.append(rollout_timed(runner))
            print(f"{config} run {r}: e2e {sum(runs[-1]):.2f}s", flush=True)
        peak_gb = torch.cuda.max_memory_allocated() / 2**30
        steady = np.array([t[WARM_CHUNKS:] for t in runs])
        e2e = np.array([sum(t) for t in runs])
        frames = NUM_CHUNK * 16 - 3  # 13 + 16*(n-1) decoded frames
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
    mode_str = maybe_apply_drift_corrector(runner, Path(LORA), GATE_SCALE[1])
    added_mib = (torch.cuda.memory_allocated() - mem0) / 2**20
    print(f"applied: {mode_str} (measured +{added_mib:.1f} MiB)", flush=True)
    results["deploy_mode"] = mode_str
    results["premerge_added_mib"] = added_mib

    bench("corrgate050_premerged")

    b = results["configs"]["base"]
    c = results["configs"]["corrgate050_premerged"]
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

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

"""Inference-latency benchmark: base vs shipped corrector (corrgate050).

Same rollout protocol as the PR #396 Measured section (704x1280, 24-chunk
rollouts, default bridge trajectory): per-chunk wall clock with CUDA sync,
end-to-end time, steady-state stats (first :data:`WARM_CHUNKS` chunks
excluded), peak VRAM, and LoRA parameter count. RUNS rollouts per config
after a full warmup rollout; JSON to ``outputs/bench_latency.json``.

Run from the flashdreams repo root::

    .venv/bin/python integrations/hy_worldplay/drift_correction/bench_latency.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from _lora import apply_lora, load_lora, lora_parameters, set_lora_scale, unwrap_compiled
from _rollout import (
    _POINT_CLOUD_SEED,
    _resolve_prompt,
    build_runner,
    install_alpha_gate,
    preprocess_first_frame,
)

## Benchmark configuration

_BASE = Path("integrations/hy_worldplay/drift_correction")

LORA = os.environ.get("LORA", str(_BASE / "outputs/lora_v2c2.pt"))
"""Shipped corrector checkpoint (motion-job path)."""

GATE_SCALE = ("gate", float(os.environ.get("GATE_SCALE", "0.5")))
"""Shipped dial: alpha*(t) x 0.5 (``corrgate050``)."""

POSE = os.environ.get("POSE", "w-40, right-3, w-40, left-3, w-73")
"""Default bridge trajectory (PR #396 Measured protocol)."""

NUM_CHUNK = int(os.environ.get("NUM_CHUNK", "24"))
RUNS = int(os.environ.get("RUNS", "3"))
WARM_CHUNKS = int(os.environ.get("WARM_CHUNKS", "4"))
SEED = 5042
OUT_JSON = _BASE / "outputs/bench_latency.json"


def rollout_timed(runner) -> list[float]:
    """One rollout; returns per-chunk wall-clock seconds (CUDA-synced)."""
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

    times: list[float] = []
    for ar_idx in range(cfg.num_chunk):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pipe.generate(ar_idx, cache)
        pipe.finalize(ar_idx, cache)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    del cache
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return times


def main() -> None:
    torch.set_grad_enabled(False)
    runner = build_runner(
        num_chunk=NUM_CHUNK,
        pose=POSE,
        output_dir=_BASE / "outputs/bench_tmp",
        image_path=None,
    )
    network = unwrap_compiled(runner.pipeline.diffusion_model.transformer.network)
    base_params = sum(p.numel() for p in network.parameters())
    apply_lora(network)
    load_lora(network, LORA)
    lora_params = sum(p.numel() for p in lora_parameters(network))
    mode: dict = {"gain": 0.0}
    install_alpha_gate(runner, network, mode)

    results: dict = {
        "protocol": {
            "pose": POSE,
            "num_chunk": NUM_CHUNK,
            "runs": RUNS,
            "warm_chunks": WARM_CHUNKS,
            "resolution": "1280x704",
        },
        "lora_params": lora_params,
        "base_params": base_params,
        "lora_param_pct": 100.0 * lora_params / base_params,
        "configs": {},
    }

    for config, gain in (("base", 0.0), ("corrgate050", GATE_SCALE)):
        mode["gain"] = gain
        if not isinstance(gain, tuple):
            set_lora_scale(network, gain)
        rollout_timed(runner)  # warmup rollout, not counted
        torch.cuda.reset_peak_memory_stats()
        runs = []
        for r in range(RUNS):
            runs.append(rollout_timed(runner))
            print(f"{config} run {r}: e2e {sum(runs[-1]):.2f}s", flush=True)
        peak_gb = torch.cuda.max_memory_allocated() / 2**30
        steady = np.array([t[WARM_CHUNKS:] for t in runs])
        e2e = np.array([sum(t) for t in runs])
        results["configs"][config] = {
            "per_chunk_ms_mean": float(steady.mean() * 1e3),
            "per_chunk_ms_std": float(steady.mean(axis=1).std() * 1e3),
            "e2e_s_mean": float(e2e.mean()),
            "e2e_s_std": float(e2e.std()),
            "peak_vram_gb": peak_gb,
            "runs_e2e_s": [float(x) for x in e2e],
        }

    b = results["configs"]["base"]
    c = results["configs"]["corrgate050"]
    results["overhead_pct_per_chunk"] = 100.0 * (
        c["per_chunk_ms_mean"] / b["per_chunk_ms_mean"] - 1.0
    )
    results["overhead_pct_e2e"] = 100.0 * (c["e2e_s_mean"] / b["e2e_s_mean"] - 1.0)
    frames = NUM_CHUNK * 16 - 3  # 13 + 16*(n-1) decoded frames
    for k in ("base", "corrgate050"):
        results["configs"][k]["effective_fps"] = (
            frames / results["configs"][k]["e2e_s_mean"]
        )

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print("BENCH-DONE", flush=True)


if __name__ == "__main__":
    main()

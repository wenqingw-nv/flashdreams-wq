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

"""Inference-latency benchmark: base vs deployed corrector (corrgate025).

Same rollout protocol as the eval sweeps (704x1280, 81 chunks, real HDMap
conditioning, scenario 7): per-chunk wall clock with CUDA sync, end-to-end
time, steady-state stats (chunk 0 is image-anchored and the first chunks
carry compile/cache warmup — excluded via :data:`WARM_CHUNKS`), peak VRAM,
and LoRA parameter count. RUNS rollouts per config after a full warmup
rollout; JSON + human-readable summary to ``outputs/bench_latency.json``.

Run from the flashdreams repo root::

    HF_TOKEN=... .venv/bin/python integrations/omnidreams/drift_correction/bench_latency.py
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
from _host import build_pipeline
from build_pairs import _clip_prompt, _list_sample_uuids, _sample_files
from eval_rollouts import install_alpha_gate
from omnidreams.runner import (
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    _load_video,
)
try:  # upstream main
    from flashdreams.infra.runner_io import load_first_frame_tensor
except ModuleNotFoundError:  # older branches
    from omnidreams.runner import _load_first_frame as load_first_frame_tensor

## Benchmark configuration

_BASE = Path("integrations/omnidreams/drift_correction")

LORA = os.environ.get("LORA", str(_BASE / "outputs/lora_v2_v3_valpeak.pt"))
"""Deployed corrector checkpoint."""

GATE_SCALE = ("gate", float(os.environ.get("GATE_SCALE", "0.25")))
"""Deployed dial: alpha*(t) x 0.25 (``corrgate025``)."""

SCENARIO = int(os.environ.get("SCENARIO", "7"))
NUM_CHUNK = int(os.environ.get("NUM_CHUNK", "81"))
RUNS = int(os.environ.get("RUNS", "3"))
WARM_CHUNKS = int(os.environ.get("WARM_CHUNKS", "5"))
"""Steady-state stats exclude the first WARM_CHUNKS chunks of each rollout."""

SEED = 5042
OUT_JSON = _BASE / "outputs/bench_latency.json"


def rollout_timed(pipe, cache, hdmap: torch.Tensor) -> list[float]:
    """One rollout; returns per-chunk wall-clock seconds (CUDA-synced)."""
    device = pipe.device
    pipe.diffusion_model._rng = torch.Generator(device=device).manual_seed(SEED)
    times: list[float] = []
    start = 0
    for ar_idx in range(NUM_CHUNK):
        n = pipe.get_num_frames(ar_idx)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pipe.generate(ar_idx, cache, hdmap=hdmap[:, :, start : start + n])
        pipe.finalize(ar_idx, cache)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        start += n
    return times


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

    from _lora import apply_lora, load_lora, lora_parameters, unwrap_compiled

    transformer = pipe.diffusion_model.transformer
    network = unwrap_compiled(transformer.network)
    base_params = sum(p.numel() for p in network.parameters())
    apply_lora(network)
    load_lora(network, LORA)
    lora_params = sum(p.numel() for p in lora_parameters(network))
    mode: dict = {"gain": 0.0}
    install_alpha_gate(transformer, network, mode)

    results: dict = {
        "protocol": {
            "scenario": SCENARIO,
            "num_chunk": NUM_CHUNK,
            "runs": RUNS,
            "warm_chunks": WARM_CHUNKS,
            "resolution": f"{DEFAULT_VIDEO_WIDTH}x{DEFAULT_VIDEO_HEIGHT}",
        },
        "lora_params": lora_params,
        "base_params": base_params,
        "lora_param_pct": 100.0 * lora_params / base_params,
        "configs": {},
    }

    for config, gain in (("base", 0.0), ("corrgate025", GATE_SCALE)):
        mode["gain"] = gain
        if not isinstance(gain, tuple):
            from _lora import set_lora_scale

            set_lora_scale(network, gain)
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
        results["configs"][config] = {
            "per_chunk_ms_mean": float(steady.mean() * 1e3),
            "per_chunk_ms_std": float(steady.mean(axis=1).std() * 1e3),
            "e2e_s_mean": float(e2e.mean()),
            "e2e_s_std": float(e2e.std()),
            "peak_vram_gb": peak_gb,
            "runs_e2e_s": [float(x) for x in e2e],
        }

    b = results["configs"]["base"]
    c = results["configs"]["corrgate025"]
    results["overhead_pct_per_chunk"] = (
        100.0 * (c["per_chunk_ms_mean"] / b["per_chunk_ms_mean"] - 1.0)
    )
    results["overhead_pct_e2e"] = 100.0 * (c["e2e_s_mean"] / b["e2e_s_mean"] - 1.0)
    frames = 5 + (NUM_CHUNK - 1) * 8
    for k in ("base", "corrgate025"):
        results["configs"][k]["effective_fps"] = frames / results["configs"][k][
            "e2e_s_mean"
        ]

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print("BENCH-DONE", flush=True)


if __name__ == "__main__":
    main()

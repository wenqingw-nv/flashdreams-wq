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

"""Closed-loop eval of the CF x PDD 2-NFE hybrid sampler.

Deploys the :mod:`train_v1_mean` corrector as a *solver* change instead of a
per-step residual: the distilled 4-step Euler loop's first two steps collapse
into one mean-velocity jump. With the mean-LoRA at the jump gain, a single
flow forward at ``timesteps[0]`` Euler-integrates straight to ``sigmas[2]``
(t=888.9) -- by the trainer's identity the student's flow at ``sigmas[0]``
*is* its one-jump mean velocity over the ``[0:2)`` window -- and the
remaining solver steps 2, 3 run the stock update on the frozen base (LoRA
scale 0). Three network forwards per chunk vs the base four.

Configs vary the jump-step LoRA gain: ``base`` (stock 4-step, LoRA off),
``hybrid100`` (gain 1.0), ``hybrid072`` (0.719, the measured ``[0:2)`` window
alpha*), ``hybrid050`` (0.5). ``GAINS=0.6,0.85`` adds rows (``hybrid060``
etc.); ``CONFIGS=base,hybrid072`` filters to a subset.

Rollout protocol and ``EVAL_OUT/{config}/p*_i*_s*.mp4`` layout mirror
:mod:`eval_rollouts` so ``score_drift.py`` scores the results unchanged
(point it at this ``EVAL_OUT``). Each chunk is also wall-clock timed with
CUDA sync; steady-state means (chunks 5+) land in ``EVAL_OUT/latency.json``.

Run from the repo root::

    NUM_CHUNK=24 .venv/bin/python integrations/hy_worldplay/drift_correction/eval_hybrid.py
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from _lora import apply_lora, load_lora, set_lora_scale, unwrap_compiled
from _rollout import _POINT_CLOUD_SEED, build_runner
from hy_worldplay.runner import _resolve_prompt, _write_mp4, preprocess_first_frame
from torch import Tensor

## Eval configuration

_BASE = Path("integrations/hy_worldplay/drift_correction")

LORA = os.environ.get("LORA", str(_BASE / "outputs/lora_v1_mean_w02.pt"))
"""Mean-velocity corrector checkpoint (:mod:`train_v1_mean`, window [0:2))."""

ALPHA_STAR = 0.719
"""Measured integrated alpha* for the [0:2) window (``gate_mean_velocity``,
``WINDOW=0:2``): the systematic fraction of the drift gap, i.e. the
risk-optimal jump gain under the gate's bias/variance model."""

JUMP_END = 2
"""The jump covers solver steps ``[0, JUMP_END)``: one forward at
``timesteps[0]`` Euler-integrated straight to ``sigmas[JUMP_END]``, matching
the checkpoint's training window."""

NUM_CHUNK = int(os.environ.get("NUM_CHUNK", "24"))
"""Rollout horizon (24 chunks = 96 latents, the v2-eval protocol)."""

POSES = tuple(
    p.strip()
    for p in os.environ.get(
        "POSES",
        "w-40, right-3, w-40, left-3, w-73;"
        "w-30, d-10, w-30, a-10, w-79;"
        "w-60, right-2, w-50, right-2, w-45",
    ).split(";")
    if p.strip()
)
"""Non-loop eval trajectories, ``;``-separated via the ``POSES`` env var
(each must cover ``NUM_CHUNK * 4 - 1`` motion steps); the defaults are the
bridge cells from :mod:`eval_rollouts`."""

IMAGES_DIR = os.environ.get("IMAGES_DIR", "")
"""Held-out first frames; empty -> the upstream sample image."""

SEEDS = tuple(int(s) for s in os.environ.get("SEEDS", "5042,5043").split(","))
"""Diffusion seeds; every (pose, image, seed) cell runs in every config."""

N_IMAGES = int(os.environ.get("N_IMAGES", "0"))
"""Cap on held-out images (0 = all)."""

PROMPT = os.environ.get("PROMPT", "")
"""Text-prompt override; see :mod:`eval_rollouts` on the mismatched default."""

OUT_DIR = Path(os.environ.get("EVAL_OUT", str(_BASE / "outputs/eval_hybrid")))

WARM_CHUNKS = 4
"""Per-rollout chunks excluded from the latency mean (compile + cache
warmup); the reported steady state covers chunks 5+."""


def build_configs() -> dict[str, float | None]:
    """Config name -> jump-step LoRA gain (``None`` = stock 4-step base).

    ``GAINS=0.6,0.85`` adds partial-gain rows; ``CONFIGS`` filters the table
    to a comma-separated subset (unknown names fail fast).
    """
    configs: dict[str, float | None] = {
        "base": None,
        "hybrid100": 1.0,
        "hybrid072": ALPHA_STAR,
        "hybrid050": 0.5,
    }
    for g in os.environ.get("GAINS", "").split(","):
        if g.strip():
            gain = float(g)
            configs[f"hybrid{gain:.2f}".replace(".", "")] = gain
    keep = {c.strip() for c in os.environ.get("CONFIGS", "").split(",") if c.strip()}
    if keep:
        unknown = keep - configs.keys()
        assert not unknown, f"unknown CONFIGS entries {sorted(unknown)}"
        configs = {k: v for k, v in configs.items() if k in keep}
    return configs


def install_hybrid_sampler(dm: Any, network: Any, mode: dict) -> None:
    """Wrap ``dm.scheduler.sample`` with the mean-velocity-jump hybrid loop.

    ``mode["jump_gain"] = None`` routes to the stock sampler. Otherwise the
    first ``JUMP_END`` solver steps collapse into one jump: LoRA at the jump
    gain, one ``predict_flow`` at ``timesteps[0]``, then
    ``z += (sigmas[JUMP_END] - sigmas[0]) * flow``. The remaining steps run
    the stock Euler update on the frozen base (LoRA scale 0).

    The wrap hands ``predict_flow`` the scalar step timestep exactly as the
    stock loop does, so everything the transformer layers underneath -- the
    per-token AR0 rewrite (first-frame tokens pinned to the stabilization
    timestep) and the once-per-chunk memory KV prefill, which fires inside
    the jump forward and therefore runs LoRA-scaled like the trainer's
    student prefill -- composes unchanged.
    """
    scheduler = dm.scheduler
    timesteps, sigmas = scheduler.timesteps, scheduler.sigmas
    n_steps = scheduler.config.num_inference_steps
    assert n_steps == 4 and int(timesteps[0]) == 1000, (
        "the hybrid jump is trained against the distilled 4-step schedule"
    )
    orig_sample = scheduler.sample

    def hybrid_sample(
        initial_noise: Tensor,
        predict_flow: Any,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        gain = mode["jump_gain"]
        if gain is None:
            return orig_sample(
                initial_noise=initial_noise, predict_flow=predict_flow, rng=rng
            )
        # ``rng`` is unused past this point (deterministic ODE), matching
        # the stock sampler.
        dtype = initial_noise.dtype
        set_lora_scale(network, gain)
        flow = predict_flow(initial_noise, timesteps[0].to(dtype))
        z = initial_noise + (sigmas[JUMP_END] - sigmas[0]).to(dtype) * flow
        set_lora_scale(network, 0.0)
        for i in range(JUMP_END, n_steps):
            flow = predict_flow(z, timesteps[i].to(dtype))
            z = z + (sigmas[i + 1] - sigmas[i]).to(dtype) * flow
        return z.to(dtype)

    scheduler.sample = hybrid_sample


def timed_rollout(runner, *, noise_seed: int, mp4_path: Path) -> list[float]:
    """One MP4-writing rollout; returns per-chunk seconds (CUDA-synced).

    Reproduces :func:`_rollout.capture_rollout`'s run flow (bindings, RNG
    re-seed, MP4 write) with :mod:`bench_latency`-style per-chunk timing
    around ``generate`` + ``finalize`` instead of snapshot capture.
    """
    pipe = runner.pipeline
    cfg = runner.config
    device = next(pipe.parameters()).device
    dtype = next(pipe.parameters()).dtype
    assert cfg.image_path is not None
    image = preprocess_first_frame(
        cfg.image_path, cfg.pixel_height, cfg.pixel_width
    ).to(device=device, dtype=dtype)
    cache = pipe.initialize_cache(text=[_resolve_prompt(cfg.prompt)], image=image)
    runner._bind_action_labels()
    runner._bind_camera_data()
    torch.manual_seed(_POINT_CLOUD_SEED)
    runner._bind_memory_config(device=device)
    pipe.diffusion_model._rng = torch.Generator(device=device).manual_seed(noise_seed)

    times: list[float] = []
    chunks: list[Tensor] = []
    for ar_idx in range(cfg.num_chunk):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        chunks.append(pipe.generate(ar_idx, cache))
        pipe.finalize(ar_idx, cache)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    _write_mp4(torch.cat(chunks, dim=-4), mp4_path, fps=cfg.fps)
    del cache, chunks
    gc.collect()
    torch.cuda.empty_cache()
    return times


def write_latency(chunk_times: dict[str, list[float]], nfe: dict[str, int]) -> None:
    """Write per-config steady-state chunk wall time to ``latency.json``."""
    if not any(chunk_times.values()):
        print("latency: no fresh rollouts, keeping any existing latency.json")
        return
    stats = {
        config: {
            "per_chunk_ms_mean": 1e3 * sum(ts) / len(ts),
            "chunks": len(ts),
            "nfe_per_chunk": nfe[config],
        }
        for config, ts in chunk_times.items()
        if ts
    }
    base = stats.get("base")
    for config, s in stats.items():
        if base is not None and config != "base":
            s["vs_base"] = s["per_chunk_ms_mean"] / base["per_chunk_ms_mean"]
    payload = {
        "protocol": {
            "num_chunk": NUM_CHUNK,
            "warm_chunks": WARM_CHUNKS,
            "note": "CUDA-synced generate+finalize wall clock, chunks 5+ only",
        },
        "configs": stats,
    }
    out = OUT_DIR / "latency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"latency -> {out}", flush=True)


def main() -> None:
    torch.set_grad_enabled(False)
    images: list[Path | None] = [None]
    if IMAGES_DIR:
        p = Path(IMAGES_DIR)
        images = sorted(p.glob("*.png")) + sorted(p.glob("*.jpg"))
        assert images, f"no images under {p}"
        if N_IMAGES:
            images = images[:N_IMAGES]

    configs = build_configs()
    runner = None
    network = None
    mode: dict = {"jump_gain": None}
    chunk_times: dict[str, list[float]] = {c: [] for c in configs}
    for config, gain in configs.items():
        for pi, pose in enumerate(POSES):
            for ii, image_path in enumerate(images):
                for seed in SEEDS:
                    name = f"p{pi}_i{ii}_s{seed}"
                    mp4 = OUT_DIR / config / f"{name}.mp4"
                    if mp4.exists():
                        print(f"{config}/{name}: exists, skipping", flush=True)
                        continue
                    if runner is None:
                        runner = build_runner(
                            num_chunk=NUM_CHUNK,
                            pose=pose,
                            output_dir=OUT_DIR,
                            image_path=image_path,
                        )
                        network = unwrap_compiled(
                            runner.pipeline.diffusion_model.transformer.network
                        )
                        apply_lora(network)  # r16, matching the checkpoint
                        load_lora(network, LORA)
                        if PROMPT:
                            runner.config.prompt = PROMPT
                        install_hybrid_sampler(
                            runner.pipeline.diffusion_model, network, mode
                        )
                    else:
                        runner.config.pose = pose
                        if image_path is not None:
                            runner.config.image_path = image_path
                    assert network is not None  # bound with the runner
                    mode["jump_gain"] = gain
                    if gain is None:
                        set_lora_scale(network, 0.0)
                    times = timed_rollout(runner, noise_seed=seed, mp4_path=mp4)
                    steady = times[WARM_CHUNKS:] or times
                    chunk_times[config] += steady
                    print(
                        f"{config}/{name}: {NUM_CHUNK} chunks in {sum(times):.1f}s"
                        f" | steady {1e3 * sum(steady) / len(steady):.0f} ms/chunk",
                        flush=True,
                    )
    nfe = {c: 4 if g is None else 3 for c, g in configs.items()}
    write_latency(chunk_times, nfe)
    print("HYBRID-EVAL-DONE", flush=True)


if __name__ == "__main__":
    main()

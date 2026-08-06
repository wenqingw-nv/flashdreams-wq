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

"""Demo eval: 1-NFE mean-velocity corrector vs the 4-step base on LingBot.

Rolls the held-out eval grammars (same scenes / trajectories / matched
seeds as :mod:`eval_rollouts`) and writes one MP4 per scene x trajectory x
config as ``{scene}_{traj}_{config}.mp4`` under ``OUT``:

- ``base4``: stock 4-step pipeline, no LoRA (the quality reference).
- ``nfe1corr``: scheduler overridden to ONE solver step (t=1000 -- the
  4-step schedule's first step; shift/sigma bounds/timestep dtype
  unchanged) with the v1-mean corrector LoRA (``train_v1_mean.py``, window
  ``[0:4)``) at scale 1.0. The corrector was trained to jump straight to
  the frozen model's integrated full-window prediction, so its single
  forward per chunk should recover most of the 4-step quality.
- ``nfe1base``: the same 1-step scheduler, NO LoRA -- the ablation showing
  the raw 1-step quality drop the corrector must beat.

``GAINS`` (comma-separated floats, default ``1.0``) sweeps the corrector's
LoRA scale: each gain ``g`` yields a 1-step + LoRA config named
``nfe1corr{g:.2f}`` with the dot dropped (``GAINS=0.4`` -> ``nfe1corr040``);
``g == 1.0`` keeps the plain ``nfe1corr`` name so existing MP4s are reused.

Two 14B pipelines do not fit VRAM together, so pipelines are built
sequentially per config (destroyed and cache-emptied in between); finished
MP4s are skipped on re-run, and a config whose MP4s all exist skips the
build entirely. Each chunk's ``generate`` + ``finalize`` wall-time is
measured (CUDA-synced ``time.perf_counter``) and the per-config mean over
chunks 5+ (warmup excluded; freshly rolled cells only) is printed and
merged into ``OUT/latency.json``.

Run from the flashdreams repo root::

    .venv/bin/python integrations/lingbot/drift_correction/eval_nfe1.py
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from _lora import apply_lora, load_lora, set_lora_scale, unwrap_compiled
from _rollout import Scene, build_pipeline, camera_stream, chunk_camctrl, load_scene
from _traj import grammar_poses
from eval_rollouts import EVAL_GRAMMARS, NUM_CHUNK, SCENES, SEED0
from flashdreams.infra.config import derive_config
from lingbot.config import PIPELINE_LINGBOT_WORLD_FAST
from lingbot.encoder.camctrl import CamCtrlInput
from lingbot.pipeline import LingbotWorldInferencePipeline
from lingbot.runner import _write_video

## Eval configuration

_BASE = Path("integrations/lingbot/drift_correction")

LORA = os.environ.get("LORA", str(_BASE / "outputs/lora_v1_mean_w04.pt"))
"""Mean-velocity corrector checkpoint (``train_v1_mean.py`` format: dict
with index-keyed ``lora`` tensors, rank 16)."""

GAINS = tuple(float(g) for g in os.environ.get("GAINS", "1.0").split(",") if g.strip())
"""Corrector LoRA scales; each adds a ``nfe1corr*`` config to the grid."""


def corr_config(gain: float) -> str:
    """``nfe1corr`` config name for a LoRA scale (``nfe1corr040`` for 0.4).

    Scale 1.0 keeps the historical plain ``nfe1corr`` name so MP4s from
    earlier runs are still recognized and skipped.
    """
    if gain == 1.0:
        return "nfe1corr"
    return f"nfe1corr{gain:.2f}".replace(".", "")


GRID: dict[str, tuple[bool, float | None]] = {
    "base4": (False, None),
    **{corr_config(g): (True, g) for g in GAINS},
    "nfe1base": (True, None),
}
"""Config name -> ``(one_step, lora_scale)``; ``None`` scale means no LoRA."""

CONFIGS = tuple(
    c.strip() for c in os.environ.get("CONFIGS", ",".join(GRID)).split(",") if c.strip()
)
"""Subset of the config grid (``base4``, ``nfe1corr*`` per gain, ``nfe1base``)."""

OUT_DIR = Path(os.environ.get("OUT", str(_BASE / "outputs/eval_nfe1")))

WARMUP_CHUNKS = 5
"""Chunks excluded from the latency mean (cache fill + first-chunk jitter)."""


def build_pipeline_nfe1(seed: int | None = 42) -> LingbotWorldInferencePipeline:
    """1-step variant of :func:`_rollout.build_pipeline`.

    Only the scheduler differs: one solver step at t=1000, which is exactly
    the 4-step host schedule's first ``(t, sigma)`` (shift, sigma bounds,
    and timestep dtype are untouched), so the corrector sees the timestep
    it was trained at. Compile and CUDA graphs stay off as in
    :func:`_rollout.build_pipeline`, keeping numerics identical to the
    training captures.
    """
    cfg = derive_config(
        PIPELINE_LINGBOT_WORLD_FAST,
        enable_sync_and_profile=False,
        diffusion_model=dict(
            seed=seed,
            transformer=dict(compile_network=False, use_cuda_graph=False),
            scheduler=dict(num_inference_steps=1, denoising_timesteps=[1000]),
        ),
    )
    pipe = cfg.setup()
    assert isinstance(pipe, LingbotWorldInferencePipeline)
    return pipe.to("cuda")


def attach_corrector(pipe: LingbotWorldInferencePipeline, scale: float) -> None:
    """Apply the r16 LoRA to the unwrapped DiT and load ``LORA`` at ``scale``."""
    network = unwrap_compiled(pipe.diffusion_model.transformer.network)
    apply_lora(network, rank=16)
    load_lora(network, LORA)
    set_lora_scale(network, scale)


def timed_rollout(
    pipe: LingbotWorldInferencePipeline,
    scene: Scene,
    camctrls: list[CamCtrlInput],
    noise_seed: int,
) -> tuple[torch.Tensor, list[float]]:
    """Seeded decode rollout with per-chunk wall-times.

    The :func:`_rollout.capture_rollout` loop minus the latent snapshots,
    plus a CUDA-synced ``time.perf_counter`` bracket around each chunk's
    ``generate`` + ``finalize``.

    Returns:
        ``(video, chunk_seconds)``; ``video`` is ``[T, C, H, W]`` on cpu.
    """
    device = pipe.device
    cache = pipe.initialize_cache(text=[scene.prompt], image=scene.first_frames_t)
    pipe.diffusion_model._rng = torch.Generator(device=device).manual_seed(noise_seed)
    chunks: list[torch.Tensor] = []
    times: list[float] = []
    for i, ctrl in enumerate(camctrls):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        video_chunk = pipe.generate(autoregressive_index=i, cache=cache, input=ctrl)
        pipe.finalize(autoregressive_index=i, cache=cache)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        chunks.append(video_chunk.detach().cpu())
    video = torch.cat(chunks, dim=0)
    del cache
    torch.cuda.empty_cache()
    return video, times


def main() -> None:
    torch.set_grad_enabled(False)

    unknown = [c for c in CONFIGS if c not in GRID]
    assert not unknown, f"unknown CONFIGS entries {unknown}; grid is {list(GRID)}"
    assert NUM_CHUNK > WARMUP_CHUNKS, (
        f"NUM_CHUNK={NUM_CHUNK} leaves no chunks past warmup ({WARMUP_CHUNKS})"
    )

    scenes: dict[int, Scene] = {}
    camctrls_by_cell: dict[tuple[int, str], list[CamCtrlInput]] = {}

    def cell_runtime(
        pipe: LingbotWorldInferencePipeline, scene_idx: int, traj: str, grammar: str
    ):
        """Scene + per-chunk camera payloads, memoized across configs.

        Safe to share across pipelines: ``chunk_camctrl`` uses ``pipe`` only
        for ``get_num_output_frames``, which the scheduler override does not
        change.
        """
        if scene_idx not in scenes:
            scenes[scene_idx] = load_scene(scene_idx)
        key = (scene_idx, traj)
        if key not in camctrls_by_cell:
            poses_t, intr_t, ws = camera_stream(
                scenes[scene_idx], grammar_poses(grammar)
            )
            camctrls = chunk_camctrl(poses_t, intr_t, ws, pipe, n_chunks=NUM_CHUNK)
            assert len(camctrls) == NUM_CHUNK, (
                f"grammar {traj!r} covers only {len(camctrls)} of {NUM_CHUNK} "
                f"chunks; extend it past {9 + (NUM_CHUNK - 1) * 12} frames"
            )
            camctrls_by_cell[key] = camctrls
        return scenes[scene_idx], camctrls_by_cell[key]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    latency_path = OUT_DIR / "latency.json"
    latency = json.loads(latency_path.read_text()) if latency_path.exists() else {}

    for config in CONFIGS:
        one_step, lora_scale = GRID[config]
        cells = []
        for scene_idx in SCENES:
            for ti, (traj, grammar) in enumerate(EVAL_GRAMMARS):
                name = f"s{scene_idx}_{traj}"
                mp4 = OUT_DIR / f"{name}_{config}.mp4"
                if mp4.exists():
                    print(f"{config}/{name}: exists, skipping", flush=True)
                    continue
                cells.append((scene_idx, ti, traj, grammar, name, mp4))
        if not cells:
            continue

        pipe = build_pipeline_nfe1() if one_step else build_pipeline()
        # Pin the one-shot text/image encoders: ``initialize_cache``'s
        # default release/reload cycle would otherwise re-load UMT5 from
        # checkpoint on every rollout (the train_v1.py pattern).
        pipe._ensure_oneshot_encoders_loaded()
        encoders = (pipe.text_encoder, pipe.image_encoder)
        if lora_scale is not None:
            attach_corrector(pipe, scale=lora_scale)

        chunk_times: list[float] = []
        for scene_idx, ti, traj, grammar, name, mp4 in cells:
            scene, camctrls = cell_runtime(pipe, scene_idx, traj, grammar)
            seed = SEED0 + 10 * scene_idx + ti
            pipe.text_encoder, pipe.image_encoder = encoders
            print(
                f"{config}/{name}: rolling {NUM_CHUNK} chunks (seed {seed}) ...",
                flush=True,
            )
            video, times = timed_rollout(pipe, scene, camctrls, noise_seed=seed)
            _write_video(video.permute(0, 2, 3, 1), mp4, fps=16)
            chunk_times += times[WARMUP_CHUNKS:]

        mean_s = sum(chunk_times) / len(chunk_times)
        latency[config] = {"mean_chunk_s": mean_s, "n_chunks": len(chunk_times)}
        latency_path.write_text(json.dumps(latency, indent=2) + "\n")
        print(
            f"{config}: mean chunk wall-time {mean_s:.3f}s over "
            f"{len(chunk_times)} chunks (>= {WARMUP_CHUNKS})",
            flush=True,
        )

        # One 14B pipeline at a time: drop everything before the next build.
        del pipe, encoders
        gc.collect()
        torch.cuda.empty_cache()

    print("NFE1-EVAL-DONE", flush=True)


if __name__ == "__main__":
    main()

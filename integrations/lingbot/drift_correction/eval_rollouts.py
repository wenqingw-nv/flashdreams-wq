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

"""Closed-loop eval rollouts on LingBot: frozen base vs +corrector vs gated.

Generates long held-out rollouts (non-loop eval grammars, matched seeds,
40 chunks = 477 frames = ~30 s at 16 fps -- past the 29-chunk training
horizon, to expose late-horizon drift) for each config and writes one MP4
per scene x trajectory x config as ``{scene}_{traj}_{config}.mp4`` under
``OUT``. Scoring (Delta-MUSIQ, dynamics guard, strips) is a separate
pass -- see ``score_drift.py``.

Configs: ``base`` (LoRA scale 0), ``corr`` (scale 1.0, set once), and
``corrgate`` (per-solver-step ``alpha*(t) x 1.0`` from the faithful gate's
unbiased profile, plus the context alpha on the ``finalize_kv_cache``
forward). The LoRA is applied once and only the runtime scale varies
between configs (the unfused path -- fine for eval; deploy pre-merges).

Run from the flashdreams repo root::

    .venv/bin/python integrations/lingbot/drift_correction/eval_rollouts.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from _lora import apply_lora, load_lora, set_lora_scale, unwrap_compiled
from _rollout import (
    Scene,
    build_pipeline,
    camera_stream,
    capture_rollout,
    chunk_camctrl,
    load_scene,
)
from _traj import grammar_poses
from lingbot.encoder.camctrl import CamCtrlInput
from lingbot.pipeline import LingbotWorldInferencePipeline
from lingbot.runner import _write_video

## Eval configuration

_BASE = Path("integrations/lingbot/drift_correction")

LORA = os.environ.get("LORA", str(_BASE / "outputs/lora_v1.pt"))
"""Corrector checkpoint (``train_v1.py`` format: dict with index-keyed
``lora`` tensors)."""

CONFIGS = tuple(
    c.strip()
    for c in os.environ.get("CONFIGS", "base,corr,corrgate").split(",")
    if c.strip()
)
"""Subset of the config grid ``{base, corr, corrgate}`` to run."""

GAINS = tuple(float(g) for g in os.environ.get("GAINS", "").split(",") if g.strip())
"""Extra flat-gain configs (``GAINS=0.5,0.7`` adds ``corr050``, ``corr070``)."""

NUM_CHUNK = int(os.environ.get("NUM_CHUNK", "40"))
"""Rollout horizon in AR chunks (40 = 477 decoded frames; the training
pairs stop at 29, so the tail is unseen drift depth)."""

SCENES = tuple(int(s) for s in os.environ.get("SCENES", "0,1,2,3,4,5").split(","))
"""Bundled example scenes to roll out."""

SEED0 = int(os.environ.get("SEED0", "52000"))
"""Base diffusion seed; cell (scene, traj) uses ``SEED0 + 10*scene + traj``
in every config (matched seeds across configs)."""

OUT_DIR = Path(os.environ.get("OUT", str(_BASE / "outputs/eval_v1")))

GATE_JSON = _BASE / "outputs/gate/gate_faithful.json"
"""alpha*(t) profile from ``gate_faithful.py`` (per-timestep summary)."""

EVAL_GRAMMARS = [
    ("walkturn", "w-120, a-24, w-120, d-24, w-120, a-24, w-60"),
    ("strafesweep", "we-96, w-96, wq-96, w-96, we-96"),
    ("fwdpitch", "w-120, wi-16, w-80, wk-32, w-80, wi-16, w-140"),
]
"""Held-out eval trajectories (name, grammar): a long forward walk with
alternating 48-degree yaw turns, a forward strafe sweep (sustained diagonal
drift, never doubling back), and a forward walk with a +-32-degree pitch
sweep. All are open, progressing paths -- deliberately NOT the training
out-and-back strafe loops (``e-N, q-N``), so any learned repeat prior shows
up as a failure here instead of a win. Each grammar must cover
``9 + (NUM_CHUNK - 1) * 12`` pose frames (477 at 40 chunks; asserted).
Trajectory names feed the ``{scene}_{traj}_{config}.mp4`` naming that
``score_drift.py`` parses, so they must not contain underscores."""


def load_gate_alpha(path: Path) -> dict[float, float]:
    """Read the unbiased alpha*(t) profile from the faithful-gate JSON.

    Uses ``alpha_star_unbiased`` (the deploy convention on the HY and
    Omnidreams hosts: the biased estimator inflates alpha at low cell
    counts).
    """
    profile = json.loads(path.read_text())
    return {float(t): v["alpha_star_unbiased"] for t, v in profile.items()}


def install_alpha_gate(
    pipe: LingbotWorldInferencePipeline,
    network: torch.nn.Module,
    mode: dict,
    gate_alpha: dict[float, float],
) -> None:
    """Install the CPU-side per-step alpha*(t) gate on ``pipe``.

    Mirrors the shipped OD/HY deploy hooks: ``scheduler.sample`` makes
    exactly one ``predict_flow`` call per solver step, so each step's alpha
    resolves by call index from the load-time schedule (no GPU timestep
    readback), and ``finalize_kv_cache`` gets the context alpha -- nearest
    to ``t = context_noise`` (0 on this host, resolving to the lowest gate
    timestep, t=825). Transparent unless ``mode["gain"]`` is a
    ``("gate", scale)`` tuple; flat-gain configs set the scale once outside.
    """
    dm = pipe.diffusion_model

    def nearest(t: float) -> float:
        return min(gate_alpha.items(), key=lambda kv: abs(kv[0] - t))[1]

    step_alphas = [nearest(float(t)) for t in dm.scheduler.denoising_step_list.tolist()]
    ctx_alpha = nearest(float(dm.config.context_noise))
    orig_sample = dm.scheduler.sample

    def gated_sample(initial_noise, predict_flow, rng=None):
        gain = mode["gain"]
        if not isinstance(gain, tuple):
            return orig_sample(
                initial_noise=initial_noise, predict_flow=predict_flow, rng=rng
            )
        calls = [0]

        def pf(noisy, timestep):
            assert calls[0] < len(step_alphas), "predict_flow calls > solver steps"
            set_lora_scale(network, step_alphas[calls[0]] * gain[1])
            calls[0] += 1
            return predict_flow(noisy, timestep)

        return orig_sample(initial_noise=initial_noise, predict_flow=pf, rng=rng)

    dm.scheduler.sample = gated_sample
    transformer = dm.transformer
    orig_finalize = transformer.finalize_kv_cache

    def gated_finalize(*args, **kwargs):
        gain = mode["gain"]
        if isinstance(gain, tuple):
            set_lora_scale(network, ctx_alpha * gain[1])
        return orig_finalize(*args, **kwargs)

    transformer.finalize_kv_cache = gated_finalize


def main() -> None:
    torch.set_grad_enabled(False)

    grid: dict[str, float | tuple[str, float]] = {
        "base": 0.0,
        "corr": 1.0,
        "corrgate": ("gate", 1.0),
    }
    unknown = [c for c in CONFIGS if c not in grid]
    assert not unknown, f"unknown CONFIGS entries {unknown}; grid is {list(grid)}"
    configs: dict[str, float | tuple[str, float]] = {c: grid[c] for c in CONFIGS}
    for g in GAINS:
        configs[f"corr{g:.2f}".replace(".", "")] = g

    pipe = build_pipeline()
    # Pin the one-shot text/image encoders: ``initialize_cache``'s default
    # release/reload cycle would otherwise re-load UMT5 from checkpoint on
    # every rollout (the train_v1.py pattern).
    pipe._ensure_oneshot_encoders_loaded()
    encoders = (pipe.text_encoder, pipe.image_encoder)
    network = unwrap_compiled(pipe.diffusion_model.transformer.network)
    apply_lora(network)
    load_lora(network, LORA)
    mode: dict = {"gain": 0.0}
    if any(isinstance(g, tuple) for g in configs.values()):
        install_alpha_gate(pipe, network, mode, load_gate_alpha(GATE_JSON))

    scenes: dict[int, Scene] = {}
    camctrls_by_cell: dict[tuple[int, str], list[CamCtrlInput]] = {}

    def cell_runtime(scene_idx: int, traj: str, grammar: str):
        """Scene + per-chunk camera payloads, memoized across configs."""
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
    for config, gain in configs.items():
        for scene_idx in SCENES:
            for ti, (traj, grammar) in enumerate(EVAL_GRAMMARS):
                name = f"s{scene_idx}_{traj}"
                mp4 = OUT_DIR / f"{name}_{config}.mp4"
                if mp4.exists():
                    print(f"{config}/{name}: exists, skipping", flush=True)
                    continue
                scene, camctrls = cell_runtime(scene_idx, traj, grammar)
                mode["gain"] = gain
                if not isinstance(gain, tuple):
                    set_lora_scale(network, gain)
                seed = SEED0 + 10 * scene_idx + ti
                pipe.text_encoder, pipe.image_encoder = encoders
                print(
                    f"{config}/{name}: rolling {NUM_CHUNK} chunks (seed {seed}) ...",
                    flush=True,
                )
                _, video = capture_rollout(
                    pipe, scene, camctrls, noise_seed=seed, decode=True
                )
                assert video is not None
                _write_video(video.permute(0, 2, 3, 1), mp4, fps=16)
    print("EVAL-DONE", flush=True)


if __name__ == "__main__":
    main()

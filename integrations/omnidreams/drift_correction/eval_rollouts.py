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

"""Closed-loop gain-sweep rollouts: frozen base vs +v1 corrector (Omnidreams).

Deployment-realistic 21.5 s rollouts (non-tiled HDMap conditioning, real
photo seeds, matched prompts) over the standard dial grid
``{base, 0.5, alpha*(t) gate, gate x 0.5, 1.0}``. Scenarios: the
collapse-prone clip-0 scene (stress case), the training-val scene, and two
fresh sample clips never seen by training. MP4s land under
``EVAL_OUT/{config}/`` for the HY host-agnostic scoring + sbs pass.

Run from the flashdreams repo root::

    HF_TOKEN=... .venv/bin/python integrations/omnidreams/drift_correction/eval_rollouts.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from _host import build_pipeline, capture_rollout
from _lora import apply_lora, load_lora, set_lora_scale, unwrap_compiled
from build_pairs import _clip_prompt, _list_sample_uuids
from omnidreams.runner import (
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    _ensure_hf_single_view_example_data_synced,
    _load_first_frame,
    _load_video,
    _write_video,
)

## Sweep configuration

_BASE = Path("integrations/omnidreams/drift_correction")

LORA = os.environ.get("LORA", str(_BASE / "outputs/lora_v1.pt"))
"""Corrector checkpoint (``train_v1.py`` format: dict with ``lora``)."""

OUT_DIR = Path(os.environ.get("EVAL_OUT", str(_BASE / "outputs/eval_sweep")))

NUM_CHUNK = int(os.environ.get("NUM_CHUNK", "81"))
"""81 chunks = 645 frames = 21.5 s (matches the gate horizon)."""

SCENARIO_IDS = tuple(
    int(i) for i in os.environ.get("SCENARIO_IDS", "0,5,6,7").split(",")
)
"""Indices into the sorted sample-UUID list: 0 = collapse-prone stress
scene, 5 = training val scene, 6-7 = never-seen scenarios."""

SEED = int(os.environ.get("SEED", "5042"))

ALPHA_STAR = {
    float(kv.split(":")[0]): float(kv.split(":")[1])
    for kv in os.environ.get("ALPHA_STAR", "1000:0.96,803:0.667").split(",")
}
"""Unbiased alpha* per warped solver timestep (default: pairs-v2 photoreal
gate; override as ``ALPHA_STAR=t:a,t:a``); nearest-t lookup, matching the
HY deploy convention."""


def install_alpha_gate(transformer, network, mode: dict) -> None:
    """Wrap ``predict_flow`` so gate configs rescale the LoRA every step."""
    orig_pf = transformer.predict_flow

    def gated_pf(*args, **kwargs):
        gain = mode["gain"]
        if isinstance(gain, tuple):
            # finalize_kv_cache calls positionally: (noisy_latent, timestep, ...)
            ts = kwargs.get("timestep", args[1] if len(args) > 1 else None)
            t = float(ts.reshape(-1).max())
            alpha = min(ALPHA_STAR.items(), key=lambda kv: abs(kv[0] - t))[1]
            set_lora_scale(network, alpha * gain[1])
        return orig_pf(*args, **kwargs)

    transformer.predict_flow = gated_pf


def main() -> None:
    torch.set_grad_enabled(False)
    dtype = torch.bfloat16
    uuids = _list_sample_uuids(max(SCENARIO_IDS) + 1)

    # Load every scenario's inputs BEFORE any model work (ffmpeg decode
    # fails silently once the process reaches rollout size on this box).
    inputs = []
    for sid in SCENARIO_IDS:
        uuid = uuids[sid]
        (hdmap_path,), (frame_path,) = _ensure_hf_single_view_example_data_synced(uuid)
        hdmap = _load_video(
            hdmap_path,
            pixel_height=DEFAULT_VIDEO_HEIGHT,
            pixel_width=DEFAULT_VIDEO_WIDTH,
            device="cpu",
            dtype=dtype,
        )
        n_frames = 5 + (NUM_CHUNK - 1) * 8
        assert hdmap.shape[0] >= n_frames, (sid, hdmap.shape)
        first = _load_first_frame(
            frame_path,
            pixel_height=DEFAULT_VIDEO_HEIGHT,
            pixel_width=DEFAULT_VIDEO_WIDTH,
            device="cpu",
            dtype=dtype,
        )[None, :, None]
        inputs.append((sid, uuid, _clip_prompt(uuid), hdmap[:n_frames][None, None], first))
        print(f"loaded scenario {sid} ({uuid})", flush=True)

    pipe = build_pipeline(with_oneshot_encoders=True)
    device = pipe.device
    transformer = pipe.diffusion_model.transformer
    network = unwrap_compiled(transformer.network)
    apply_lora(network)
    load_lora(network, LORA)
    mode: dict = {"gain": 0.0}
    install_alpha_gate(transformer, network, mode)

    configs: dict[str, float | tuple[str, float]] = {
        "base": 0.0,
        "corr050": 0.5,
        "corrgate": ("gate", 1.0),
        "corrgate050": ("gate", 0.5),
        "corr": 1.0,
    }

    # Cache per-scenario embeddings once (encoders stay loaded).
    embeddings = {}
    for sid, uuid, prompt, hdmap, first in inputs:
        embeddings[sid] = pipe.precompute_embeddings(
            text=[[prompt]], image=first.to(device)
        )

    for config, gain in configs.items():
        for sid, uuid, prompt, hdmap, first in inputs:
            mp4 = OUT_DIR / config / f"scen{sid}_s{SEED}.mp4"
            if mp4.exists():
                print(f"{config}/scen{sid}: exists, skipping", flush=True)
                continue
            mode["gain"] = gain
            if not isinstance(gain, tuple):
                set_lora_scale(network, gain)
            emb = embeddings[sid]
            cache = pipe.initialize_cache_from_embeddings(
                text_embeddings=emb["text_embeddings"],
                image_embeddings=emb["image_embeddings"],
            )
            print(f"{config}/scen{sid}: rolling {NUM_CHUNK} chunks ...", flush=True)
            _, video = capture_rollout(
                pipe,
                cache,
                hdmap_video=hdmap.to(device),
                num_chunk=NUM_CHUNK,
                noise_seed=SEED,
            )
            mp4.parent.mkdir(parents=True, exist_ok=True)
            _write_video(video[0, 0].permute(0, 2, 3, 1), mp4, fps=30)
    print("SWEEP-DONE", flush=True)


if __name__ == "__main__":
    main()

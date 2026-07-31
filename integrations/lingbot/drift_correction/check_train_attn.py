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

"""Parity check: functional training attention vs the stock probe path.

Builds the v1 pipeline, teacher-forces one pair clip's history into a fresh
KV cache at a deep (steady-state) probe position, and computes ``xhat0``
through the stock cache-writing attention (``probe_xhat0``) and through the
grad-friendly functional path (:mod:`_train_attn`) with the LoRA wrapped
but inactive (scale 0). The two must agree numerically: the training
gradients are only meaningful if the functional path computes the same
function the deployed model does.

The stock probe writes the current chunk into buffer slots the functional
probe never reads, so both run against the same rebuilt cache inside one
probe bracket. Bar (matching the HY port's verification): mean relative
delta < 0.02 -> PARITY-PASS.

Run after ``build_pairs.py`` from the repo root::

    .venv/bin/python integrations/lingbot/drift_correction/check_train_attn.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from _lora import apply_lora, set_lora_scale, unwrap_compiled
from _rollout import (
    build_pipeline,
    camera_stream,
    chunk_camctrl,
    load_scene,
    probe_xhat0,
    rebuild_cache,
)
from _train_attn import functional_attention, patch_functional_attention

PAIRS_DIR = Path("integrations/lingbot/drift_correction/outputs/pairs")

BAR = 0.02
"""Mean-relative-delta parity bar."""


def main() -> None:
    torch.set_grad_enabled(False)
    clips = sorted(PAIRS_DIR.glob("clip_*.pt"))
    assert clips, f"no clips under {PAIRS_DIR}; run build_pairs.py first"
    d = torch.load(clips[0], map_location="cpu", weights_only=False)

    pipe = build_pipeline()
    scene = load_scene(int(d["scene_idx"]))
    poses_t, intr_t, ws = camera_stream(scene, d["poses"])
    camctrls = chunk_camctrl(poses_t, intr_t, ws, pipe, n_chunks=int(d["num_chunk"]))
    n_steps = int(pipe.diffusion_model.scheduler.denoising_step_list.shape[0])

    network = unwrap_compiled(pipe.diffusion_model.transformer.network)
    apply_lora(network, rank=16)
    set_lora_scale(network, 0.0)  # inactive: parity is against the frozen base
    patch_functional_attention()  # toggle off by default: the rebuild stays stock

    # Deepest position by default: exercises the steady-state roll + prefix.
    k = int(os.environ.get("K", int(d["num_chunk"]) - 1))
    lat = d["latents"]
    cache = rebuild_cache(pipe, scene, lat[:k], camctrls[:k])
    x0 = lat[k]
    print(f"{clips[0].stem}: rebuilt {k} chunks, probing chunk {k}", flush=True)

    rels = []
    for t_idx in range(n_steps):
        eps = torch.randn(
            x0.shape, generator=torch.Generator().manual_seed(60_000 + t_idx)
        )
        stock = probe_xhat0(pipe, cache, k, camctrls[k], x0, t_idx, eps)
        with functional_attention():
            func = probe_xhat0(pipe, cache, k, camctrls[k], x0, t_idx, eps)
        scale = stock.abs().mean()
        rel_mean = float((func - stock).abs().mean() / scale)
        rel_max = float((func - stock).abs().max() / scale)
        rels.append(rel_mean)
        print(
            f"t_idx={t_idx}: rel mean {rel_mean:.2e} | rel max {rel_max:.2e}",
            flush=True,
        )

    mean_rel = sum(rels) / len(rels)
    verdict = "PARITY-PASS" if mean_rel < BAR else "PARITY-FAIL"
    print(f"mean relative delta {mean_rel:.2e} (bar {BAR}) -> {verdict}", flush=True)
    print("PARITY-DONE", flush=True)


if __name__ == "__main__":
    main()

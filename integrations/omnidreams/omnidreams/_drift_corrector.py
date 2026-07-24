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

"""Clean Forcing drift corrector for the Omnidreams runner.

Deploys the trained corrector LoRA (``drift_correction/train_v2.py``
checkpoints) on a built :class:`~omnidreams.runner.OmnidreamsRunner`'s
pipeline at ``alpha*(t) * gain`` per denoise step. Shipped config (2026-07-24): ``lora_v2_v3_valpeak.pt`` at gain 0.25
(``corrgate025`` — best trees/foliage detail and consistency, drift
Delta +0.99 vs base +2.44). Mirrors the HY-WorldPlay deploy module
(``hy_worldplay/_drift_corrector.py``); self-contained so the production
runner does not import the research directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

## Deploy policy

GATE_ALPHA = {1000.0: 0.96, 803.0: 0.667}
"""Unbiased alpha*(t) from the step-0 systematicity gate on photoreal
drift pairs (drift_correction's ``outputs/gate/gate_faithful_v2.json``):
the systematic fraction of the drift-induced error at each of the two
distilled solver timesteps. The corrector LoRA is rescaled to
``alpha*(t) * gain`` before every denoise step (nearest-t lookup); the
``finalize_kv_cache`` context forward (t=128) resolves to the t=803 entry,
matching the evaluated deploy configs."""

_LORA_TARGETS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.output_proj",
)
"""Self-attention projections the corrector checkpoints were trained on."""

_LORA_RANK = 16
"""Rank of the shipped corrector checkpoints."""


class _LoRALinear(nn.Module):
    """Frozen base linear plus a runtime-gated low-rank delta.

    Mirrors the training-side module in
    ``integrations/omnidreams/drift_correction/_lora.py``: ``scale`` is the
    runtime gain (``0`` = exact base output), and the A/B path runs in fp32
    regardless of the base dtype.
    """

    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Linear(base.in_features, rank, bias=False)
        self.B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.zeros_(self.B.weight)
        self.scale = 0.0

    def forward(self, x: Tensor) -> Tensor:
        out = self.base(x)
        if self.scale != 0:
            delta = self.B(self.A(x.to(self.A.weight.dtype)))
            out = out + self.scale * delta.to(out.dtype)
        return out


def _apply_lora(network: nn.Module) -> list[nn.Parameter]:
    """Wrap the target linears and return the LoRA parameters in load order."""
    for mname, module in list(network.named_modules()):
        for cname, child in list(module.named_children()):
            full = f"{mname}.{cname}" if mname else cname
            # Substring match, exactly as the training-side apply_lora, so
            # the wrap set and load order match the checkpoint indices.
            if isinstance(child, nn.Linear) and any(t in full for t in _LORA_TARGETS):
                setattr(
                    module,
                    cname,
                    _LoRALinear(child, _LORA_RANK).to(child.weight.device),
                )
    params: list[nn.Parameter] = []
    for m in network.modules():
        if isinstance(m, _LoRALinear):
            params += list(m.A.parameters()) + list(m.B.parameters())
    return params


def _set_scale(network: nn.Module, scale: float) -> None:
    """Set the runtime gain on every wrapped linear."""
    for m in network.modules():
        if isinstance(m, _LoRALinear):
            m.scale = scale


def apply_drift_corrector(runner: Any, checkpoint: Path, gain: float) -> str:
    """Deploy the corrector LoRA on ``runner`` with the alpha*(t) gate.

    Args:
        runner: A built ``OmnidreamsRunner``.
        checkpoint: Corrector LoRA checkpoint (``train_v1``/``train_v2``
            format: a dict whose ``"lora"`` entry maps load-order indices
            to tensors).
        gain: Global gain composed with the alpha*(t) profile; the
            shipped configuration (``corrgate025``) is 0.25.

    Returns:
        A log-line string describing the deployed configuration.
    """
    network = runner.pipeline.diffusion_model.transformer.network
    if hasattr(network, "_orig_mod"):  # unwrap torch.compile
        network = network._orig_mod
    params = _apply_lora(network)

    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)["lora"]
    assert len(sd) == len(params), (
        f"corrector checkpoint has {len(sd)} LoRA tensors but the network "
        f"exposes {len(params)}; rank or target mismatch."
    )
    for i, p in enumerate(params):
        p.data.copy_(sd[i].to(p.device, p.dtype))

    # Per-step gate: rescale the LoRA to alpha*(t) x gain before every
    # denoise step (nearest-t lookup; finalize_kv_cache calls positionally).
    transformer = runner.pipeline.diffusion_model.transformer
    orig_pf = transformer.predict_flow

    def gated_pf(*args, **kwargs):
        ts = kwargs.get("timestep", args[1] if len(args) > 1 else None)
        t = float(ts.reshape(-1).max())
        alpha = min(GATE_ALPHA.items(), key=lambda kv: abs(kv[0] - t))[1]
        _set_scale(network, alpha * gain)
        return orig_pf(*args, **kwargs)

    transformer.predict_flow = gated_pf
    return f"corrected (alpha*(t) x {gain})"

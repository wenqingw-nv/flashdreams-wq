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

"""Clean Forcing drift corrector for the LingBot-World runner.

Deploys a trained corrector LoRA (``drift_correction/train_v1.py``
checkpoints) on a built runner's pipeline at ``alpha*(t) * gain`` per
denoise step. Mirrors the HY-WorldPlay / Omnidreams deploy modules;
self-contained so the production runner does not import the research
directory.

The LoRA is **pre-merged**: at load time each distinct ``alpha*(t) * gain``
value gets its own cached copy of the target projection weights with the
scaled delta folded in, and the per-step gate — driven CPU-side from the
load-time solver schedule — swaps the set in. Unlike the HY/OD hooks the
swap is an in-place ``copy_`` into the live weight storage, not a pointer
re-point: this host's DiT runs under a CUDA graph (captured once the KV
window fills), and captured replays read the storage that was live at
capture time. Set ``DRIFT_CORRECTOR_UNFUSED=1`` for the runtime
A/B-matmul fallback (only valid while the DiT graph has not captured;
use ``use_cuda_graph=False`` for long unfused rollouts).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

## Deploy policy

GATE_ALPHA = {999.0: 0.878, 978.0: 0.582, 947.0: 0.574, 825.0: 0.637}
"""Unbiased alpha*(t) from the step-0 systematicity gate on strafe-loop
drift pairs (drift_correction's ``outputs/gate/gate_faithful.json``): the
systematic fraction of the drift-induced error at each of the distilled
solver's four warped timesteps. Deploy dial (gain, gate on/off) follows
the eval sweep; the profile itself is the measured diagnostic."""

_LORA_TARGETS = ("self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o")
"""Self-attention projections the corrector checkpoints were trained on."""

_LORA_RANK = 16
"""Rank of the corrector checkpoints."""


class _LoRALinear(nn.Module):
    """Frozen base linear plus a runtime-gated low-rank delta.

    Mirrors the training-side module in
    ``integrations/lingbot/drift_correction/_lora.py``: ``scale`` is the
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


def _nearest_alpha(t: float) -> float:
    """Return the :data:`GATE_ALPHA` entry with the nearest timestep."""
    return min(GATE_ALPHA.items(), key=lambda kv: abs(kv[0] - t))[1]


def _target_linears(network: nn.Module) -> list[nn.Linear]:
    """Target linears in checkpoint load order (same walk as ``_apply_lora``)."""
    linears: list[nn.Linear] = []
    for mname, module in network.named_modules():
        for cname, child in module.named_children():
            full = f"{mname}.{cname}" if mname else cname
            if isinstance(child, nn.Linear) and any(t in full for t in _LORA_TARGETS):
                linears.append(child)
    return linears


def _premerge_weight_sets(
    linears: list[nn.Linear], sd: dict, gain: float
) -> tuple[dict[float, list[Tensor]], int]:
    """Cache ``W + gain*alpha*(B @ A)`` per distinct gate value.

    ``sd`` holds the checkpoint tensors in load order (``A_i`` at ``2i``,
    ``B_i`` at ``2i + 1``). The merge runs in fp32 and is cast back to the
    base weight dtype.

    Returns:
        The per-alpha weight sets and the total cached bytes.
    """
    sets: dict[float, list[Tensor]] = {}
    added_bytes = 0
    for alpha in sorted(set(GATE_ALPHA.values())):
        merged: list[Tensor] = []
        for i, lin in enumerate(linears):
            a = sd[2 * i].to(lin.weight.device, torch.float32)
            b = sd[2 * i + 1].to(lin.weight.device, torch.float32)
            w32 = lin.weight.detach().to(torch.float32, copy=True)
            w = w32.addmm_(b, a, alpha=gain * alpha).to(lin.weight.dtype)
            merged.append(w)
            added_bytes += w.numel() * w.element_size()
        sets[alpha] = merged
    return sets, added_bytes


def apply_drift_corrector(
    runner: Any, checkpoint: Path, gain: float, *, unfused: bool | None = None
) -> str:
    """Deploy the corrector LoRA on ``runner`` with the alpha*(t) gate.

    Args:
        runner: A built ``LingbotWorldRunner``.
        checkpoint: Corrector LoRA checkpoint (``train_v1`` format: a dict
            whose ``"lora"`` entry maps load-order indices to tensors).
        gain: Global gain composed with the alpha*(t) profile.
        unfused: Force the runtime A/B-matmul path instead of the default
            per-step pre-merged weights. ``None`` reads the
            ``DRIFT_CORRECTOR_UNFUSED`` environment variable.

    Returns:
        A log-line string describing the deployed configuration.
    """
    if unfused is None:
        unfused = os.environ.get("DRIFT_CORRECTOR_UNFUSED", "0") == "1"
    network = runner.pipeline.diffusion_model.transformer.network
    if hasattr(network, "_orig_mod"):  # unwrap torch.compile
        network = network._orig_mod
    transformer = runner.pipeline.diffusion_model.transformer
    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)["lora"]

    if unfused:
        params = _apply_lora(network)
        assert len(sd) == len(params), (
            f"corrector checkpoint has {len(sd)} LoRA tensors but the network "
            f"exposes {len(params)}; rank or target mismatch."
        )
        for i, p in enumerate(params):
            p.data.copy_(sd[i].to(p.device, p.dtype))
        orig_pf = transformer.predict_flow

        # Per-step gate: rescale the LoRA to alpha*(t) x gain before every
        # denoise step (nearest-t lookup). Python-level scale changes are
        # invisible to a captured DiT graph — see the module docstring.
        def gated_pf(*args, **kwargs):
            ts = kwargs.get("timestep", args[1] if len(args) > 1 else None)
            t = float(ts.reshape(-1).max())
            _set_scale(network, _nearest_alpha(t) * gain)
            return orig_pf(*args, **kwargs)

        transformer.predict_flow = gated_pf
        return f"corrected (alpha*(t) x {gain}, unfused)"

    # Pre-merged path (default): one cached weight set per distinct
    # alpha*(t) value; the gate copies the set into the live weight
    # storage — no LoRA matmuls in the hot path, and in-place writes stay
    # visible to CUDA-graph replays and compiled forwards.
    linears = _target_linears(network)
    assert len(sd) == 2 * len(linears), (
        f"corrector checkpoint has {len(sd)} LoRA tensors but the network "
        f"exposes {2 * len(linears)}; rank or target mismatch."
    )
    weight_sets, added_bytes = _premerge_weight_sets(linears, sd, gain)
    current: list[float | None] = [None]

    def _swap(alpha: float) -> None:
        if alpha != current[0]:
            for lin, w in zip(linears, weight_sets[alpha]):
                lin.weight.data.copy_(w)
            current[0] = alpha

    # Drive the gate CPU-side: the FlowMatch scheduler makes exactly one
    # ``predict_flow`` call per solver step in a Python loop, so each
    # step's alpha resolves from the load-time schedule by call index —
    # a per-step GPU timestep readback would stall the launch queue.
    scheduler = runner.pipeline.diffusion_model.scheduler
    step_alphas = [_nearest_alpha(t) for t in scheduler.denoising_step_list.tolist()]
    ctx_alpha = _nearest_alpha(
        float(runner.pipeline.diffusion_model.config.context_noise)
    )
    orig_sample = scheduler.sample

    def gated_sample(initial_noise, predict_flow, rng=None):
        calls = [0]

        def pf(noisy, timestep):
            assert calls[0] < len(step_alphas), "predict_flow calls > solver steps"
            _swap(step_alphas[calls[0]])
            calls[0] += 1
            return predict_flow(noisy, timestep)

        return orig_sample(initial_noise=initial_noise, predict_flow=pf, rng=rng)

    scheduler.sample = gated_sample
    orig_finalize = transformer.finalize_kv_cache

    def gated_finalize(*args, **kwargs):
        _swap(ctx_alpha)
        return orig_finalize(*args, **kwargs)

    transformer.finalize_kv_cache = gated_finalize
    return (
        f"corrected (alpha*(t) x {gain}, pre-merged {len(weight_sets)} weight "
        f"sets, +{added_bytes / 2**20:.0f} MiB)"
    )

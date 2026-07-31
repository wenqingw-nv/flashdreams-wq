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

"""CPU tests for the Clean Forcing deploy hook (``lingbot/_drift_corrector.py``)."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from lingbot._drift_corrector import (
    _LORA_RANK,
    GATE_ALPHA,
    _apply_lora,
    _LoRALinear,
    _nearest_alpha,
    _premerge_weight_sets,
    _set_scale,
    _target_linears,
)

pytestmark = pytest.mark.ci_cpu


def _toy_network() -> nn.Module:
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.ModuleDict(
                {n: nn.Linear(4, 4, bias=False) for n in ("q", "k", "v", "o")}
            )
            self.ffn = nn.Linear(4, 4, bias=False)

    return nn.Sequential(Block(), Block())


def _random_checkpoint(linears) -> dict[int, torch.Tensor]:
    sd = {}
    for i, lin in enumerate(linears):
        sd[2 * i] = 0.1 * torch.randn(_LORA_RANK, lin.in_features)
        sd[2 * i + 1] = 0.1 * torch.randn(lin.out_features, _LORA_RANK)
    return sd


def test_lora_linear_is_identity_at_zero_scale():
    torch.manual_seed(0)
    base = nn.Linear(8, 6, bias=False)
    lora = _LoRALinear(base, rank=2)
    nn.init.normal_(lora.A.weight)
    nn.init.normal_(lora.B.weight)  # non-zero delta path
    x = torch.randn(3, 8)
    lora.scale = 0.0
    assert torch.equal(lora(x), base(x))


def test_lora_linear_is_identity_at_zero_init_b():
    torch.manual_seed(0)
    base = nn.Linear(8, 6, bias=False)
    lora = _LoRALinear(base, rank=2)
    lora.scale = 1.0  # B is zero-initialized
    x = torch.randn(3, 8)
    assert torch.equal(lora(x), base(x))


def test_apply_lora_wraps_only_attention_targets_and_set_scale_reaches_all():
    net = _toy_network()
    params = _apply_lora(net)
    wrapped = [m for m in net.modules() if isinstance(m, _LoRALinear)]
    assert len(wrapped) == 8  # 2 blocks x q/k/v/o, ffn untouched
    assert len(params) == 2 * len(wrapped)
    assert not isinstance(net[0].ffn, _LoRALinear)
    _set_scale(net, 0.25)
    assert all(m.scale == 0.25 for m in wrapped)


def test_gate_profile_nearest_t_lookup():
    # The distilled 4-step schedule resolves to its own gate entries; the
    # finalize context forward (context_noise=0) resolves to the low-t entry.
    for t in GATE_ALPHA:
        assert _nearest_alpha(t) == GATE_ALPHA[t]
    assert _nearest_alpha(0.0) == GATE_ALPHA[825.0]


def test_gate_profile_is_a_strict_attenuation():
    assert all(0.0 < a <= 1.0 for a in GATE_ALPHA.values())


## Pre-merged weight path


def test_target_linears_match_apply_lora_load_order():
    torch.manual_seed(0)
    net = _toy_network()
    linears = _target_linears(net)
    wrapped_net = copy.deepcopy(net)
    params = _apply_lora(wrapped_net)
    wrapped = [m for m in wrapped_net.modules() if isinstance(m, _LoRALinear)]
    assert len(params) == 2 * len(linears) == 2 * len(wrapped)
    for lin, w in zip(linears, wrapped):
        assert torch.equal(lin.weight, w.base.weight)  # same walk order


def test_premerged_weights_match_the_unfused_delta():
    torch.manual_seed(0)
    net = _toy_network()
    linears = _target_linears(net)
    sd = _random_checkpoint(linears)
    gain = 0.5
    sets, added_bytes = _premerge_weight_sets(linears, sd, gain)
    assert set(sets) == set(GATE_ALPHA.values())
    x = torch.randn(3, 4)
    for alpha, merged in sets.items():
        for i, lin in enumerate(linears):
            unfused = lin(x) + gain * alpha * (x @ sd[2 * i].T @ sd[2 * i + 1].T)
            premerged = torch.nn.functional.linear(x, merged[i])
            assert torch.allclose(premerged, unfused, atol=1e-5)
    n_weights = sum(lin.weight.numel() for lin in linears)
    assert added_bytes == len(sets) * n_weights * 4  # fp32 toy weights


def test_premerge_does_not_mutate_base_weights():
    torch.manual_seed(0)
    net = _toy_network()
    linears = _target_linears(net)
    before = [lin.weight.detach().clone() for lin in linears]
    _premerge_weight_sets(linears, _random_checkpoint(linears), gain=1.0)
    for lin, w in zip(linears, before):
        assert torch.equal(lin.weight, w)


def test_copy_swap_updates_storage_in_place():
    # The gate swaps sets with ``weight.data.copy_`` so a CUDA-graph replay
    # (which reads the storage captured at record time) sees the update;
    # the storage pointer must therefore stay fixed across swaps.
    torch.manual_seed(0)
    net = _toy_network()
    linears = _target_linears(net)
    sets, _ = _premerge_weight_sets(linears, _random_checkpoint(linears), gain=1.0)
    lin = linears[0]
    ptr = lin.weight.data.data_ptr()
    for alpha in sorted(sets):
        lin.weight.data.copy_(sets[alpha][0])
        assert lin.weight.data.data_ptr() == ptr
        assert torch.equal(lin.weight.data, sets[alpha][0])

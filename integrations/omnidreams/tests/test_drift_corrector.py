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

"""CPU-only unit tests for the Clean Forcing drift-corrector deploy hook.

Covers the behaviours a deployment depends on:

* ``_LoRALinear`` is a strict identity at ``scale == 0`` and at the
  zero-initialized ``B`` (so wrapping the network never changes base
  outputs until a trained checkpoint is loaded and gated on).
* ``_apply_lora`` wraps exactly the self-attention projections the
  training-side module wraps (same match rule -> same checkpoint order),
  and ``_set_scale`` reaches every wrapped linear.
* The ``alpha*(t)`` gate profile resolves by nearest-t lookup, including
  the context-noise forward.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from omnidreams._drift_corrector import (
    GATE_ALPHA,
    _apply_lora,
    _LoRALinear,
    _set_scale,
)

pytestmark = pytest.mark.ci_cpu


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
    lora = _LoRALinear(base, rank=2)  # B is zero-initialized
    lora.scale = 1.0
    x = torch.randn(3, 8)
    assert torch.allclose(lora(x), base(x))


def test_lora_linear_applies_scaled_delta():
    torch.manual_seed(0)
    base = nn.Linear(8, 6, bias=False)
    lora = _LoRALinear(base, rank=2)
    nn.init.normal_(lora.A.weight)
    nn.init.normal_(lora.B.weight)
    x = torch.randn(3, 8)
    lora.scale = 0.5
    delta = lora(x) - base(x)
    lora.scale = 1.0
    assert torch.allclose(2.0 * delta, lora(x) - base(x), atol=1e-5)


def test_apply_lora_wraps_only_attention_targets_and_set_scale_reaches_all():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.ModuleDict(
                {
                    n: nn.Linear(4, 4, bias=False)
                    for n in ("q_proj", "k_proj", "v_proj", "output_proj")
                }
            )
            self.mlp = nn.Linear(4, 4, bias=False)

    toy = Toy()
    params = _apply_lora(toy)
    wrapped = [m for m in toy.modules() if isinstance(m, _LoRALinear)]
    assert len(wrapped) == 4  # q/k/v/output projections but not the mlp
    assert len(params) == 8  # A + B per wrapped linear
    assert not isinstance(toy.mlp, _LoRALinear)
    _set_scale(toy, 0.25)
    assert all(m.scale == 0.25 for m in wrapped)


def test_gate_profile_nearest_t_lookup():
    def alpha_at(t: float) -> float:
        return min(GATE_ALPHA.items(), key=lambda kv: abs(kv[0] - t))[1]

    assert alpha_at(1000.0) == GATE_ALPHA[1000.0]
    assert alpha_at(803.0) == GATE_ALPHA[803.0]
    # The context-noise forward (t=128) resolves to the low-t entry,
    # matching the evaluated deploy configs.
    assert alpha_at(128.0) == GATE_ALPHA[803.0]


def test_gate_profile_is_a_strict_attenuation():
    assert all(0.0 < a <= 1.0 for a in GATE_ALPHA.values())

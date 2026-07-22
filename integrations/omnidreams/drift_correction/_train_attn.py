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

"""Grad-friendly functional self-attention for corrector training (Omnidreams).

The production ``SelfAttention.forward`` routes the current chunk's K / V
through the rolling ``BlockKVCache`` (in-place write into a no-grad buffer,
then a buffer read), which severs gradients into the ``k_proj`` / ``v_proj``
projections; the fused RoPE Triton kernel is likewise inference-only (raw
stores, no backward).

The functional variant below computes Q / K / V directly (differentiable
non-interleaved RoPE), reads the *history prefix* from the cache buffer
(no grad -- replayed history is frozen in v1), and runs attention over
``[prefix, current]`` without writing the cache. For a single probe forward
per ``start`` / ``finalize`` bracket this is mathematically identical to the
stock path. Deployment keeps the stock path; the weights are shared.

The patch is a *toggle*, not a wholesale swap: history-replay forwards
(``finalize_kv_cache``) must keep the stock cache-writing path, so probes
run inside ``functional_attention()`` and everything else stays stock.

Mirrors :class:`omnidreams.transformer.impl.modules.SelfAttention` and the
``rope_kernel`` semantics (cos/sin in fp32, cast to ``x.dtype`` before the
multiply); revisit on upstream changes.
"""

from __future__ import annotations

import contextlib
import math

import torch
from omnidreams.transformer.impl.modules import SelfAttention
from torch import Tensor

from flashdreams.core.attention.kvcache import BlockKVCache

_FUNCTIONAL = False
"""Process-wide toggle read by the patched forward."""

_STOCK_FORWARD = SelfAttention.forward
"""Original cache-writing forward, used whenever the toggle is off."""


def _rope_half(x: Tensor, freqs: Tensor) -> Tensor:
    """Differentiable equivalent of the fused non-interleaved RoPE kernel.

    Rotates pair ``(d, d + D/2)`` by angle ``freqs[..., d]`` (the full-width
    ``shift_t`` layout duplicates the angles across halves, so the first
    ``D/2`` lanes carry them).

    Args:
        x: Activations ``[B, S, H, D]``.
        freqs: Angles ``[S, 1, 1, D]`` from
            :meth:`RotaryPositionEmbedding3D.shift_t`.

    Returns:
        Rotated tensor, same shape/dtype, fresh storage.
    """
    half = x.shape[-1] // 2
    ang = freqs[:, 0, 0, :half].float()
    cos = ang.cos().to(x.dtype)[None, :, None, :]
    sin = ang.sin().to(x.dtype)[None, :, None, :]
    a, b = x[..., :half], x[..., half:]
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)


def _functional_forward(
    self: SelfAttention,
    x: Tensor,
    kv_cache: BlockKVCache,
    rope_freqs: Tensor | None = None,
) -> Tensor:
    """Cache-free re-implementation of ``SelfAttention.forward``.

    Reads the replayed-history prefix ``[0, write_start)`` from the cache
    buffers (frozen) and uses the freshly projected current-chunk K / V
    directly, so the visible attention span matches the stock read-back
    exactly -- in the filling phase and (post-roll) in steady state alike.
    """
    if not _FUNCTIONAL:
        return _STOCK_FORWARD(self, x, kv_cache, rope_freqs)
    batch_shape = x.shape[:-2]
    batch_size = math.prod(batch_shape)
    L, D = x.shape[-2:]
    n, d = self.n_heads, self.head_dim
    assert n * d == D, "n * d must be equal to D"

    q = self.q_norm(self.q_proj(x).reshape(batch_size, L, n, d))
    k = self.k_norm(self.k_proj(x).reshape(batch_size, L, n, d))
    v = self.v_proj(x).reshape(batch_size, L, n, d)
    if rope_freqs is not None:
        q = _rope_half(q, rope_freqs)
        k = _rope_half(k, rope_freqs)

    write_start, _ = kv_cache._current_write_bounds()
    prefix = kv_cache._seq_slice(0, write_start)
    keys = torch.cat([kv_cache._k[prefix], k], dim=-3)
    vals = torch.cat([kv_cache._v[prefix], v], dim=-3)

    out = self.attn_op(q, keys, vals)
    return self.output_proj(out.reshape(batch_shape + (L, n * d)))


def patch_functional_attention() -> None:
    """Install the toggled functional forward, process-wide (idempotent)."""
    SelfAttention.forward = _functional_forward


@contextlib.contextmanager
def functional_attention():
    """Run probe forwards through the grad-friendly, non-writing path."""
    global _FUNCTIONAL
    prev = _FUNCTIONAL
    _FUNCTIONAL = True
    try:
        yield
    finally:
        _FUNCTIONAL = prev

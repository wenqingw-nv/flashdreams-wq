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

"""Grad-friendly functional self-attention for corrector training (LingBot).

The production ``SelfAttention.forward`` routes the current chunk's K / V
through the rolling ``BlockKVCache`` (in-place write into a no-grad buffer,
then a buffer read), which severs gradients into the ``k`` / ``v``
projections; the fused interleaved-RoPE Triton kernel is likewise
inference-only (raw stores, no backward).

The functional variant below computes Q / K / V directly (differentiable
out-of-place interleaved RoPE), reads the *history prefix* from the cache
buffer (no grad -- teacher-forced history is frozen in v1), and runs
attention over ``[prefix, current]`` without writing the cache. Inside one
``start`` / ``finalize`` probe bracket this is mathematically identical to
the stock read-back -- in the filling phase and (post-roll) in steady state
alike. Deployment keeps the stock path; the weights are shared.

The patch is a *toggle*, not a wholesale swap: history rebuilds
(``rebuild_cache`` drives ``finalize_kv_cache`` through the same forward)
must keep the stock cache-writing path, so probes run inside
``functional_attention()`` and everything else stays stock. With the toggle
on, every block of a full network forward takes the functional path, which
is side-effect free and therefore compatible with per-block
``torch.utils.checkpoint(use_reentrant=False)`` on the student (grad) pass.

Mirrors :class:`flashdreams.recipes.wan.transformer.impl.modules.SelfAttention`
(standard RoPE mode, ``apply_rope_before_kvcache=True``) and the
``rope_kernel`` semantics (cos/sin in fp32, cast to ``x.dtype`` before the
multiply); revisit on upstream changes.
"""

from __future__ import annotations

import contextlib
import math

import torch
from torch import Tensor

from flashdreams.core.attention.kvcache import BlockKVCache
from flashdreams.recipes.wan.transformer.impl.modules import SelfAttention

_FUNCTIONAL = False
"""Process-wide toggle read by the patched forward."""

_STOCK_FORWARD = SelfAttention.forward
"""Original cache-writing forward, used whenever the toggle is off."""


def _rope_interleaved(x: Tensor, freqs: Tensor) -> Tensor:
    """Differentiable equivalent of the fused interleaved-RoPE Triton kernel.

    The production kernel (``rope_kernel.apply_rotary_pos_emb``) is
    inference-only: raw ``tl.store`` writes with no backward, so gradients
    through it are silently dropped. Mirrors its arithmetic exactly: pair
    ``(2k, 2k+1)`` rotated by angle ``freqs[..., k]``, cos/sin in fp32 then
    cast to ``x.dtype`` before the multiply.

    Args:
        x: Activations ``[B, S, H, D]``.
        freqs: Angles in the full-width ``[S, 1, 1, D]`` ``shift_t`` layout,
            where each pair's angle is duplicated; pair ``k`` reads lane
            ``2k`` (the kernel's ``stride_fd * 2`` skip).

    Returns:
        Rotated tensor, same shape/dtype, fresh storage.
    """
    ang = freqs[:, 0, 0, 0::2].float()
    cos = ang.cos().to(x.dtype)[None, :, None, :]
    sin = ang.sin().to(x.dtype)[None, :, None, :]
    pairs = x.unflatten(-1, (x.shape[-1] // 2, 2))
    a, b = pairs[..., 0], pairs[..., 1]
    return torch.stack((a * cos - b * sin, b * cos + a * sin), dim=-1).flatten(-2)


def _functional_forward(
    self: SelfAttention,
    x: Tensor,
    kv_cache: BlockKVCache,
    rope_freqs: Tensor | None = None,
) -> Tensor:
    """Cache-free re-implementation of ``SelfAttention.forward``.

    Reads the teacher-forced history prefix ``[0, write_start)`` from the
    cache buffers (frozen) and uses the freshly projected current-chunk
    K / V directly, so the visible attention span matches the stock
    read-back exactly -- in the filling phase and (post-roll) in steady
    state alike.
    """
    if not _FUNCTIONAL:
        return _STOCK_FORWARD(self, x, kv_cache, rope_freqs)
    assert self.apply_rope_before_kvcache, (
        "cache-relative RoPE mode would need the cached-window freqs for the "
        "prefix; the LingBot-World recipe applies RoPE before the cache."
    )
    batch_shape = x.shape[:-2]
    batch_size = math.prod(batch_shape)
    L, D = x.shape[-2:]
    n, d = self.n_heads, self.head_dim
    assert n * d == D, "n * d must be equal to D"

    q = self.norm_q(self.q(x)).reshape(batch_size, L, n, d)
    k = self.norm_k(self.k(x)).reshape(batch_size, L, n, d)
    v = self.v(x).reshape(batch_size, L, n, d)
    if rope_freqs is not None:
        q = _rope_interleaved(q, rope_freqs)
        k = _rope_interleaved(k, rope_freqs)

    write_start, _ = kv_cache._current_write_bounds()
    prefix = kv_cache._seq_slice(0, write_start)
    keys = torch.cat([kv_cache._k[prefix], k], dim=-3)
    vals = torch.cat([kv_cache._v[prefix], v], dim=-3)

    out = self.attn_op(q, keys, vals)
    return self.o(out.reshape(batch_shape + (L, n * d)))


def patch_functional_attention() -> None:
    """Install the toggled functional forward, process-wide (idempotent)."""
    SelfAttention.forward = _functional_forward  # type: ignore[method-assign]


@contextlib.contextmanager
def functional_attention():
    """Run probe forwards through the grad-friendly, non-writing path.

    Grad passes that use per-block gradient checkpointing must also run
    ``backward()`` inside this context: the checkpoint recomputation
    re-invokes each block's forward and has to retake the functional path.
    """
    global _FUNCTIONAL
    prev = _FUNCTIONAL
    _FUNCTIONAL = True
    try:
        yield
    finally:
        _FUNCTIONAL = prev

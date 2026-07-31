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

"""Rollout capture + teacher-forced history probes on the LingBot host.

The LingBot host keeps history only in the rolling ``BlockKVCache`` (no
clean-latent buffer), so the history-substitution seam works by replaying
arbitrary per-chunk latents through the KV-finalize path on a fresh cache:
``context_noise = 0`` makes that replay exact (the clean latent is fed to
the cache-update forward unchanged). A probe then runs single denoise-step
forwards at the next chunk position without ever finalizing it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from flashdreams.infra.config import derive_config
from flashdreams.infra.diffusion.model.base import DiffusionModel
from lingbot.config import PIPELINE_LINGBOT_WORLD_FAST
from lingbot.encoder.camctrl import CamCtrlInput, I2VCamCtrlInput
from lingbot.encoder.utils import get_Ks_transformed, preprocess_example_poses
from lingbot.pipeline import LingbotWorldInferencePipeline
from lingbot.runner import (
    _INTRINSICS_REFERENCE_HEIGHT,
    _INTRINSICS_REFERENCE_WIDTH,
    _load_first_frame,
    ensure_example_data_downloaded,
)

PIXEL_HEIGHT = 464
PIXEL_WIDTH = 832
"""Runner-default frame size; intrinsics are rescaled to it."""


def build_pipeline(seed: int | None = 42) -> LingbotWorldInferencePipeline:
    """Build the v1 pipeline for capture / probes.

    Compile and CUDA graphs are disabled everywhere: probes replay KV caches
    out of the normal AR order, which a captured graph (whose slot pointers
    bake in one cache object) cannot express — and eager keeps capture and
    probe numerics identical.
    """
    cfg = derive_config(
        PIPELINE_LINGBOT_WORLD_FAST,
        enable_sync_and_profile=False,
        diffusion_model=dict(
            seed=seed,
            transformer=dict(compile_network=False, use_cuda_graph=False),
        ),
    )
    pipe = cfg.setup()
    assert isinstance(pipe, LingbotWorldInferencePipeline)
    return pipe.to("cuda")


## Scenes and camera streams


@dataclass
class Scene:
    """First frame + intrinsics + prompt of one example scene."""

    first_frames_t: Tensor
    """``[1, C, H, W]`` first frame in ``[-1, 1]`` on CUDA."""

    intrinsics_row: Tensor
    """``[4]`` fx/fy/cx/cy rescaled to :data:`PIXEL_HEIGHT` x :data:`PIXEL_WIDTH`."""

    prompt: str
    """Scene prompt (empty for scenes without an upstream ``prompt.txt``)."""


def load_scene(example_idx: int) -> Scene:
    """Load one bundled example scene (downloads on first use)."""
    d = ensure_example_data_downloaded(is_rank_zero=True, example_idx=example_idx)
    first = _load_first_frame(
        d / "image.jpg",
        pixel_height=PIXEL_HEIGHT,
        pixel_width=PIXEL_WIDTH,
        device="cuda",
    )
    ks = torch.from_numpy(np.load(d / "intrinsics.npy")).to("cuda", torch.float32)
    ks = get_Ks_transformed(
        ks,
        height_org=_INTRINSICS_REFERENCE_HEIGHT,
        width_org=_INTRINSICS_REFERENCE_WIDTH,
        height_resize=PIXEL_HEIGHT,
        width_resize=PIXEL_WIDTH,
        height_final=PIXEL_HEIGHT,
        width_final=PIXEL_WIDTH,
    )
    prompt_path = d / "prompt.txt"
    prompt = prompt_path.read_text().strip() if prompt_path.exists() else ""
    return Scene(first_frames_t=first, intrinsics_row=ks[0], prompt=prompt)


def camera_stream(
    scene: Scene, poses: np.ndarray
) -> tuple[Tensor, Tensor, float]:
    """Normalize a raw ``[T, 4, 4]`` c2w pose array into runner form.

    Returns ``(poses_t, intrinsics_t, world_scale)`` with the runner's
    world-scale normalization applied (framewise, on encoded-length poses).
    """
    c2ws, trans_normalizer = preprocess_example_poses(poses.astype(np.float64))
    poses_t = torch.from_numpy(c2ws).to("cuda", torch.float32)
    intr_t = scene.intrinsics_row[None, :].expand(poses_t.shape[0], -1).contiguous()
    return poses_t, intr_t, float(trans_normalizer)


def chunk_camctrl(
    poses_t: Tensor,
    intr_t: Tensor,
    world_scale: float,
    pipe: LingbotWorldInferencePipeline,
    n_chunks: int,
) -> list[CamCtrlInput]:
    """Slice the pose stream into per-chunk camera payloads (runner logic)."""
    out: list[CamCtrlInput] = []
    start = 0
    for i in range(n_chunks):
        n = pipe.get_num_output_frames(i)
        if start + n > poses_t.shape[0]:
            break
        out.append(
            CamCtrlInput(
                intrinsics=intr_t[start : start + n],
                poses=poses_t[start : start + n],
                world_scale=world_scale,
            )
        )
        start += n
    return out


## Rollout capture


@dataclass
class ChunkSnapshot:
    """Per-chunk artifacts captured during a rollout."""

    clean_latent: Tensor
    """Unpatchified clean latent ``[C, T_lat, H_lat, W_lat]`` (cpu, fp32)."""

    ar_index: int
    """AR step this chunk was produced at."""


def capture_rollout(
    pipe: LingbotWorldInferencePipeline,
    scene: Scene,
    camctrls: list[CamCtrlInput],
    noise_seed: int | None = None,
    decode: bool = False,
) -> tuple[list[ChunkSnapshot], Tensor | None]:
    """Run a seeded rollout, capturing per-chunk clean latents.

    Args:
        pipe: Pipeline from :func:`build_pipeline`.
        scene: Scene providing the first frame and prompt.
        camctrls: Per-chunk camera payloads from :func:`chunk_camctrl`.
        noise_seed: Optional per-rollout diffusion RNG re-seed.
        decode: Also return the decoded video (``[T, C, H, W]``, cpu).

    Returns:
        ``(snapshots, video)``; ``video`` is ``None`` unless ``decode``.
    """
    device = pipe.device
    cache = pipe.initialize_cache(text=[scene.prompt], image=scene.first_frames_t)
    if noise_seed is not None:
        pipe.diffusion_model._rng = torch.Generator(device=device).manual_seed(
            noise_seed
        )
    transformer = pipe.diffusion_model.transformer
    snaps: list[ChunkSnapshot] = []
    chunks: list[Tensor] = []
    for i, ctrl in enumerate(camctrls):
        video_chunk = pipe.generate(autoregressive_index=i, cache=cache, input=ctrl)
        fs = cache.final_state
        assert fs is not None
        lat = transformer.unpatchify_and_maybe_gather_cp(fs.clean_latent)
        snaps.append(
            ChunkSnapshot(clean_latent=lat.detach().float().cpu(), ar_index=i)
        )
        pipe.finalize(autoregressive_index=i, cache=cache)
        if decode:
            chunks.append(video_chunk.detach().cpu())
    video = torch.cat(chunks, dim=0) if decode else None
    del cache
    torch.cuda.empty_cache()
    return snaps, video


## Teacher-forced history rebuild + probes


def _encode_ctrl(
    pipe: LingbotWorldInferencePipeline,
    cache,
    ar_index: int,
    ctrl: CamCtrlInput,
) -> Tensor:
    """Run the composite I2V + camera encoder for one AR step (patchified)."""
    i2v_chunk = pipe._preprocess_i2v_input(ar_index, cache.image)
    enc = pipe.encoder(
        input=I2VCamCtrlInput(i2v=i2v_chunk, camctrl=ctrl),
        autoregressive_index=ar_index,
        cache=cache.encoder_cache,
    )
    return pipe.diffusion_model.transformer.patchify_and_maybe_split_cp(enc)


def rebuild_cache(
    pipe: LingbotWorldInferencePipeline,
    scene: Scene,
    latents: list[Tensor],
    camctrls: list[CamCtrlInput],
):
    """Teacher-force a history into a fresh cache via the KV-finalize path.

    Replays chunks ``0..len(latents)-1`` in order: for chunk ``k`` the
    encoder runs normally (advancing the causal camera anchor), then the
    unpatchified latent ``latents[k]`` is written into the KV cache through
    the exact ``finalize`` path a real rollout uses (``context_noise=0`` —
    no noise, no denoising).

    Returns:
        The rebuilt pipeline cache, positioned for a probe at
        ``ar_index = len(latents)``.
    """
    dm = pipe.diffusion_model
    transformer = dm.transformer
    cache = pipe.initialize_cache(text=[scene.prompt], image=scene.first_frames_t)
    for k, (lat, ctrl) in enumerate(zip(latents, camctrls)):
        enc_p = _encode_ctrl(pipe, cache, k, ctrl)
        tc = cache.transformer_cache
        tc.start(k)
        lat_p = transformer.patchify_and_maybe_split_cp(
            lat.to(dm.device, dm.dtype)
        )
        fs = DiffusionModel.FinalState(
            clean_latent=lat_p,
            autoregressive_index=k,
            cache=tc,
            input=enc_p,
        )
        dm.finalize(final_state=fs)
    return cache


def probe_xhat0(
    pipe: LingbotWorldInferencePipeline,
    cache,
    ar_index: int,
    ctrl: CamCtrlInput,
    x0: Tensor,
    step_idx: int,
    noise: Tensor,
) -> Tensor:
    """One denoise-step forward at chunk ``ar_index`` on a rebuilt cache.

    Anchors natively: ``z_t = (1 - sigma_t) * x0 + sigma_t * noise`` with
    the solver's own warped ``(t, sigma)`` at ``step_idx``, then converts
    the predicted flow to the x0 estimate (the distilled host's target
    space): ``xhat0 = z_t - sigma_t * v``.

    The probe never finalizes ``ar_index`` — call repeatedly with different
    ``(step_idx, noise)`` on the same cache; discard the cache afterwards.

    Args:
        pipe: Pipeline from :func:`build_pipeline`.
        cache: Cache from :func:`rebuild_cache` (histories already written).
        ar_index: Probe chunk position (== number of teacher-forced chunks).
        ctrl: Camera payload of the probe chunk.
        x0: Unpatchified clean latent of the probe chunk (any device).
        step_idx: Solver step index into the warped schedule (0..3).
        noise: Unpatchified noise tensor, same shape as ``x0``.

    Returns:
        Unpatchified ``xhat0`` (cpu, fp32).
    """
    dm = pipe.diffusion_model
    transformer = dm.transformer
    scheduler = dm.scheduler
    sigma = scheduler.denoising_sigmas[step_idx]
    t = scheduler.denoising_step_list[step_idx].to(dtype=dm.dtype)

    if not hasattr(cache, "_probe_enc"):
        cache._probe_enc = {}
    if ar_index not in cache._probe_enc:
        cache._probe_enc[ar_index] = _encode_ctrl(pipe, cache, ar_index, ctrl)
        cache.transformer_cache.start(ar_index)
    enc_p = cache._probe_enc[ar_index]

    x0 = x0.to(dm.device, torch.float32)
    noise = noise.to(dm.device, torch.float32)
    z_t = ((1.0 - sigma) * x0 + sigma * noise).to(dm.dtype)
    z_p = transformer.patchify_and_maybe_split_cp(z_t)
    flow = transformer.predict_flow(
        noisy_latent=z_p,
        timestep=t,
        cache=cache.transformer_cache,
        input=enc_p,
    )
    flow_u = transformer.unpatchify_and_maybe_gather_cp(flow)
    xhat0 = z_t.float() - float(sigma) * flow_u.float()
    return xhat0.detach().cpu()

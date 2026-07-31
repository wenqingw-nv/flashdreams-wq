<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LingBot-World port — Clean Forcing (board)

*Started 2026-07-31 (owner /goal directive). Third FlashDreams host. Playbook:
`integrations/hy_worldplay/drift_correction/LIBRARY_DESIGN.md`; process rules per HY
`drift_correction/HANDOFF.md` (gate first, owner eyeball final, boards stay concise).*

## Host map (survey 2026-07-31)

- 14B Wan2.1-family DiT (both v1 `lingbot-world-fast` and v2 slugs), `dim=5120`, 40 layers;
  LoRA targets identical to HY: `blocks.{i}.self_attn.{q,k,v,o}`. `compile_network=True`
  (unwrap `_orig_mod`), `guidance_scale=1.0` (no uncond branch).
- FlowMatch scheduler, 4 steps, `denoising_timesteps=[1000, 821, 642, 321]` warped at shift=10
  -> solver visits ~[1000, 979, 947, 825]. `context_noise=0` (finalize feeds clean latents).
- Chunks: `len_t=3` latent frames (9/12 decoded frames), 464x832 px = 58x104 latent,
  rolling `BlockKVCache` window 63 latent frames (21 chunks) + sink 0; NO clean-latent history
  buffer -> history substitution must rebuild a fresh KV cache by teacher-forcing latents
  through the finalize path (`context_noise=0` makes this exact).
- Camera: per-frame Plücker rays from frame-relative SE(3) (`encoder/camctrl.py`), FiLM after
  self-attn per block. No pose-string grammar; poses come as `[T,4,4]` c2w + `[T,4]` intrinsics
  (`.npy`) — trajectory generator can reuse `flashdreams.serving.webrtc.controls.CameraPoseIntegrator`.
- Clean-latent capture seam: between `pipeline.generate` and `pipeline.finalize`,
  `cache.final_state.clean_latent` (patchified; unpatchify via transformer helper).
- **Deploy constraint (new vs HY/OD): DiT CUDA graph** (`use_cuda_graph=True`, capture at
  AR = (sink+window)/len_t = 21 for base slug, 6 for taehv slug). Captured replay holds weight
  pointers from capture time -> per-step weight-set *re-pointing* goes inert after capture.
  Deploy options (decide at stage 5): in-place `copy_` swap into live storage (~8.4 GB/set on
  14B ≈ ~1 ms on GB300 HBM), or `use_cuda_graph=False` when corrector on, or per-alpha graphs.
  (HY sets `use_cuda_graph=False` -> shipped HY gate unaffected; OD path unaffected.)

## Stages + pre-stated kill bars (set BEFORE data, per playbook)

1. **Smoke run** (v1, example data, 21 blocks) — proves env + downloads. IN PROGRESS.
2. **Adapter**: `_rollout.py` (build_runner, capture_rollout w/ per-chunk clean latents + poses),
   `_traj.py` (mixed-geometry loop trajectories from example scenes), `_lora.py` (HY mirror),
   probe = fresh-cache teacher-forced history rebuild + one denoise-step forward.
3. **Step-0 gate** (`gate_faithful.py` port): loop pairs, lap-1 teacher, unbiased alpha*(t) at the
   4 solver timesteps. KILL BARS: mean rel drift gap < 0.01 -> nothing to correct, STOP;
   alpha* < 0.7 at the drift-dominant timesteps -> REPORT to owner before any training
   (HY precedent: 0.81@t1000 / ~0.53-0.58 below, proceeded with gate as deploy knob).
4. **Pairs + v1 training** (gate-conditional): mixed-geometry legs (HY repeat-prior lesson:
   never pure loops), zero real videos; LoRA r16 q/k/v/o; kill bar: val R^2 plateau < 0.15 ->
   stop and report. 200-step checkpoints, resumable, via safe_run.sh.
5. **v2 (DAgger; contraction OFF by default — cross-host evidence: progression killer),
   dial sweep gain x alpha*(t)-gate, owner-eyeball sbs, deploy hook** (pre-merged weight sets +
   CPU-side schedule gate from day one, graph-compatible swap per the constraint above),
   ci_cpu tests, README refresh recipe.

## Open

- VRAM budget: 21-chunk window KV ≈ 78 GB (cond-only) + 28 GB weights on 14B — single job at a
  time; probes rebuild ONE cache per history. Consider window15/sink3 preset for training probes
  if full-window probes don't fit alongside owner jobs.
- Which slug to ship on: v1 base slug first (docs' primary example); v2 = same pipeline, refresh
  recipe applies.

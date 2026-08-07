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
3. **Step-0 gate — RUN 2026-07-31 (`outputs/gate/gate_faithful.json`, 18 cells, 6 scenes x
   laps 3-6): AT THE REPORT-FIRST BAR, awaiting owner GO/NO-GO.**
   Unbiased alpha*(t): **0.878 @ t=999**, 0.582 @ 978, 0.574 @ 947, 0.637 @ 825; 36% of cells
   >= 0.7. Mean rel drift gap **0.462** (46x the nothing-to-correct bar; grows monotonically
   with lap depth in every clip — drift is real and large on this host). Profile is HY-shaped
   but slightly stronger at every timestep (HY: 0.81 / 0.53-0.58, shipped at -48% drift with
   the alpha*(t) gate as deploy knob). Caveats as on HY: alpha* is a lower bound (corrector
   sees z_t), and the t=999 step dominates few-step hosts.
   Original kill bars: mean rel < 0.01 -> STOP (not triggered); alpha* < 0.7 majority ->
   REPORT first (triggered -> this entry).
4. **Pairs + v1 training — DONE 2026-08-02** (owner GO on the gate 2026-07-31): 800 steps,
   final val R^2 +0.419 (kill bar 0.15 cleared; plateau band 0.40-0.47 from ~step 200).
   Checkpoint `outputs/lora_v1.pt`. **v1 closed-loop eval (18 held-out cells/config,
   40 chunks, `outputs/eval_v1/scores.json`)**: Δ-drift base +8.27 -> corr(g1.0) +5.30
   (-36%) / corrgate(x1.0) +4.63 (**-44%**, target >= 30%); MUSIQ 58.8 -> 61.8/62.6 (late
   54.5 -> 57.2/59.1); latesim/lag2s up. **Guard breach: dynamics -31%** (65.6 -> ~45.5,
   guard ~20%) — the known drift-progression trade at full gain (HY shipped 0.5, OD 0.25);
   dial sweep at gains 0.5/0.7 (flat + gated) running. Watch items for eyeball: seam motion
   ratio 1.37 @ corr g1.0 (base 1.00; corrgate 1.19), cuts 1.1 vs 0.6.
5. **Dial sweep — DONE 2026-08-02** (`outputs/eval_v1/scores.json`, 7 configs x 18 cells):
   guard-compliant AND >= 30% Δ-cut: **corr050 (flat 0.5): Δ -44%, dyn -14%, best late MUSIQ
   60.15, seam 1.07** · corrgate070: Δ -42%, dyn -17%. Full gain (either mode) breaches the
   dynamics guard (-29..31%); corrgate050 undershoots drift (-26%). Flat 0.5 beating the gated
   dials means deploy pre-merge = ONE weight set, zero runtime swaps (CUDA-graph constraint
   moot for the recommended config). sbs eyeball material: `outputs/eval_v1/sbs/` (8 x
   base-vs-corr050 + corr050-vs-corrgate070). **AWAITING OWNER EYEBALL + ship decision
   (config / v2 round).** Deploy hook + tests already landed (`lingbot/_drift_corrector.py`);
   remaining after verdict: set shipped defaults, deploy-hook GPU verify, README refresh
   recipe, boards close-out.

## Open

- VRAM budget: 21-chunk window KV ≈ 78 GB (cond-only) + 28 GB weights on 14B — single job at a
  time; probes rebuild ONE cache per history. Consider window15/sink3 preset for training probes
  if full-window probes don't fit alongside owner jobs.
- Which slug to ship on: v1 base slug first (docs' primary example); v2 = same pipeline, refresh
  recipe applies.

## SHIP DECISION — owner eyeball 2026-08-07: DO-NOT-SHIP per-step corrector on this host

90 s (120-chunk) sbs verdict: base does not visibly collapse (63-frame rolling window makes this
host drift-resistant at practical horizons); corr050 preserves color/saturation but LOSES texture
detail vs base at later frames — the cost exceeds the benefit the instruments credited (Δ-MUSIQ
was tracking the saturation axis). Owner eyeball outranks instruments (7th overrule on record
across hosts). The corrector, checkpoints, and recipe stay in-tree for refresh if a future
checkpoint/base drifts harder; LingBot product value from this effort = the 1-NFE speed result
(2.2x chunk speedup at quality parity, outputs/eval_nfe1/), decision on that thread pending.

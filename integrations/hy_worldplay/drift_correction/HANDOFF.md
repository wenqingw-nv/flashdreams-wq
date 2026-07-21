<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# HANDOFF — port the paper-final Counterfactual Forcing to HY-WorldPlay

*2026-07-21. For a fresh agent taking over this integration. Owner: wenqingw. Read this fully
before running anything.*

## 1. Goal

Refresh this port to the paper-final method and this window's findings: (a) confirm/upgrade the
corrector recipe to the locked paper version, (b) apply the deployment learnings (gain dial,
α*(t) gate), (c) pursue the **HY-specific progression fix** — the commanded-future
motion-conditioned teacher — which supersedes the on-hold v4 design. The paper
(`~/projs/drift_correction/iclr2026`) is FROZEN and firewalled from this work: nothing from HY
goes into it, and nothing here may touch it.

## 2. What already exists here (do not rebuild)

- `REPORT.md` — port results: Δ-drift −50–58% at full gain; **negative drift** with the α*(t)
  gate (+4.4 MUSIQ over base); base's saturation-runaway eliminated. `OWNER_FEEDBACK.md` =
  owner decisions; `TODO.md` = open items.
- Trained LoRAs: `outputs/lora_{v1,v2,v3}.pt` (v2 = deployed candidate; v3 = round-2 DAgger,
  eval pending). Pairs: `outputs/pairs{,_dagger1,_dagger2}/clip_*.pt` (strafe-loop, zero real
  videos). Gates: `outputs/gate/gate_faithful.json` (α*(t): 0.81@t=1000, ~0.53–0.58 below — the
  t-dependence is why the gate is a deploy knob here).
- Scripts: `build_pairs.py`, `train_v1.py`, `train_v2.py`, `eval_rollouts.py`, `score_drift.py`,
  `gate_faithful.py`, `gate_systematicity.py`, helpers `_rollout.py`, `_pairs.py`, `_lora.py`,
  `_train_attn.py`. NEW and untested on GPU: `gate_oracle.py` (pose-memory oracle gate — see §5).
- Branch `hy-worldplay-counterfactual-forcing`, remote `wenqingw-nv/flashdreams-wq`.
- Env: run from the flashdreams repo root with `.venv/bin/python`. GPU jobs go through
  `~/projs/drift_correction/scripts/safe_run.sh` (2-slot lock in `/tmp/gpu_slots`, driver-wedge
  protection — this GB300 box has a known UVM driver bug; keep jobs short and resumable).

## 3. The paper-final method (what "latest version" means)

Recipe (proven on Wan2.1-1.3B; this port's v2 already approximates it in x̂0 space):
1. **Counterfactual teacher**: target = frozen model's prediction at the SAME noisy state with
   the clean-counterpart history; loss normalized per-sample by the drift gap
   ‖target − base‖². Native anchoring (z_t from the model's own rollout) — GT-anchoring
   provably injects drift.
2. **v2 closed loop**: DAgger pool aggregation (round-0 + corrected-rollout round-1) +
   drift-contraction term (commit corrected x̂0 into the next block's history WITH grad,
   penalize that block's gap; weight 0.5).
3. **Deploy merged** (zero inference cost) with a gain dial; per-timestep α*(t) gating where the
   gate profile is non-flat (it is, on this host).

Deployment guidance (measured): **gain 0.7** = guard-compliant (Δ −26%, dynamics within 20% of
base) · **α*(t)-gated** = drift-critical · full gain only where progression doesn't matter.
Static-background use cases align WITH the anchoring (background stability is the point);
validate one case before shipping: a commanded entrance of a new story element at gain 0.7 vs
1.0 vs off (the pull resists novel content).

## 4. Findings from the paper host you must respect (2026-07-19..21)

- **Dichotomy (measured, closes the naive future-teacher family):** a teacher conditioned on the
  TRUE future makes the deterministic student regress to a conditional mean → visible blur
  (owner eyeball overruled off-frontier instrument numbers, twice). A teacher conditioned on a
  PREDICTED future re-anchors (the model's own future is conservative). The progression signal
  lives in the *unpredictable* share of the future. **Do not port GT-future or predicted-future
  teachers as-is.**
- **The HY escape clause:** on HY-WorldPlay the future *trajectory* is a command input
  (viewmats/actions) — not unpredictable. A teacher conditioned on the commanded future poses
  rotates the anchoring force into trajectory-following, leaving only content-along-the-path
  uncertain. This is the design to pursue (§5), ranked ABOVE the on-hold v4 (mixed-geometry +
  trust region).
- **Owner's eyeball outranks instruments.** Every quantitative win this window that failed the
  owner's eye died. Always produce side-by-side mp4s (`ffmpeg hstack`, name which side is which
  in the filename) and wait for the verdict before building on a result.
- **Gate first, compose later.** Every arm gets a cheap go/no-go measurement before any training
  (this window: 4 of 6 arms died at their gates for hours instead of days). Kill bars are set
  BEFORE the data comes in.
- Seam artifacts: HY's production-trained base has NO wash pulse (that's a lightly-adapted-host
  artifact). HY's boundary artifact is a content-level jump-cut (context/memory handoff) —
  different mechanism; overlap/blend at the content level is the relevant fix class here.

## 5. Suggested next arms, in order

1. **Re-eval v3 (round-2 DAgger)** — trained, eval pending (was on hold). Standard eval +
   owner eyeball. ~0.5 d.
2. **Gate the pose-memory oracle** — `gate_oracle.py` (written, never run): extends the
   teacher's memory with clean lap-1 frames at the poses the camera is ABOUT to visit (pure
   index-set change, fully in-distribution; no bidi pass needed). Reports α*_clean vs α*_oracle,
   info content, and a new-pose-leak diagnostic (if loop pairs already FOV-cover future poses,
   info ≈ 0 → needs non-loop pairs). ~2–4 h GPU. Kill bar: info < 0.05 → stop.
3. **Commanded-future teacher (the big one)**: teacher conditioned on future viewmats/actions +
   pose-matched clean memory. Design sketch: build pairs on FORWARD trajectories (not loops)
   where the "future" context = commanded poses + (if available) clean content at those poses
   from a slow/reference traversal; gate it exactly like #2 before training. This is the
   dichotomy-escape experiment — if α* holds and info is real, train v2-recipe with the swapped
   teacher and judge progression on the bridge trajectory (owner's canonical eyeball scene:
   `outputs/eval_v2bridge/`, the camera should REACH the bridge).
4. **Protocol metrics port**: one-code-path anchoring / lag-2s identity / cut-rate / RAFT
   dynamics (see `~/projs/drift_correction/scripts/posthoc_metrics.py` for the reference
   implementation; add RAFT since HY has real motion commands).

## 6. Pitfalls that cost us time (learn from our scars)

- `safe_run.sh` env: caller env vars pass through, but never name your own vars generically
  (an internal `N` once clobbered a caller's `N=16` → ZeroDivision).
- Never `pkill -f` with a pattern that appears in your own command line (it self-kills the
  chain; happened twice). Prefer PID-waits: `while kill -0 <pid>; do sleep 30; done`.
- Wan VAE encode wants `[B, C, T, H, W]`; imageio gives `[T, H, W, C]` — permute carefully.
- The causal models cache attention masks per shape (`self.block_mask is None` pattern) — keep
  window shapes constant per process, or swap masks explicitly.
- Checkpoint every ≤200 steps with RESUME support; the box wedges (driver bug, owner has a
  ticket). Queue long chains via `queue_runner.sh` + `@reboot` crontab if the box is flaky.
- HY specifics: `HyWorldPlayCtrl.memory_frame_indices` slices `clean_latent_history` by
  absolute frame index (bounds-checked, no budget cap); `make_ctrl` requires k ≥ 1; BlockKVCache
  enforces strict start/finalize alternation (`start_probe_chunk`/`finish_probe_chunk`).
- Don't trust chunk-lag luminance autocorr (pulse-ac) across configs — it also fires on smooth
  luminance drift (SF measures 0.90 with visually pulse-free videos). Use it only within-config,
  same content.

## 7. Decision log pointers

- Method + all paper-host evidence: `~/projs/drift_correction/issues.md` (post-mortems at the
  bottom), `~/projs/drift_correction/flashdreams_value.md` (the pitch, incl. deployment dial),
  memory notes in the owner's session (v3b dichotomy, nfb=7 host finding).
- This port's history: `REPORT.md`, `OWNER_FEEDBACK.md`, `TODO.md`, git log of this directory.

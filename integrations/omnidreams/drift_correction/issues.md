<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OmniDreams port — quality issues board

**SHIPPED (owner decision 2026-07-24): `outputs/lora_v2_v3_valpeak.pt` @ α*(t)×0.25 (`corrgate025`)
— the OmniDreams deployment.** md5 `84beb8c014346833ce182310373b5d32`; runner flag
`OmnidreamsRunnerConfig.drift_corrector` (+ `drift_corrector_gain`, default 0.25); quick start in
`README.md`. Owner verdict: best of all arms — better trees/foliage detail and consistency, less
repeated street view; Δ +0.99 vs base +2.44 (60% cut), MUSIQ 50.1 > base, dyn −17%.
Rules: gate first with pre-set kill bars · owner eyeball outranks instruments · failed arms live in
git history, not on this board.

| # | issue | status | pointers |
|---|---|---|---|
| 1 | **scen6 partial repetition (residual)** — some repeated rows of buildings/shops + 3 repeated white trucks remain in the shipped config (the full v2 repeat prior — crosswalk suppression + trailer walls — was fixed by pairs-v3 mixed lap lengths; the residual is the loop prior itself). Fix attempts on record: pairs-v3 (period removed, prior remains), pairs-v4 re-anchored fork pairs + UW (gate PASSED α*_ub 0.892 / rel_v 0.28 — the loop-free counterpart design is valid — but the sweep FAILED owner eyeball: corrgate025 blurrier than base, corrgate worse with trees/buildings merged "like cubes"; **v4 round CLOSED — FAILED, 6th instrument-vs-eyeball overrule**) | OPEN — owner-accepted at ship; next attempt needs a fresh design + gate | evidence + rechecks under `outputs/eval_sweep_v2f/repeat_prior_evidence/`, `outputs/eval_sweep_v3/repeat_prior_recheck/`, `outputs/eval_sweep_v4p/v4_recheck/`; failed arms (LP, v4, ×0.5 dials) in git history |
| 2 | **Softness vs base (residual)** — shipped config sharpness_ratio 0.598 vs base 0.869; owner-accepted (trees/foliage detail judged BETTER by eye). **Instrument disqualification on record: sharpness_ratio has mispredicted the owner verdict in BOTH directions on this host** (v3-vp 0.598 = owner-acceptable detail; v4-vp 0.94 > base = owner-blurry) — it is disqualified as a kill-bar instrument for OmniDreams; future rounds use owner frame-stacks for the sharpness axis | OPEN — accepted | `outputs/eval_g025_v3_valpeak/g025_recheck/` stacks |
| 3 | **Dynamics −17% vs base** in the shipped config (13.8 vs 16.6 RAFT) | OPEN — accepted (within the HY-port guard band) | `eval_g025_v3_valpeak/scores.json` |

## Standing findings

- **Gain-prediction analysis** (gain*(t) = α*(t) × ρ(t); `gain_predict.py` + `TBIN_EVAL` in the
  trainers): verdict **MIXED** — HOLDS on HY (ranked all arms exactly as the owner verdicts), FAILS on
  the OD eyeball ordering; prospective v4 test = top-1 HIT / near-tie MISS (prediction recorded
  pre-sweep 2026-07-24, 11:20 UTC).
- **Fork-pairs gate finding stands**: a re-anchored fork counterpart (no content revisits) carries a
  large, ~89%-systematic drift signal (`outputs/gate/gate_faithful_v4.json`) — a valid pair design for
  future rounds even though the v4 sweep failed the eyeball.
- Ablation flags kept in the code: `PAIR_SCHEME=fork`, `UW=1`, `SNAP_EVERY`, `TBIN_EVAL`
  (trainers); `LP_SIGMA`, `CONFIGS`, `ALPHA_STAR` (eval_rollouts).
- Checkpoint staging: ship LoRA at
  `integrations/omnidreams/drift_correction/outputs/lora_v2_v3_valpeak.pt` on this box (outputs are
  gitignored); hand the file + md5 above to whatever hosting the team prefers.

Full arm-by-arm history (v1/v2/v3/v4 rounds, LP arm, dial sweeps, verifications, owner verdicts):
git log of this directory, 2026-07-22..24.

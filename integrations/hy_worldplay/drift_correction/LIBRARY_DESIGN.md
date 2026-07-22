<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Design notes — Clean Forcing as a FlashDreams tool + the generic-corrector question

*2026-07-22, owner-approved direction: retrain the LoRA per base model for now; these notes record
the library shape and the pretrain→post-train idea for when they're wanted.*

## 0. The library at a glance

```
                 ┌──────────── FlashDreams host ────────────┐
                 │ adapter (per model): rollout·probe·lora  │
                 └────────────────────┬─────────────────────┘
                ┌─── host-agnostic core (one code path) ────┐
                                      ▼
  [1] GATE   α*(t) diagnostic ── no gap / low α* ──────► STOP
                 │ go                                (kill bars pre-stated)
                 ▼
  [2] PAIRS  clean ↔ drifted histories at matched z_t
                 ▼
  [3] TRAIN  r_φ LoRA on FROZEN host   (v1 → v2: +DAgger +contraction)
                 ▼
  [4] SCORE  drift/dyn/seam + owner eyeball · dial sweep g × α*(t)-gate
                 ▼
  [5] DEPLOY runner hook:   motion → r_φ @ dial   │   static → OFF
             (per-step gate ⇒ unfused ~0.3% · else merged = zero cost)
```

## 1. Library: host adapter + host-agnostic core

Everything in this port generalizes except three functions. Target shape (first consumer:
omnidreams port; HY becomes adapter #1):

**Host adapter (per model, ~1–2 days):**
- `capture_rollout(runner) -> [ChunkSnapshot]` — per-chunk history, produced clean latent, ctrl.
- `predict(z_t, t, history, ctrl) -> x̂0 (or v)` — probe forward with substituted history
  (teacher/student/base share it; target space follows the host's solver: x̂0 for distilled,
  velocity for multi-step).
- LoRA attach points (which modules; r16 q/k/v/o is the proven default).

**Host-agnostic core (≈90% already written in this port):**
- `gate` — α*(t) step-0 diagnostic (`gate_faithful.py`): go/no-go BEFORE training + the deploy
  gate profile. Kill bar: mean rel gap < 0.01 → nothing to correct; α* below bar → report first.
- `pairs` — clean/drifted pair construction (`build_pairs.py`: strafe-loop counterfactuals for
  pose-conditioned hosts; generic prefix-rollout pairing otherwise).
- `train` — v1 counterfactual loss (gap-normalized) + v2 DAgger/contraction (`train_v{1,2}.py`).
- `score` — drift/dynamics/seam/latesim suite (`score_drift.py`) + official VBench hook.
- `deploy` — runner hook (`_drift_corrector.py`): gain, α*(t) per-step gate, content-keyed
  selection (static → off), unfused-LoRA path when the gate is per-step.

**Playbook (the part that transferred perfectly twice):** measure α*(t) (~1 GPU-day) → pairs →
v1 → DAgger+v2 (~2 GPU-days) → dial sweep {gain, gate, gate×gain} → owner eyeball → content-keyed
ship. Kill bars set before each stage.

## 2. Generic corrector (pretrain → per-base post-train) — recorded, not scheduled

Question: can one LoRA learn shared "drift physics" and be post-trained cheaply per base?

- Measured decomposition (paper host): transfer to a sibling checkpoint retains quality but only
  ~25% of drift calibration (zero-shot open-loop R² +0.03 vs +0.39 in-place) → the corrector =
  shared component + dominant checkpoint-specific residual.
- Recipe if pursued: **Stage A** train one LoRA on pairs pooled across checkpoints (the
  gap-normalized loss makes hosts comparable; one-line trainer change). **Stage B** warm-start
  per checkpoint and post-train briefly (warm-starting is proven: v2 inits from v1). Value case:
  continuous deployment — cut per-checkpoint re-instantiation from ~1 GPU-day to ~1–2 h.
- **Gate first (cheap, ~2–3 GPU-h):** α*_shared = the α* estimator with checkpoints in place of
  noise seeds, on the existing 4K/8K/14K/20K adaptation checkpoints. α*_shared ≳ 0.5 → Stage A/B
  worth building; ≈ 0.1 → only the *recipe* is generic, keep retraining per model.
- Hard limits: across architectures the LoRA tensors don't fit — weight-level genericity is
  hypernetwork-grade research; within-family only. Stage B is never zero.

**Current policy (owner, 2026-07-22): retrain the LoRA per base model. §1 is the reuse story;
§2 waits.**

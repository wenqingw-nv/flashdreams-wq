# Owner feedback — camera-following diagnosis + v4 go-ahead (2026-07-15)

Diagnosis agreed. Same root cause we hit on the T2V host — the corrector inherits the biases of its
training-pair distribution — your strafe-loop version is the structured extreme of it.

**New cross-host evidence that sharpens your v4 design (measured on our Wan2.1 finals an hour ago):**
late-frame similarity-to-start on identical references — v1 corrector (no DAgger/contraction) 0.20 ≈
Self-Forcing baseline 0.17 (healthy progression), v2 (+DAgger+contraction) 0.50 (sticky). **The contraction
term is the primary progression-killer even with non-loop pairs.** Your v3/v3b contraction work and your
fix-2 (mixed-geometry pairs) are therefore both load-bearing — treat contraction weight as the explicit
progression↔drift knob, not a fixed constant.

## Instructions

1. **Before any retrain (free):** I'll eyeball your two gate side-by-sides. You add a **quantitative
   trajectory metric** to the grid — you have camera poses, so report pose-following error (or
   frame-similarity-to-start curves) for base/corr/gated. No more eyeball-only motion claims; and re-report
   the revisit-memory number under that metric with the inflation you flagged acknowledged in the writeup.
2. **Go on v4** with your fixes 1+2 together: mixed-geometry pairs are the cure, the trust-region term is
   the seatbelt — λ swept over {0.1, 0.3, 1.0}, normalized, warm-up scheduled (raw-magnitude auxiliary
   terms diverge; that was our week-1 lesson). Keep contraction OFF or ≤0.1 in v4 given the cross-host
   evidence above. Add the q/k-only LoRA as a cheap third arm if the grid has room, but don't bet on it
   alone (attention-only restrictions cost correction power in our ablations).
3. **v4 is the final iteration** — we're at day 2 of the 3-day timebox. If v4 doesn't restore
   camera-following at ≥baseline quality under the new metric, stop and deliver the deploy-config
   recommendation from the morning grid plus the failure analysis. That's a valid deliverable; a clean
   negative with mechanism beats a tuned maybe.

Report results here or in your scores.json convention as before. — owner (wenqingw)

---

## Response — agent (2026-07-15)

**Instruction 1 (trajectory metric, before retrain) — done.** Metric: ``advance_score`` = 1 − mean
late-half frame-correlation to frame 0 (in ``score_drift.py``, all future grids carry it; per-video values
in every ``scores.json``). Measured on the bridge forward-trajectory cell (24 chunks):

| config | corr-to-start @25/50/75/end | advance_score |
|---|---|---|
| base | 0.60 → 0.48 → 0.41 → 0.37 | **0.58** |
| v2 @1.0 | 0.61 → 0.56 → 0.61 → 0.62 | 0.38 |
| v2 @0.85 | plateau ≈0.6 | 0.38 |
| v2 gated | plateau ≈0.6 | 0.38 |

Gate/gain do not move it — the repeat prior is in the weights, consistent with your cross-host numbers
(your v2 0.50 vs v1 0.20). **Revisit-memory retraction:** the earlier "2.4× better revisit consistency"
(lap-MSE, corrected rollouts) is inflated by the same repeat prior and is withdrawn; it will only be
re-reported for a checkpoint whose advance_score is at parity with base.

**Instruction 2 (v4) — armed with your amendment.** Mixed-geometry pairs (legs 3/4/5 → laps 12/16/20,
150 clips, prompt-synthesized seeds) + trust-region λ ∈ {0.3, 0.1, 1.0} (drift-gap-normalized,
100-step linear warmup), **contraction OFF (CW=0)** per your cross-host evidence, t-weighted sampling
(0.4 at t=1000 per the non-flat α*(t): 0.81 vs ~0.55), init from v2, 300 steps/arm. Tonight's v3
(CW 0.25) vs v3b (CW 0) on loop data will give this host's own contraction-stickiness ablation as a
bonus row. q/k-only arm queued only if the grid finishes with margin, per your caveat.

**Instruction 3 — acknowledged.** v4 is final; pass bar = advance_score ≈ base at ≥ baseline quality
(plus the standing Δ/dyn/sat/sharp guards). Otherwise: deploy-config recommendation from the grid +
failure analysis (loop-teacher bias + contraction stickiness as mechanisms) as the deliverable.

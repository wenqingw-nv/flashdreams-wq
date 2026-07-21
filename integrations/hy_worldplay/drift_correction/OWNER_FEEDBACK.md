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

---

## Owner addendum (2026-07-16) — machine limits after the CPU lockup

Admin root-caused last night's freeze: CPU soft lockups from concurrent-job overload (no OOM). Standing
rules for this box, effective now: **total ≤3 concurrent heavy jobs across ALL agents** (GPU generation,
training, and CPU-heavy video-decode scoring all count), and prefix every job with
`OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8`. Two slots are occupied by my sfc resume +
T11 retrain until ~tomorrow morning — schedule your v4 chain as the third slot, one arm at a time, and
stagger any decode-heavy scoring behind the training arms.

## Owner (2026-07-16) — v4 ON HOLD

Priority call: the progression-bias fix runs on the paper host (Wan2.1-1.3B) FIRST — its diagnostics and
contraction/reference conclusions will de-risk your v4 design anyway. Do not launch v4 or other GPU/decode
jobs until I post the go-ahead here. Use the time for the writeup: metric definitions, the v1-blur→v2 story,
the retraction note, and the deploy-config recommendation draft from the morning grid.

## Owner (2026-07-17) — mandatory launcher after 2nd incident

A second freeze occurred (UVM driver wedge, single job running — not overload). Box was rebooted.
**Effective now: launch ALL GPU/decode jobs through `~/projs/drift_correction/scripts/safe_run.sh <cmd>`**
— it enforces a driver health probe, a 2-slot box-wide concurrency lock (/tmp/gpu_slots), thread caps,
CPU quota, and the non-VMM torch allocator (`backend:native` — do NOT set expandable_segments). This
applies to your v4 chain when it gets the go-ahead.

## Owner (2026-07-17) — MACHINE IN MAINTENANCE MODE

Driver bug report filed; box is in maintenance (may be power-cycled/re-imaged anytime). **Launch NOTHING
until further notice.** All v4 work stays writing-only. Will post here when hardware returns.

## Owner eyeball (2026-07-21) — gain-sweep first verdicts

On the fresh sweep rollouts (`outputs/eval_sweep`): **corr050 reaches the bridge** (progression
preserved) **with better color saturation than base**; small pulses remain but without much image-quality
degradation. The **gated config anchors — does not reach the bridge** (consistent with its latesim 0.86 vs
base 0.67). Corrector-induced chunk-cadence pulse diagnosed and measured (corr seam-motion ratio
0.92–0.96 vs base ~1.0; mechanism = per-chunk anchoring kick + flat-gain over-correction at low-α*
timesteps + contraction term). Direction: low gain (~0.5) is the progression-preserving deploy candidate;
score the full grid with the seam-aligned motion ratio before locking.

## Owner (2026-07-21) — maintenance hold LIFTED; HANDOFF scope active

Box cleared for GPU jobs again (owner confirmation, in-session). All jobs still go through
`~/projs/drift_correction/scripts/safe_run.sh`. Active scope = `HANDOFF.md` acceptance criteria
(drift reproduction, progression gain sweep, static-background demo) + §5 unfinished business
(v3 re-eval, protocol metrics port). v4 / research arms remain on hold.

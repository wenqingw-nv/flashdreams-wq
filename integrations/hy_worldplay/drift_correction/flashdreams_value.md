# Why Clean Forcing is useful for FlashDreams

*2026-07-21 — companion to `flashdreams/integrations/hy_worldplay/drift_correction/{REPORT,issues,TODO}.md`.*

**Measured on the HY-WorldPlay port:** Δ-drift −50–58% at full gain; **negative drift** with the
α*(t) gate (+4.4 MUSIQ over base); the base's saturation/texture runaway eliminated, structures
intact. **SHIPPED 2026-07-21 (content-keyed rule, owner-verified both ways):** commanded-motion
jobs → corrgate050 (α*(t)×0.5 per solver step; bridge: latesim 0.669, dyn ≈ base, Δ −48%, owner
eyeball PASS); static jobs → corrector OFF (static scenes measure *negative* drift — nothing to
correct; owner ordering: base > corrgate050 > corrgate > corr on quality and pulse). Selector is
free: trajectory type is a runtime input. Note: the gate needs a non-flat α*(t) (HY 0.81→0.53);
on flat-α* hosts (our paper host: 0.91–0.99 everywhere) it degenerates to plain gain — measure
α*(t) first when porting the dial. Per-step gating can't merge into one weight set: motion jobs
run unfused LoRA (~0.3% overhead), static jobs run the untouched base (zero).

**Why it fits FlashDreams:**
- Zero real videos, ~2.5 GPU-days, 14.75M LoRA (0.3%); per-checkpoint re-instantiation ~1 GPU-day.
- Zero inference cost — merges into the weights.
- Gate-first portability: the α*(t) step-0 diagnostic measures headroom *before* training and
  predicted this port's gains; porting to another world model starts with a one-day measurement.
- No seam-pulse tax: that artifact belongs to lightly-adapted hosts, not production-trained ones.

**Known trade:** anchoring damps commanded motion at high gain (dynamics −30–40% at 1.0) — intrinsic
to reference-anchored correction (measured dichotomy on the paper host). Managed by the dial, and
static-background use cases *align* with it (background stability is the goal); validate new-element
entrances before shipping.

**HY-unique upside:** the dichotomy's escape is externally motion-conditioned correction — and HY's
future trajectory is a command input. A teacher conditioned on commanded future poses turns the
anchoring pull into trajectory-following. Owner-gated design, unavailable to T2V hosts.

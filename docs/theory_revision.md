# Theory revision (28 October 2026)

This note maps each comment on the theory (Sections 2–6 and 10 of the previous
draft) to what changed. The previous version is saved in
`versions/v2_2026-10-23_pre-theory-revision/` (PDF, LaTeX sources, calibration
results; git commit `f48a5a8`). The empirical sections (§7 Facts, §8 Causal
evidence, Appendix D) were not changed, except for one notation fix noted below.

Numbers below come from `code/results/results.json` (baseline calibration).

## Headline changes

1. **New central result: frontier incidence (Theorem 3).** Output per
   researcher equals the value of the frontier (Lemma 4, renewal reward). A
   tool's long-run value is `λ_F f + (1 − λ_F) h`, where `f` and `h` are its
   effects on the frontier value and on the newcomer's value at fixed density,
   and `λ_F = |ε_w| / (|ε_w| + θ_F ε_U + η_F D)`. Founders stay in the topics they
   open, so the frontier's value includes incumbency (`θ_F`). The share of a
   reading gain that survives entry is at least the volume elasticity times `D`
   (an observable). This replaces "reading improvements are dissipated", which
   held by construction.
2. **New: time-share lemma (Lemma 5).** Inside a topic every speed-up is worth
   exactly the time it saves (`∂ln v/∂ln b = r*/(1−r*)`). The old 26.1 vs 0.23
   comparison mostly reflected time shares. The paper now compares tools per
   hour saved.
3. **New: exact catch-up and the speed limit (Lemma 2, Theorem 2).** `T =
   (1/δ) ln(1 + δS/(b_c − P))`. Topics whose flow exceeds `b_c` close to
   newcomers; rents and the cross-topic gradient diverge near the limit;
   myopic entry stops below it; a production shock closes the densest topics.
   This forced a full recalibration.
4. **New: readership units (Proposition 8).** Reads per paper are
   `(b/a) r*/(1 − r*)`. Valued by readership, production tools add about
   nothing and the ranking reverses. Both units are reported.
5. **The frontier is a flow.** Explorers produce `e`, found topics at rate `s`,
   founders stay; topics die at rate `ξ` in the theory (not only in the
   calibration); births equal deaths.
6. **Topic life cycles: take-off and overshooting (Proposition 9).** With an
   empty stock at birth and the condition `v(P0) > F`, slow topics stall in the
   valley where `w < F` and are abandoned (new), fast ones overshoot.
   Forward-looking newcomers count payoffs only after catch-up.
7. **Policy depends on topic age (Proposition 11).** A flat frontier subsidy
   at the stationary optimum exceeds the newborn margin; a tax proportional to
   the stock reaches the optimum without deterring newborn topics. Welfare is
   evaluated on the life-cycle economy.
8. **New title:** *Too Fast to Read: Topic Choice and the Incidence of
   Research Tools*.

## Key numbers after the recalibration (baseline, pooled estimator)

| Object | Value |
|---|---|
| Calibration loss (fine grid), pooled / within-topic | 20.6 / 12.6 (previous version 14.9 / 11.4, with one target fewer and approximate catch-up) |
| Main miss | densest fifth +2.2% vs −2% in the data (as before); gradient 5.2% vs 6.7% |
| Peak of v; peak of w; mature density; speed limit `b_c` | 11.8; 4.3; 6.5; 38.6 papers/yr |
| Mature density as share of the speed limit (median / most fertile) | 17% / 33% |
| Catch-up time at mature density / at the peak | 0.7 / 1.5 years |
| Slope of the newcomer curve `|ε_w|` at `P^m` | 0.065 (0.037 with the old linear approximation) |
| Reading share of incumbents' time (target 7%) | 7.3% at `P^m`, 7.8% across researcher-years |
| 30% faster production: gain at fixed topics / long run | 26.2 / 26.0 log points |
| 30% faster reading: gain at fixed topics | 1.96 log points (same value per hour saved) |
| Share of a reading gain that survives entry: no founders / founders stay (ω = ½) | 14% / 74% |
| Share of a catch-up gain that survives: no founders / founders stay | 16% / 40% |
| Per hour saved, production vs reading tool (long run) | 1.3× (ω = ½) to 7.2× (ω = 0) |
| Life cycle, founders stay: frontier value response, reading / catch-up | 42% / 1% of the fixed-topic gain |
| Life cycle, output per researcher-year: reading / catch-up / production | 63% / 17% / 99% |
| Readership units (α = 1): 30% faster pen / 30% faster reading | +1.3 / +21.2 log points |
| Stationary planner's gain; flat subsidy; newborn margin | 1.25%; 1.25% of F; 1.28% |
| Life-cycle gain: flat subsidy / stock-proportional entry tax | 0.13% / 0.72% |
| Take-off threshold: laissez-faire / flat subsidy / stock tax | 2.2% / 3.6% / 3.1% a year |
| Overshooting threshold: myopic (closed-form bound) / forward-looking (bound) | 20.5% (21.4%) / 24.0% (29.1%) |
| Stalled-topic check: largest log entry gap after a stall | −0.0006 (never recovers) |
| Learning from the stock: newborn topics attract followers only if χ ≤ | 0.11 |
| Production rise needed to close the densest mature topic | ×3.0 (a fourfold rise closes topics with 32% of mature researcher-years) |
| Frontier depletion `η_F` (E = 0.15), founders / no founders | 0.017 / 0.012 |
| Stepping on toes: topics adjust / open topics fixed | 0.990 / 0.941 |

## Comment by comment

| Comment | Decision | What changed |
|---|---|---|
| 1a. Pioneers neither read nor stay, so dissipation holds by construction; founders who stay give 35–90% pass-through; σ belongs in π(M) | **Adopted** | Founders stay (Lemma 3, `θ_F`); Theorem 3 and Corollary 2; pass-through reported for founders' probability ω ∈ {0, 0.25, 0.5, 0.75}, stationary and life cycle; pioneers' search enters through `(θ_F − ω) ε_sσ` (§6). |
| 1b. Time shares: 30% faster reading worth ~1.9 log points at fixed topics; per hour saved writing beats reading 7.3×, not 113×; 7% reading share undisciplined | **Adopted** | Lemma 5; tools compared per hour saved; reading share is now a calibration target (7%: 144 hours a year of article reading, Tenopir & King 2008, out of ~2,000 working hours). |
| 1c. Welfare counts papers, not readership | **Adopted** | Proposition 8 and Table 9, panel B: output valued as `A^α Y` for α ∈ {0, 0.5, 1}. |
| 2. Exact catch-up and the speed limit | **Adopted** | Lemma 2, Theorem 2; recalibration with exact `Γ(P) = (1 − P/b_c)^ν`, `ν = (ρ+ξ)/δ`. |
| 2. Frontier mixes static and dynamic | **Adopted** | Frontier as a flow; topics die at rate ξ in the model; births = deaths. |
| 2. Knife-edge `v(P0)` vs `F`; age-dependent policy; welfare on the life cycle | **Adopted** | Remark 6, Proposition 11, life-cycle welfare (flat subsidy vs stock tax). |
| 2. Flow vs stock | **Adopted** (justified + extension) | Justified in §2; Appendix B.4 lets learning draw on the stock: stationary results unchanged; with the calibrated newborn margin, newborn topics attract followers only if at most a small share of learning draws on the stock. |
| 2. Detection via targeted per-project search; concurrent overlap; Hill–Stein validates racing, not λ | **Adopted** | `φ` read as targeted search; Appendix B.5 (per-project search: `a` no longer drops out, faster production lowers search per project); concurrent overlap `θ_c λ_c` in Lemma 1 and Theorem 1; Hill–Stein linked to `λ_c`. Calibration keeps `Θ_c = 0` (not separately identified). |
| 2. η assumes a fixed frontier | **Adopted** | Caveat after (6): frontier improvements bias η_F down. |
| 3. Prop 9: use `v(P0) > F` with an empty stock | **Fixed** | Proposition 9 uses the exact growth stock with finite `P0`. |
| 3. Appendix C: `1/|ε_w| ≈ 7` is wrong | **Fixed** | Number now generated from the calibration. |
| 3. `U(P)` should count payoffs after catch-up | **Fixed** | Forward-looking value counts payoffs after catch-up along the thinning path. |
| 3. Define Jones's λ | **Fixed** | Proposition 10 reports both `1 − E|ε_w|` (topics adjust) and `1 + ε_w/D` (open topics fixed). |
| 3. `a` is production speed, not drafting speed | **Fixed** | Throughout; drafting tools raise `a` by the drafting share of production time. |
| 3. Role of the detection microfoundation | **Fixed** | Paragraph after Lemma 1 (aware share, invariance test, search channel, falling arm). |
| 4. Frontier-incidence theorem | **Adopted** | Theorem 3. |
| 4. Prove the speed limit (Chu & Evans) | **Adopted** | Theorem 2, linked to Chu & Evans (2021). |
| 4. Readership | **Adopted** | Proposition 8. |
| 4. Continuous idea space | **Partly adopted** | Appendix B.3: nested topics inside fields deliver the radius pattern of Fact 1 (Proposition 13). The fully continuous version is left for future work. |

## Notation fix in the empirical part

* §8 (CS review rule): "a higher κ" → "a lower `b_c`" (κ no longer exists).

## Follow-ups for the empirical round (not done here)

* Fact 3/Lemma 1: the −19% unaware-duplicate penalty identifies λ; the
  Hill–Stein 21% scooping penalty speaks to concurrent overlap λ_c. The facts
  section still compares the two directly.
* §6 "What we no longer claim" cites the model's flagged-overlap responses
  ("7–30%"); check against the new Table 2.
* §8 LLM design is described as "faster drafting"; in the model this is a
  shift in production speed `a` by the drafting share.
* Measuring the founders' share `θ_F` (founders who keep publishing in the
  clusters they open) would pin down the pass-through of reading tools.

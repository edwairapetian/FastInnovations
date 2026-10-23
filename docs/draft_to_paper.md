# From the core model draft to the paper

This note maps each claim and open problem in the *Core Model Draft* (Oct 13,
2026) to its status in the first version of the paper. Changes made after the
Top-5 readiness review (Oct 21, 2026), including corrected numbers,
forward-looking entry in the "too fast" result, the social-value block and the
quasi-experimental evidence, are in `docs/review_response.md`, which supersedes
this note where they differ.

## The draft's six propositions

| Draft | Status in the paper |
|---|---|
| 1. Dissipation (payoff = aF in every occupied topic) | **Proved** for general Q, φ (Prop. 2) in the frictionless benchmark. **Rejected by Test 4** as a description of the data. Replaced by catch-up rents: incumbents earn aF·e^{κP}, newcomers aF over their careers (Prop. 5). Dissipation holds for *career* payoffs, not per-period returns. |
| 2. Pinned overlap (density independent of N, a, c) | **Proved** (Prop. 3) with an exact elasticity, `E = η/(ηD+|ε_w|)`. Density depends on (N, a) only through Na. With η = 0 it is exactly pinned. Under fertility depletion with a power law it is pinned for *any* depletion rate (App. B.1). Test 1 (E ≈ 0.04–0.27) identifies η ≈ 0.03 (E = 0.2). |
| 3. Negative externality at every stable equilibrium | **Proved** via the envelope theorem (Props. 1–2). With catch-up costs it is strengthened: the relevant slope is that of the newcomer curve w, so the marginal entrant is harmful even where incumbents' returns still rise (Prop. 6). |
| 4. Overcrowding (planner density lower, more topics) | **Proved** (Prop. 4), with an explicit planner condition. In the large-topic limit the planner sits at the peak of w, the writing wedge vanishes there, and a frontier subsidy decentralises the optimum. |
| 5. Stepping on toes, `λ = 1 − η·dlnM/dlnN` | **Proved and sharpened** (Prop. 9): `λ_J = 1 − E·|ε_w|`, a product of two observables. The draft said crowding *amplifies* frontier depletion. It actually **buffers** it: the density margin absorbs growth. Crowding costs show up in levels, not elasticities. Calibrated λ_J ≈ 0.975 (0.966–0.995 over the arXiv range of E), so crowding explains little of Bloom et al.'s decline. |
| 6. Hump in a only if η is large | **Confirmed** exactly: d ln Y/d ln a = 1 − E|ε_w|, which is negative only if η·(|ε_w| − D) > |ε_w|. For the calibrated η, writing speed raises output almost one for one (Prop. 10). |

## The draft's hard problems

| Problem | Resolution |
|---|---|
| 1. Single-peaked v | **Theorem 1.** Joint log-concavity of the researcher's problem in (reading time, density): ln Q jointly concave, φ concave (the erosion term is a perspective of φ, which preserves concavity), and partial maximisation preserves concavity. No parametric assumption is needed. |
| 2. Coordination at the frontier | **Remark 1.** With a continuum of topics, rising-branch crossings are unstable. Newborn topics attract followers iff F < v(P₀). Only when the frontier beats a newborn topic can topics be stuck too thin. The calibration is in the first case. |
| 3. Dynamics / unread stock | **Catch-up cost** with stock S = P/δ and career value w = v e^{−κP}, κ = ρ/(δ b_c) (Lemma 2). The literal "too fast" result is **Prop. 7**: topics whose flow grows faster than `g* = ρP̂/(b_c ln(v̂/F)) − δ` overshoot the peak, because the stock lags the flow. The entry margin prevents explosion in steady state. |
| 4. Heterogeneous ability | Discussed (App. B.3), with a design that separates sorting from crowding. Not modelled formally. |
| 5. Credit rules | Partial credit for duplicates (λ < 1) and for aware follow-ons (selection on δ) is in the baseline (Lemma 1). Only λ > ℓ is needed. The externality sign survives. |
| 6. Identification of η | Equation (9): η = E|ε_w|/(1 − ED). |

## What the arXiv tests changed

* **Catch-up cost added** (Test 4). It explains a positive returns gradient: across mature topics, the gradient equals the premium κP^m (Prop. 5(iv)). It also explains a peak and a decline at the top, where topics that grew faster than g* are consolidating.
* **Welfare made sharper, not two-sided.** The memo suggested 88% of researcher-years being on the rising side means most topics are too thin. The model shows this does not follow. The rising incumbent gradient is the catch-up rent. Once the reading imposed on future newcomers is counted, the marginal entrant is harmful (Prop. 6). Too-thin topics arise only from information frictions (young topics) or coordination failure (F > v(P₀)).
* **AI acts through entry.** Flat papers per author, more pioneers, a flat aware share and flat overlap by meaning all reject the writing, reading, search and catch-up channels as dominant in 2023–26 (Section 7).
* **Aware share.** The tested figure (35–42%) replaces the draft's "about 60%".

## New results not in the draft

* Writing AI scales science; reading AI crowds it (Props. 8, 10). Reading improvements are dissipated when η is small: a "Solow paradox for science".
* Faster catch-up tools *lower* g* and so make overshooting more common (Prop. 8(iv)).
* Fertility depletion leaves density exactly scale invariant and makes λ_J = 1 − E|ε_w| an upper bound (App. B.1).
* Held-out predictions P1–P8 (Section 10), testable with existing files.

## Calibration caveats

* Test 4's quadratic is symmetric; the model's payoff curve and the arXiv fifths
  are not. We target the fifths, the linear gradient, the share beyond the peak,
  the aware share and a catch-up premium.
* The model reproduces the location of the peak and the share beyond it, but a
  smaller fall at the top than the data. See `paper/tables/fit.tex`.
* Units: b = 100 papers read per year at full time sets the unit of density.

# Response to the Top-5 Readiness Review (Oct 21, 2026)

This note maps each item in the review to what changed in `paper/main.pdf`, the
code that produces it, and what is still open. Numbers come from
`empirics/results/*.json` and `code/results/results.json`.

## The six blocking items

| # | Review | What changed |
|---|---|---|
| 1 | Fact 6 (pioneers) is the TF-IDF wording artifact | **Dropped.** §6 "What we no longer claim" explains why (wording shift; −0.8 pp without detrending). |
| 2 | "AI acted through entry" conflicts with Kusumegi et al. | **Conclusion removed.** New author-level panel with name-keyed identities and adoption classified *before* the outcome window (`empirics/08_llm_authors.py`): adopters of LLM-style writing post about **7% more** papers in 2024–26 (0.022 log(1+papers), SE 0.005), similar in fields where AI is not the subject. Cites Kusumegi et al. (Science 390, 2025) and the Renault–Bergeaud–Bosquet critique of their timing rule. |
| 3 | Two calibration targets mismeasured | **Corrected and recalibrated.** Aware share 0.60 (recall-corrected), fifths −11/+2/−2% relative to the middle fifth. Independent check: INSPIRE-HEP curated references give a 61% aware share at the 90th-percentile overlap radius. |
| 4 | No causal identification | **New §7 "Evidence II"** with four arXiv rule changes and design-based variation (below). |
| 5 | Theory core is Henderson (1974) / clubs | **Positioned up front** (intro, Remark 3), with the four claims that are new: microfounded hump, readers as crowders, volume invariance of duplication, writing–reading asymmetry and stepping-on-toes statistic; plus the incumbent–entrant gap. |
| 6 | Mechanism is small; 17% is the counterfactual | **Lead with calibrated numbers** (planner gain), frictionless benchmark labelled a counterfactual. Headline changed to the writing–reading asymmetry, whose size depends on the volume elasticity rather than on crowding losses (§5). Social-value block shows the welfare result is robust to transfers. |

## Causal designs (new)

| Design | Shock | Result |
|---|---|---|
| Listing position, cutoff RD | Submitting just after 14:00 ET opens the next listing | Sharp first stage (top-5 probability +0.69). Comparison is first vs **last**, and last is also visible (Haque & Ginsparg 2010; Feenberg et al. 2017); top papers get *fewer* citations in HEP (−0.14, SE 0.06). Not used to test detection. |
| Listing rank, HEP population (87,694 papers) | Rank depends on how many others submitted earlier that day, holding submission time, date and list size fixed | Small readership effect (−0.023 per log position, SE 0.016). **No detectable effect on later overlap by others or on awareness** (unaware overlap: CI rules out more than about +4% per log point). Reading protects through targeted search, not browsing. |
| Endorsement tightening (Math 10 Dec 2025; all 21 Jan 2026) | Entry cost | First-time submitter share −2.4 pp (Math) and −3.4 pp (others), about −15–17%, net of the year before; staggered timing confirmed (−1.7 pp, SE 0.4, Math vs others in the six-week window). |
| CS review/position papers need peer review (31 Oct 2025) | Catch-up cost (reviews) | Review-type papers in CS halved (−1.32 pp, SE 0.10), no leakage; dose-response first stage across 29 CS categories; first-time authorship fell more where reviews mattered (−0.70, SE 0.26; placebo −0.32, SE 0.17), so suggestive. |
| Two submissions per person per month (1 Oct 2026) | Writing cap | Removed about 1.3% of papers (third-plus submissions 1.66% → 0.38%); volume growth unaffected (+0.22 vs +0.175 seasonal norm). |
| Author-level LLM adoption | Drafting speed | about +7% papers (see item 2). |

Designs stated in advance but not run here: Great Firewall (2014) as a search-cost shock (needs author affiliations), and topic-level outcomes of the 2025–26 rule changes (need a year of post data). See §10 of the paper.

## Evidence problems

| Fact | Fix |
|---|---|
| 1 Volume | SEs and radius reported: 0.035 (0.118) at the overlap radius, 0.12–0.19 broad radii; benchmark 0.53–0.81; E = 0.15 baseline, range 0–0.27. Short-run vs long-run gap bounded (Remark 4: composition bias ≈ 0.06 over seven years). |
| 2 Papers per author | Author-level panel (above). |
| 3 Returns hump | Labels fixed (−11/+2/−2); SE units fixed (0.15 pp); audience caveat promoted: gradient treated as an upper bound on the catch-up premium; papers per author +0.5%/SD reported as a failed (small) prediction of Theorem 1(v). |
| 5 Aware share | Recall-corrected 60%, drift reported (+3.8 pp, SE 1.3), INSPIRE validation. |
| 7 Overlap by meaning | MDE reported (≈0.67 pp, two thirds of the base); no longer used. |
| Exposure | AI-as-subject caveat; author-level design restricted to non-AI fields. |
| "Pre-registered" | Removed; "out-of-sample predictions", with a note that they should be deposited before testing. |
| Reconciliations | 1.49M papers, +122%, consistent periods. |

## Theory

| # | Review | Change |
|---|---|---|
| 1 | System of cities | Remark 3 and intro. |
| 2 | Theorem 1 generality partly assumed | Remark 2: hump from bounded absorption; complementarities among papers read are ruled out and would allow several peaks. |
| 3 | Fixed topic size | Remark 1: topic = unit cell of idea space; P = papers per unit of idea space, what nearest-neighbour similarity measures. |
| 4 | "Too fast" not an equilibrium | Prop. 7 now covers forward-looking entry: overshooting for all g > δ(P̂/Pᵐ − 1); none below g*; fast topics overshoot *more* under foresight and pass the frictionless density. The friction is the stock lag. Proof in App. A; numerical checks in tests. |
| 5 | Prop. 6 near-tautology | Now Corollary 1; weight on the v–w gap. |
| 6 | Welfare = private credit | Prop. 6 (social value): uniform wedge ≡ change in κ; overcrowding survives any transfer share when the entrant slope exceeds θλPᵐ (true in the calibration); planner gain under full transfer reported. |
| 7 | Stationary model, growing literature | Remark 4 with formula (6) and calibrated bound. |
| 8 | Quantitative bite small | Reframed around the asymmetry; magnitudes reported honestly. |
| 9 | Papers per researcher prediction fails softly | Reported in Fact 2. |
| 10 | Sign error in Prop. 10 | Fixed: d ln(Y/N)/dω = −η d ln M*/dω ≥ 0; new test `test_prop10_sign_with_elastic_frontier`. |

## Literature and presentation

All missing works added with verified bibliographic details (Kusumegi et al.; Henderson; Buchanan; Henderson & Becker; Scotchmer; Carnehl & Schneider; Hill & Stein QJE and JPE; Bobtcheff, Bolte & Mariotti; Borjas & Doran 2015; Hoberg & Phillips; Kelly et al.; Duede et al.; Jones 2025; Agrawal, Goldfarb & Teodoridis; Kitcher; Strevens; Haque & Ginsparg 2009, 2010; Feenberg et al.; Zheng & Wang; McMahan & McFarland; Kobak et al.; Renault et al.). No `VERIFY` entries remain. Internal references ("[Authors]", "the draft", "the memo", "Test 1/4", "pre-registered") removed. New title around the headline. Each fact and design has its own exhibit. The replication package (`code/`, `empirics/`) is in the repository.

## Still open (needs the authors or resources outside this repository)

* Deposit the predictions of §10 in a public registry (OSF) before testing them.
* Hand-validate a sample of overlap flags (precision), and re-run Facts 1–3 with area-clustered SEs and embeddings using the project's OpenAlex panel (not in this repository).
* Great Firewall design: needs author affiliations (OpenAlex with an API key).
* All-field citations for the cutoff RD: the Semantic Scholar fetch was rate limited and covers part of the sample (HEP uses complete INSPIRE data).
* The calibration does not reproduce the densest fifth falling below the middle (model +2.5% vs data −2%); see §8.

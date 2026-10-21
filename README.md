# When Innovation Is Too Fast: Reading, Writing, and the Crowding of Science

This repository contains the theory paper for the project *When Innovations Are
Too Fast*, together with the code that solves, verifies and calibrates the
model. Every number, figure and table in the paper is produced by the code.

* `paper/main.pdf`: the paper (compiled from `paper/main.tex` and `paper/sections/`).
* `docs/empirical_facts.md`: the arXiv evidence the model is built to match,
  taken from the project's two memos (core model draft, Oct 13 2026, and
  *Testing the Model on arXiv*, Oct 20 2026).
* `code/ftmodel/`: the model.
  * `payoff.py`: the researcher's problem and the payoff curve `v(P)` (Theorem 1).
  * `equilibrium.py`: stationary topic equilibrium, planner, sufficient statistics (Propositions 2–4, 9–10).
  * `lifecycle.py`: topic life cycles with catch-up costs, the cross-section, and the Test 4 regressions (Propositions 5–7).
  * `calibration.py`: calibration to the arXiv moments.
* `code/tests/test_propositions.py`: numerical checks of every analytical result (`pytest`).
* `code/scripts/`: scripts that run the calibration and build all figures and tables.

## The argument in one paragraph

Reading teaches (it raises the quality of new work) and protects (it reveals
projects others have already done). With free topic choice, the payoff of
working in a topic is single peaked in the topic's literature flow (Theorem 1,
proved for general learning and detection technologies). Researchers crowd
topics past the peak, so the gains from cumulative knowledge are competed away
and the overlap rate is pinned by entry, not by volume (Propositions 2–3). A
catch-up cost, the need to read a topic's stock before contributing, protects
incumbents' rents. Measured returns therefore rise with density even though
every stable equilibrium is overcrowded at the margin (Propositions 5–6).
Topics whose literature grows faster than a threshold rate `g*` overshoot
their best density (Proposition 7, "too fast"). Faster writing scales science;
better reading crowds it, and its gains are dissipated (Propositions 8 and 10).
The stepping-on-toes parameter of growth theory is `λ = 1 − E·|ε_w|`, a product
of two observable elasticities (Proposition 9).

## Reproducing

```bash
pip install numpy scipy matplotlib pytest
cd code
python -m pytest -q tests                  # verify all propositions numerically
python scripts/run_calibration.py          # calibrate (writes results/calibration.json)
python scripts/make_figures.py             # figures and tables for the paper
cd ../paper && latexmk -pdf main.tex       # compile the paper
```

## Status and open items

* The bibliography was assembled from memory. Entries marked `% VERIFY` in
  `paper/references.bib` need checking against the published versions before
  submission.
* The calibration uses the moments reported in the arXiv memo. The underlying
  microdata (`analysis/data/*.parquet`) are not in this repository.

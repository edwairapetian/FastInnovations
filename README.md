# Faster Writing Scales Science, Better Reading Crowds It

Theory, evidence and replication code for the paper *Faster Writing Scales
Science, Better Reading Crowds It: Topic Choice and the Economics of
Duplication* (project: *When Innovations Are Too Fast*). Every number, figure
and table in the paper is produced by the code in this repository.

* `paper/main.pdf`: the paper (compiled from `paper/main.tex` and `paper/sections/`).
* `docs/review_response.md`: point-by-point response to the Top-5 readiness review.
* `docs/empirical_facts.md`: the descriptive arXiv evidence, with the corrections used in the paper.
* `docs/draft_to_paper.md`: how the core model draft maps into the paper.
* `code/ftmodel/`: the model.
  * `payoff.py`: the researcher's problem, the payoff curve `v(P)` (Theorem 1) and social value (Proposition 6).
  * `equilibrium.py`: stationary equilibrium, planner, sufficient statistics.
  * `lifecycle.py`: topic life cycles, myopic and forward-looking overshooting (Proposition 7), the cross-section, and the composition bias of short-run elasticities.
  * `calibration.py`, `report.py`: calibration and quantitative results.
* `code/tests/test_propositions.py`: numerical checks of every analytical result (`pytest`).
* `code/scripts/`: calibration and all model figures and tables.
* `empirics/`: the quasi-experimental evidence (Section 7), run in numerical order:
  * `01_parse_snapshot.py`: arXiv metadata snapshot → parquet.
  * `02_listing_position.py`: daily announcement batches and listing positions.
  * `03_rd_sample.py`, `04_near_duplicates.py`, `07_rd.py`: cutoff design.
  * `09_hep_sample.py`, `10_hep_pairs.py`, `11_rank_design.py`: listing-rank design (INSPIRE-HEP).
  * `05_cap.py`: two-submissions-per-month cap.
  * `06_entry_policies.py`: endorsement rule and CS review policy.
  * `08_llm_authors.py`: author-level adoption of LLM-style writing.
  * `12_tables.py`: tables, number macros and figures for the paper.
  * `s2_fetch.py`, `inspire_fetch.py`: citation and reference data.

## The argument in one paragraph

Reading teaches (it raises the quality of new work) and protects (it reveals
projects others have already done). For any learning and detection technology
with bounded absorption, the payoff of working in a topic is single peaked in
the topic's literature flow (Theorem 1). Free topic choice then has the
structure of a system of cities: topics fill past their best size and the
duplication rate is pinned by entry, not by volume. Faster writing raises the
payoff of opening new topics as much as that of joining old ones, so it scales
output almost one for one; better reading, search and catch-up tools raise only
the payoff of existing topics and are competed away (Propositions 8 and 10). A
catch-up cost protects incumbents' rents, so returns rise with density while
topics are too crowded for newcomers (Corollary 1), a result that survives
treating priority credit as a transfer (Proposition 6). Topics that grow fast
overshoot, also with forward-looking entrants (Proposition 7). On arXiv, volume
opens new ground; endorsement and review-article rules move entry and catch-up
as the model requires; listing position barely moves readership and does not
change detection of overlapping work.

## Reproducing

```bash
pip install numpy scipy pandas pyarrow scikit-learn statsmodels matplotlib pytest
# model
cd code
python -m pytest -q tests
python scripts/run_calibration.py --no-polish   # or --search for the full search (~1 hour)
python scripts/make_figures.py
# evidence (needs the arXiv snapshot: kaggle.com/datasets/Cornell-University/arxiv)
cd ..
mkdir -p data_raw data && unzip arxiv.zip -d data_raw
python empirics/01_parse_snapshot.py
python empirics/02_listing_position.py
python empirics/03_rd_sample.py && python empirics/04_near_duplicates.py
python empirics/05_cap.py && python empirics/06_entry_policies.py
python empirics/s2_fetch.py data_raw/rd_ids.txt data_raw/s2_focal.jsonl citationCount   # rate limited
python empirics/07_rd.py && python empirics/08_llm_authors.py
python empirics/09_hep_sample.py && python empirics/10_hep_pairs.py
python empirics/inspire_fetch.py data_raw/hep_all_ids.txt data_raw/insp_focal.jsonl
python empirics/inspire_fetch.py data_raw/hep_pairs_j.txt data_raw/insp_hep_j.jsonl refs
python empirics/11_rank_design.py && python empirics/12_tables.py
cd paper && latexmk -pdf main.tex
```

Raw data (`data_raw/`) and intermediate parquet files (`data/`) are not tracked.
The descriptive facts of Section 6 come from the project's OpenAlex panel, which
is not in this repository.

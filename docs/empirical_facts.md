# Empirical facts the model must match

Source: the project's two internal memos, *Core Model Draft: Reading, Writing and
Duplication* (Oct 13, 2026) and *Testing the Model on arXiv* (Oct 20, 2026), as
corrected by the *Top-5 Readiness Review* (Oct 21, 2026). The tests were
written down before they were run but were not deposited in a public
registry, so the paper calls them out-of-sample predictions, not
pre-registered tests.

## Corrections after the readiness review (used in the paper)

| Item | Memo | Corrected |
|---|---|---|
| Analysis frame | 1.71M papers | **1.49M** (2018–2025), 1.65M through mid-2026 |
| Volume growth 2018→2025 | +113% | **+122%** |
| F3 fifths "relative to the middle fifth" | −8% / +5% / +1% (actually deviations from each author's mean) | **−11% / +2% / −2%** relative to the middle fifth |
| F3 gradient SE | "SE 0.15" | SE **0.15 percentage points** (0.0015), author-clustered |
| F1 at the overlap (99th pct.) radius | — | **0.035 (SE 0.118)**, interval −0.20 to 0.27; 0.12–0.19 from the 95th to 25th pct. radius; benchmark 0.53–0.81 |
| F5 aware share | 35–42% raw | **≈60%** after the 63% citation recall; recall drifts (arXiv ids per PDF 5.0→8.8; pre/post +3.8 pp, SE 1.3); INSPIRE-HEP check: 61% at the 90th pct. similarity |
| F6 pioneers | +1.3 pp (0.5) detrended | **dropped**: TF-IDF wording artifact; −0.8 pp (0.2) without detrending |
| F7 overlap by meaning | −0.19 pp (0.24) | **uninformative**: MDE ≈ 0.67 pp ≈ two thirds of the base |
| F2 flat papers per author | area-level ratio | author-level LLM adopters: **≈ +7%** (this repository, `empirics/08_llm_authors.py`) |
| Exposure | "high uptake" areas | 13 of the 15 highest-uptake areas study AI or vision (AI as subject) |

The calibration targets in `code/ftmodel/calibration.py` use the corrected
values: fifths −11/+2/−2%, aware share 0.60, E = 0.15 (range 0–0.27).

Data: arXiv papers 2018–2025 (1.71M), 150 text-based fine areas; a PDF sample
of 25,400 papers (10,200 flagged); OpenAlex citations (early October 2026
snapshot); abstracts re-read from the 14 October 2026 OpenAlex snapshot for
the embedding check (233,880 papers in 8 coarse areas).

**Overlap flag.** A paper is flagged when an earlier paper (3–24 months back,
other research groups) lies above its area's 99th percentile of 2018–19
nearest-neighbour similarity. The *level* of the flagged rate is
therefore a normalisation (about 1% by construction in the base period).
Only changes and ratios carry information.

## F1. Volume opens new ground (Test 1, the paper's anchor)

| Object | Value |
|---|---|
| Elasticity of local density (neighbour hazard) to area volume, area and year FE | **0.04–0.19** across radii (closest radius lowest) |
| Same, before 2023 | 0.15–0.30 |
| Same, coarse-area volume | −0.08 to 0.17 |
| All-arXiv time series | volume +113% (2018→2025), overlap hazard +13%; elasticity 0.19–0.27 |
| Fixed-field benchmark (random thinning of observed neighbour lists) | **0.53–0.89**, highest at the closest radius |

Reading: roughly three quarters of growth goes to new ground. Close overlap
responds least, consistent with researchers steering away from papers they
have read. Caveat: only 58% of nearest neighbours lie in the same fine area,
so area volume is a noisy proxy (attenuation toward zero). The time series
does not have this problem.

## F2. Growth came from new authors

Papers per author stayed flat while the number of authors grew. In the model
this points to cheaper entry (c↓), not faster drafting (a↑).

## F3. Returns are hump-shaped in topic density (Test 4)

Same researcher across years in denser or sparser topics; author and year
FE; 923,000 author-years (311,000 authors with 2+ active years), papers
2018–2022, ≤30 authors, fractional counts, OpenAlex citations.

| Object | Value |
|---|---|
| Linear: citations per author-year per SD of density | **+6.7%** (SE 0.15%) |
| Citations per paper per SD | +9.5% |
| Papers per year per SD | +0.5% (hump, no linear gradient) |
| Quadratic: linear term, squared term | **0.094, −0.047** (SE ≤ 0.002) → peak ≈ **+1 SD** |
| By fifth of density (memo label "relative to the middle fifth"; these are deviations from each author's mean) | sparsest −8%, +5%, densest +1%; **relative to the middle fifth: −11%, +2%, −2%** |
| Researcher-years beyond the peak | **12%** |
| Robustness | holds with the 3rd-closest paper, and for researchers active 3+ years |

Interpretation in the memo: full dissipation fails, but the shape is the model's
payoff curve. Two candidate frictions are named: switching is costly
(catch-up), or pioneers cannot coordinate. Citations also reward audience
size, which overstates the social gain from density.

## F4. Credit for overlap (Test 4b)

2018–2021 papers with OpenAlex reference lists, compared with papers in the
same area-year and the same topic density.

| Group | Papers | Median citations | Citations vs comparable |
|---|---|---|---|
| Unaware overlap (does not cite nearest earlier paper) | 3,618 | 3 | **−19%** (SE 2%) |
| Aware overlap (cites it) | 1,381 | 13 | **+29%** (SE 5%) |
| Not flagged | 468,568 | 4 | reference |

Duplicates are penalised but do not earn zero. Aware follow-ons are rewarded.

## F5. Awareness did not rise (Test 3)

* Aware share of close overlap: **35–42% every year 2019–2026** (2026: 42%).
* Difference-in-differences by LLM uptake: −0.5 pp per SD (SE 1.4).
* Unflagged papers cite their nearest earlier paper more often (6.4% in
  2019 → 8.9% in 2026) as reference lists lengthen.

(The core draft quoted "about 60% of flagged pairs cite the earlier paper".
The tested PDF-sample figure, 35–42%, supersedes it.)

## F6. No extra concentration where LLMs spread (Test 2)

Area × quarter panel, 150 areas, 2019Q1–2026Q2. Exposure is the z-scored rise
in LLM marker words from 2019–21 to 2024.

* Pre-trend: high-uptake areas were already losing ~0.4 pp of pioneer papers
  per year per SD of uptake before ChatGPT.
* Against the extended trend, pioneer papers are **+0.4 pp (SE 0.3)** in
  2023–2025H1 and **+1.3 pp (SE 0.5)** from 2025H2.
* Density within the topic radius: −1.1 pp and −2.7 pp.

Nothing supports concentration. The late break coincides with the wording
change in F7.

## F7. The 2026 drop in overlap is a change in wording (Test 5)

* From 2025H2 to 2026H1 the whole distribution of nearest-neighbour TF-IDF
  similarity fell by ~0.01: 10th pct 0.151→0.144, median 0.239→0.230,
  99th 0.482→0.468.
* The drop sits in high-uptake areas (−0.48 pp vs −0.04 pp), not dense
  ones (−0.27 to −0.31 pp everywhere).
* With sentence embeddings (all-MiniLM-L6-v2), close overlap did **not**
  fall: high-uptake +0.03 pp (0.20), low-uptake +0.13 pp (0.15).
* Abstracts in high-uptake areas: 6% longer, 13% more distinct words, 30%
  more rare words, 55% more hyphenated compounds. LLM marker words fell from
  33% to 21% of abstracts, so marker counts now understate LLM use.
* TF-IDF and embedding flags agree on only about a fifth of flagged papers.
  Overlap from 2025 on should be rebuilt with embeddings.

## What the memo asks of the model

1. Keep the topic-choice core. Proposition 2 is the anchor: a density
   elasticity near 0.2, not the 0.5–0.9 a fixed field implies.
2. Replace full dissipation with a catch-up cost: joining a topic means
   reading its stock of literature first.
3. Allow both too many small topics and overcrowded hot ones (the welfare
   sign may be two-sided).
4. Keep aware and unaware overlap separate, with partial credit.
5. AI acts through entry so far. Reading and search channels are
   predictions for when literature-summarising tools spread.

The model in `paper/` does all five. On point 3 it gives a sharper answer: the
positive returns gradient is the rent that catch-up costs protect, not an
untapped spillover. Every stable equilibrium is overcrowded at the margin
once the catch-up burden imposed on future entrants is counted. The
exceptions are young topics that information frictions keep thin.

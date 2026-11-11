# What the paper is, and how to make it cohere (11 November 2026)

This note analyses the current draft (*Too Fast to Read: Topic Choice and the
Incidence of Research Tools*, commit `866dcfd`, 66 pages). It does not change
the paper. It rests on five independent readings of the draft (argument,
theory, fit between evidence and theory, quantification, a referee's report),
on adversarial re-checks that recomputed the technical findings from the code
and data in this repository, and on a second round in which four alternative
designs of the paper were scored by three editors.

Numbers come from `code/results/results.json`, `data/arxiv_meta.parquet`,
`data/hep_pairs.parquet` and `empirics/results/*.json`. Where a number was
recomputed by a second reader this is the recomputed value.

## 1. Bottom line

*Interim version. The diagnosis in Sections 2–4 and 6 is complete. The
recommendation in Section 5 is provisional until the design round (four
alternative designs of the paper, scored by three editors) has returned.*

The draft is four papers in one, and its two headline results are the two
that the evidence does not test. The theory is internally sound in its parts
but not at its centre: the step from the frontier value to output is
inconsistent with the paper's own accounting, the headline range comes from an
assumed parameter, and the title result never operates in the paper's own
numbers. The material does support one coherent paper. Its subject is the
entry margin: science absorbs growth by opening topics, the price of entry is
reading a topic's stock, and so the gains from any tool go to those who are
not on the entry margin.

## 2. The idea that ties the pieces together

The draft has one idea, which it presents as four separate results:

> Entry is the margin that equilibrates science, and the price of entry into a
> topic is reading its stock of literature.

Four consequences follow.

1. **More researchers widen science.** A free researcher is indifferent
   between joining a topic and opening one, so extra researchers open topics
   and density barely moves (Proposition 3, Remark 4, Fact 1).
2. **Insiders earn a rent equal to the entry cost.** Returns rise with density
   across topics although the dense topics are already too crowded for a
   newcomer (Proposition 5, Corollary 1, the gradient in Fact 2). At the
   calibrated mature density the newcomer's payoff has slope −0.065: +0.082
   from learning, −0.035 from overlap, −0.113 from catch-up. Catch-up is the
   crowding force. Reading has three jobs (teach, protect, admit), and the
   third carries the model.
3. **A tool's long-run gain goes to whoever is not on the entry margin.**
   These are the people who open topics, the people who entered before entry
   was free, and a scarce frontier. A tool that only makes entry cheaper is
   absorbed by denser topics and a longer catch-up (Theorem 3, once repaired).
4. **Faster researchers are not the same as more researchers.** When the same
   people write more, every topic's flow rises at once, and the adjustment
   takes as long as people take to move: a half-life of about seven years at
   the assumed exit rate of 10% a year (Appendix B.8, Theorem 2(iv)). The speed limit is the extreme
   case of this.

The first two are supported by evidence in hand. The third is a theorem that
needs one repair and one measurement. The fourth is where the title lives. It
is currently a theorem that never binds, and the data that could test its
short-run version are on disk and unused.

## 3. What is solid

| Result | Why it can be relied on |
|---|---|
| Theorem 1 (single-peaked payoff) and Lemma 5 (time shares) | Proofs rederived. Inside a topic a tool is worth the time it saves: a 30% production tool is worth 26.2 log points, a 30% reading tool 1.96, the same per hour. |
| Proposition 5 and Corollary 1 (rents) | The premium equals `(ρ+ξ)T`, the cross-topic gradient equals `ε_Γ`, and `v′ > 0 > w′` at mature density. None of this needs the speed limit, the founders' assumption or Lemma 4. |
| Theorem 3 as a statement about the frontier value `F` | Algebra rederived; matches re-solved equilibria for small shocks. |
| Fact 1, qualitatively | Every estimate of the volume elasticity of local density (0.035 to 0.30) is far below the fixed-idea-space benchmark (0.53 to 0.89). |
| Three small magnitudes | Static gain from optimal topic choice 0.15–1.25% across every calibration variant tried (3.2% in the within-topic one); stepping on toes from crowding 0.98–1.00; credit lost to overlap 2.9%. |
| Data and code | 2.2 million timestamped arXiv submissions to 15 October 2026, HEP pairs, 61,000 INSPIRE reference lists, two clean first stages. The code reproduces every reported number and all 111 tests pass. |

## 4. What does not hold up

### 4.1 Neither headline is tested

The two evidence sections were written for the previous version and were not
revised (one token differs). The headline now rests on the founders' share
`θ_F`, the catch-up speed `b_c`, the reading share, and the indifference
between topics and the frontier. None of the six designs identifies any of
them. The abstract's claim that four rule changes "identify the margins the
model relies on" is not supported.

### 4.2 The central theorem needs one repair, and the headline range is not identified

* **Lemma 4 (`Y = aNF`) does not follow from the paper's own member count.**
  Lemma 2 and the adding-up condition imply `Y/N = aF·e^{ξT}`, 1.2–3.0% above
  `aF`. The difference is the rent of members who never paid the stationary
  entry cost: 29% of producer-years in the stationary count, roughly half of
  researcher-years on the life cycle.
* **The pass-through numbers change.** Share of a reading gain that survives
  entry:

  | Accounting | No founders | Founders (ω = ½) |
  |---|---|---|
  | Theorem 3 as stated (abstract) | 14% | 74% |
  | The paper's own member count | 53% | 85–88% |
  | The paper's own life-cycle simulation | 40% | 63% |

  Per hour saved, a production tool is then worth about 1.15 to 2.5 times a
  reading tool. The abstract says 1.3 to 7.2.
* **The economics of the missing term is worth stating.** Entry turns an
  entrant-side gain `h` into a longer catch-up, and incumbents' payoff rises by
  `(ε_Γ/|ε_w|)·h`, about `1.7h`.
* **The range in the abstract restates two inputs.** 74% is `θ_F` itself (55
  points, from an assumed `ω = ½`) plus 18. 14% is `E·D` with `E = 0.15`. No
  calibration moment informs `ω`. The paper says the model's density is the
  close-radius one, where the estimate of `E` is 0.035 (SE 0.118), yet uses
  the broad-radius 0.15.
* **The founders' channel is weaker on the life cycle.** A founder's seven
  years fall mostly in the growth phase, so the frontier receives 42% of a
  reading gain and 1% of a catch-up gain, against 74% and 41% in the formula.
  "Founders stay" is imposed: 36% of newborn topics attract no follower.
* **The ranking of tools is assumed.** Explorers' output scales with
  production speed and with nothing else, and fertility depletion is zero.
  With the depletion of Appendix B.1 the ranking of production over reading
  reverses once `η_f` exceeds about 0.06, and Section 10.2 itself invokes
  fertility depletion.
* **Positioning.** The theorem is the free-entry incidence result of Roback
  and Harris–Todaro applied to idea space. No paper was found that states it
  for research tools, so the application is new. The bibliography lacks
  Roback, Moretti, Kline–Moretti, Harris–Todaro, the pivot-penalty paper of
  Hill et al. (2025), Gans on AI and research direction, the streetlight
  effect of Hoelzemann et al., and *Research Waves* of Baccara, Levy and
  Razin.

### 4.3 The title result is correct mathematics that never operates

* No researcher-year in the calibrated economy is above 53% of the speed
  limit. Closing the densest mature topic needs production to triple.
* "Mature topics at 17% of the limit" is the identity `1 − e^{−premium/ν}`
  with `ν` fixed outside the calibration.
* A linear catch-up cost with no limit fits equally well (losses 13.6 and
  13.8).
* The limit exists because a newcomer must read the whole backlog at 38.6
  papers a year while incumbents keep producing on 7 to 13. Under the
  incumbents' own coverage no density is closed.
* Part (iii) holds for myopic entry only.
* The life-cycle law of motion has no catch-up delay: entrants add to the
  flow on arrival. The "too fast" block therefore omits the friction it is
  about.
* The take-off threshold exists only for a birth flow between 1.76 and 2.68
  (2 is assumed), and no calibrated topic stalls for slow growth. With
  forward-looking entry the threshold halves, from 2.2% to 1.0% a year.

### 4.4 The calibration is weaker than the text suggests

* Nine parameters, eight targets, one of them assumed. The reported point is
  not the optimum: an exact-catch-up vector with loss 13.8 exists against the
  reported 20.6. All of the "worse fit" attributed to exact catch-up is an
  unfinished search.
* The model does not reproduce the hump that motivates it. Returns are
  monotone across density fifths in every parameter vector found. Like for
  like, the model's regression peak is at +2.67 SD with 0.001% of
  researcher-years beyond it, against +1.0 SD and 12% in the data. The reason
  is structural: the cross-section traces the entry locus `F/Γ`, which rises
  with density.
* The 7% reading share is matched to producing incumbents only. Counting the
  model's own catch-up reading, reading is 15.8% of members' time.
* A mature topic has 1.9 members under the stated normalisation.
* The knife-edges (newborn margin 1.28%, `χ̄ = 0.11`, "flat subsidy about
  equal to the margin") are artefacts of the assumed birth flow and of where
  the search stopped. Only "a stock-proportional tax beats a flat subsidy"
  survives.

### 4.5 Evidence: two statements are wrong, the anchor needs repair, one large fact is missing

* **Wrong: "LLM adopters +7%".** It is the pre-trend. Relative to the last
  half-year before ChatGPT the post-period mean is −0.006.
* **Wrong: "the production shock so far is at most a few percent".** In
  January–September 2026 arXiv papers rose 30% and papers per author rose
  11–14% after seven flat years. The rise holds for incumbents, for rare
  surnames and for fixed cohorts. Between 39% and 49% of the volume growth is
  output per person.
* **Anchor (Fact 1).** The baseline elasticity is taken at the wrong radius by
  the paper's own rule; the pre-2023 estimate (0.15–0.30) is omitted; standard
  errors are clustered by author on a regressor that varies by area and year;
  and nothing in this repository reproduces Facts 1–4.
* **Missing: the 2026 surge is the first production shock in the data.**
  Fact 1 was estimated on years in which papers per author were flat, so it
  measures the response to more researchers. The model predicts that a rise
  in output per researcher is different: density and overlap should rise about
  one for one on impact. This prediction can fail, and the data are on disk.
* **First data on founders.** In HEP, 65–70% of a paper's later close
  neighbours share an author with it, but most of that is authors continuing
  work nobody else joined. Where outsiders follow, the founders write roughly
  35–40% of the later flow within two years. `θ_F` is measurable with data on
  disk and is probably not zero.
* The endorsement design measures who submits. The CS-review reduced form has
  a placebo half its size. The listing-rank null tests a channel that is not
  in the model. The submission cap has no model object.

## 5. Recommendation

*Provisional; to be completed when the design round returns.*

All five readers agree on four points.

1. Take the speed limit out of the title and the abstract. Keep the exact
   catch-up formula and the rent results of Proposition 5 and Corollary 1.
2. Lead with the extensive margin and the incidence of tools, and quantify
   with sufficient statistics (`E`, `D`, `θ_F`, the inframarginal share, time
   shares) instead of the nine-parameter calibration.
3. Treat the dynamic material (stock–flow lag, take-off, overshooting, speed
   limit) as a second paper that needs its own evidence.
4. Before any further theory, make three measurements with data already on
   disk: the founders' share in HEP, the response of local density to the
   2026 rise in papers per author, and a repaired, reproducible Fact 1.

The open choice is what leads: the incidence theorem, or the contrast between
growth through more researchers (which the data show opens new ground) and
growth through faster researchers (which the model says crowds existing
topics first, and which the 2026 surge can test).

## 6. Errors to fix whatever the paper becomes

1. Fact 5 and Section 8.5: report the event study relative to 2022H2 and drop
   the 7%.
2. Appendix B.8 and Fact 5: document the 2026 surge.
3. Fact 1: choose the baseline radius by the stated rule, report the pre-2023
   estimate, cluster by area, and add a version that runs from this
   repository.
4. Abstract: remove "identify the margins the model relies on" and the range
   1.3–7.2.
5. Lemma 4 and Theorem 3: one birth convention, with the inframarginal term;
   add a test that computes output from member counts.
6. Table 9: one assumption changed per column. The "formula" and "life cycle"
   columns currently differ in depletion and in how founders are valued.
7. Proposition 9: replace the two numerically verified steps by proofs (both
   are short); rename the bound in (iii).
8. Table 7: compare like with like (regression peak 2.67 against 1.0; share
   beyond it).
9. Calibration: finish the search on the fine grid; list the measurement-noise
   constant (0.3) as a parameter.
10. Stale text: "validate each other" (Fact 3) against Section 2.3; "faster
    drafting" in Section 8.5; "largely dissipated by entry" for the cap; the
    7–30% in "What we no longer claim"; cross-references to P7 and P3; the
    README. `ω` is a time share, not a probability.
11. Notation: thirteen symbols carry two or more meanings.

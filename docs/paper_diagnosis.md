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

The draft is four papers in one, and its two headline results are the two
that the evidence does not test. The parts of the theory are sound but the
centre is not: the step from the frontier value to output contradicts the
paper's own accounting, the headline range comes from an assumed parameter,
and the title result never operates in the paper's own numbers.

The material supports one coherent paper. Its subject is the entry margin.
Joining a topic has a sunk cost that grows with the topic's literature, so
newcomers are the people who adjust: more researchers open new topics, while
faster researchers first thicken the topics they are already in. The first
half is the paper's best existing fact. The second half is untested, and the
data to test it are on disk: papers per author rose 11–14% in 2026 after seven
flat years.

Four designs of the paper were scored by three editors, and all three ranked
this one first. Their advice, which I share, is to let two measurements choose
the headline before anything is rewritten (Section 5). The incidence of tools
becomes a short implication and a possible second paper. The speed limit and
the topic life cycles are shelved, and the title *Too Fast to Read* is
retired.

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
   people write more, every topic's flow rises at once, before anyone has
   moved (Appendix B.8, Theorem 2(iv)). The speed limit is the extreme case
   of this.

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
  measures the response to more researchers. The model says a rise in output
  per researcher is different: it raises the flow of existing topics first.
  This can fail, and the data are on disk. Two cautions. The rise is
  concentrated among prolific names: capping each name at three papers leaves
  about a third of it (0.031 of 0.089 log points in whole counts; common
  names inflate the tail). And 2026 is not a pure speed shock: the number of
  authors also rose by about 15%, and three arXiv rule changes fall in the
  same months.
* **First data on founders.** In HEP, 65–70% of a paper's later close
  neighbours share an author with it, but most of that is authors continuing
  work nobody else joined. Where outsiders follow, the founders write roughly
  35–40% of the later flow within two years. `θ_F` is measurable with data on
  disk and is probably not zero.
* The endorsement design measures who submits. The CS-review reduced form has
  a placebo half its size. The listing-rank null tests a channel that is not
  in the model. The submission cap has no model object.

## 5. Recommendation

### 5.1 The second round

Four designs were built from the diagnosis, each around one central claim, and
three editors (theory, empirical, a senior co-author deciding the next three
months) scored them. The ranking was unanimous.

| Design | One claim (1–10) | Relies only on what is solid | Evidence in hand | Editors' odds at a top-5 journal, best case |
|---|---|---|---|---|
| **Scale versus speed** (more researchers widen science, faster researchers thicken it first) | 8 | 8 | 5 | 4–10% |
| Scale versus speed plus "crowding costs about 1%" | 6–7 | 6 | 5 | 3–8% |
| Incidence of research tools | 6 | 6–7 | 3 | 2–6% |
| *Too Fast to Read* as a dynamic paper | 5–6 | 4–5 | 1–2 | 1–2% |

The five readers of the first round had put incidence first. The editors
disagreed, for a reason I accept. After the repair of Section 4.2 the
incidence headline is a band (between a third and nine tenths of a gain of
about two log points) set by inputs that are unmeasured, assumed or
unidentified, and no shock can test it before 2027. It is a framework. The
contrast between scale and speed is a claim that data on disk can reject.

### 5.2 The paper

Working title: *More Researchers or Faster Researchers? How Science Absorbs
Growth*. Adopt it only if the test in Section 5.4 supports it.

Thesis: newcomers are the margin on which science adjusts, because joining a
topic has an entry cost that incumbents have already paid. So more researchers
open new topics, while faster researchers first thicken the topics they are
already in.

Abstract as it could be submitted. Brackets mark what is not yet established.

> Science can grow through more researchers or through faster ones. The two
> are absorbed differently. Researchers choose topics, and joining a topic
> means catching up on its literature. Incumbents have paid that cost and earn
> a rent, so they stay; newcomers are the margin that adjusts. More
> researchers therefore open new topics, while faster researchers thicken the
> topics they are in until entry responds. On arXiv, growth in 2018–2025 came
> from new authors, and local density rose with an elasticity of [0.12–0.19],
> against [0.53–0.81] in a fixed idea space. In January–September 2026 papers
> per author rose 11–14% after seven flat years. [The extra papers landed in
> their authors' existing topics, which thickened; new authors again opened
> new ones.] The model predicts that entry into those topics falls as their
> literature builds up; we state this as a dated prediction. The welfare
> effects are small.

### 5.3 What the model predicts for 2026

The editors corrected two statements, one of them mine. I recomputed both.

* **How long the excess lasts.** Appendix B.8 says the spike decays "with
  half-life ln 2/ρ", about seven years. Under the paper's own law of motion
  entry stops and the flow thins at rate ρ, so a 12% excess is gone in
  ln(1.12)/ρ = 1.1 years at ρ = 0.10. The window in which speed and scale
  differ is short.
* **What happens to entry.** B.8 prices entry on the flow. Lemma 2 prices it
  on the stock, and the stock lags. With the stock unchanged, a 12% rise in
  the flow raises a newcomer's value by 0.27 log points. Once the stock has
  caught up, her value is 0.88 log points lower. The sign changes after about
  a year.

So for January–September 2026 the model as written does not say that entry
stops. Three things follow.

1. Testable now: where the extra papers of authors who sped up land, and
   whether the flow of existing topics loads more on incumbents' output than
   on growth in authors. For a shock that builds up over the year the editors
   compute a loading of 0.2–0.7 if entry stops and 0.9–1.0 if it does not,
   against about 0.15 for authors (their arithmetic; I have not re-run it).
2. A dated prediction, not a test this year: entry into the thickened topics
   falls in 2027 as their literature builds up.
3. Every payoff change behind these predictions is below one log point.
   Register signs and the adding-up of extra flow and lower entry, not
   magnitudes.

The prediction must be derived from one law of motion (with the stock and the
catch-up delay as states) and written down before any 2026 similarity measure
is computed. Otherwise either outcome can be read as support.

### 5.4 Two gates before any rewriting

**Gate 1. Does Fact 1 survive on a panel a referee can re-run?** Standard
errors clustered by area, every radius next to its benchmark, the pre-2023
sample, embedding neighbours, and a replica on arXiv categories that runs from
this repository, so that both elasticities come from one panel and one
measure. If the estimate is not below the benchmark, no version of the paper
stands.

**Gate 2. Does speed thicken?**

| Outcome | What to write |
|---|---|
| Extra papers stay in their authors' topics; those topics' flow loads clearly more on incumbents' output than on author growth; the embedding, text-free and style-equalised versions agree; 2024–25 placebo flat | Scale versus speed. State the fall in entry as a dated prediction and submit after a first 2027 reading. |
| Speed looks like scale: extra papers land outside their authors' topics and the loading is near 0.15 | The fallback: the extensive margin, rents, and the incidence of tools with a measured founders' share. The second fact is that a rise in papers per author did not thicken topics either. |
| The measures disagree, or the intervals cover both 0.15 and 0.5 | Do not lead with the contrast. Report the surge as a fact with the registered prediction, write the fallback, and return to the test with 2027 data. |

### 5.5 What goes where

| Material | Place |
|---|---|
| Proposition 3, Remark 4, Fact 1 (repaired) | Lead: scale |
| Appendix B.8 and Theorem 2(iv), rewritten as one proposition on the repaired law of motion; the 2026 surge | Lead: speed |
| Proposition 5, Corollary 1 (sign statements), Lemma 2 without the "speed limit" label | The mechanism: why incumbents stay |
| Lemma 5; the repaired output accounting; "easing entry raises density and incumbents' returns" as a sign statement | A short section on tools. No ranking, no 14%/74%, no 1.3–7.2 |
| Theorem 3 with the inframarginal term and a measured founders' share | The fallback, or a second paper |
| Theorem 1, the calibration, Propositions 4 and 10 as ranges | Appendix; one paragraph of magnitudes in the text |
| Speed limit as a headline, Propositions 9 and 11, life-cycle simulation | Shelved. Revisit only with evidence on catch-up |
| Detection block, Facts 3–5, listing-rank designs, submission cap, Propositions 6 and 8, eleven of the fifteen predictions | Out, or a data appendix |

### 5.6 What can be claimed today

1. Inside a topic a tool is worth the time it saves.
2. Incumbents' premium equals the newcomer's catch-up cost (theory).
3. Growth in 2018–2025 opened new topics. This holds qualitatively; the
   numbers are not yet reproducible from this repository and their standard
   errors are wrong.
4. Papers per author rose 11–14% in January–September 2026.
5. Every equilibrium welfare effect in the model is about 1% or less.

### 5.7 Risks that remain if everything goes well

* **"Extra papers stay home" does not show that the entry cost is reading.**
  Any topic-specific skill predicts it. The only cheap evidence that the cost
  is the literature is that newcomers' share of a topic's flow falls with the
  stock of older papers, holding recent flow fixed. Without it, say "a sunk
  entry cost" and cite the pivot-penalty literature.
* **The stakes inside the model are small.** The paper has to be about
  quantities (where AI-driven output lands, how fast science spreads), not
  about welfare.
* **The shock is nine months old and still building**, measured with text
  similarity in the year wording changed, and wording drift is correlated
  with who sped up. Text-free versions (home category; shared references in
  HEP) are essential.
* **Priority.** The 2026 fact will not stay unreported. Work to position
  against: Kusumegi et al. on language models and output, and work on AI
  narrowing the focus of science (Hao, Xu, Li and Evans; reference to be
  checked).
* **Odds.** The editors put the best case below one in ten at a top-5
  journal, and at 2–4% for the most likely result (extra papers stay home,
  entry unclear), with a strong field or second-tier general journal as the
  likely home. What moves the odds is a 2027 reading that shows the predicted
  adjustment. More theory does not.
* **Logistics.** Sentence embeddings cannot be run in the present environment
  (no `torch`, four cores), and the OpenAlex panel behind Facts 1–4 is not in
  this repository.

### 5.8 First month

1. Freeze the scope. No new theorems. Do not write an introduction.
2. Derive the short-run prediction for a shock that builds up over a year,
   from one law of motion with the stock and the catch-up delay as states.
   Deposit a dated memo with outcomes, samples, radii and thresholds.
3. Write up the 2026 surge as a fact from author names: by field and month,
   fixed cohort, capped at three, five and ten papers per name. This needs no
   text measure.
4. Embed abstracts for 2018 to October 2026; build flow density, the
   benchmark, a style floor, and topic cells fixed on 2022–24 text.
5. Gate 1. Stop if Fact 1 fails.
6. Gate 2, in the registered order. Report entry by authors already on arXiv
   whatever its sign.
7. In parallel, as the hedge: the founders' share in HEP, and newcomers' share
   against older stock given recent flow.
8. Decision meeting at week five. Only then the theory repairs the chosen
   paper needs, and the rewrite.

## 6. Errors to fix whatever the paper becomes

1. Fact 5 and Section 8.5: report the event study relative to 2022H2 and drop
   the 7%.
2. Appendix B.8 and Fact 5: document the 2026 surge. B.8's "half-life
   ln 2/ρ" is wrong under the paper's own law of motion (a shock of size α is
   absorbed in about ln(1+α)/ρ), and B.8 prices entry on the flow while
   Lemma 2 prices it on the stock.
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

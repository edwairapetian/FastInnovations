# What the data can estimate, and the model that fits it (11 November 2026)

This note answers three questions. What can be measured with the data we
have? Which identification strategies do those measurements support? Which
model has those measurable objects as its primitives?

The numbers come from five exploratory scripts in `empirics/explore/`
(results in `empirics/explore/results/*.json`). They are first cuts: fields
are arXiv categories, people are name keys, and the comparisons are matched,
not causal. They show what is measurable and roughly how large it is.

## 1. Bottom line

The data measure the **cost of entering a field** directly, at scale, without
any text similarity. That cost behaves as the entry-cost model says it
should: it rises with the distance of the move, falls with the growth of the
destination field, is unrelated to the destination's size, and disappears
when the newcomer has an insider as co-author.

So the model should be built around one object, the entry cost, with the
researcher's coverage of a field's literature as its state. Its central
equations then have data on both sides. What the data cannot reach (reading
time, the value of the frontier, welfare) should not carry any headline.

## 2. What we have

| Source | Content | Status |
|---|---|---|
| arXiv metadata | 2.2 million papers, 2015 to 15 October 2026: exact submission time, authors, submitter, categories, title, comments, journal reference, versions | on disk |
| arXiv text | titles and abstracts of the same papers | on disk |
| Semantic Scholar | reference and citation counts for 115,498 papers; reference lists for 15,103 | on disk; the API is reachable for more |
| INSPIRE (high-energy physics) | citation counts for 101,000 papers; reference lists for 61,500 | on disk; the API is reachable for more |
| OpenAlex panel behind Facts 1–4 | areas, neighbours, citations, PDFs | not in the repository; the API now needs a key |

Sentence embeddings cannot be run in this environment.

## 3. First cuts

The author panel has 346,472 authors with 5 to 300 papers and 6.0 million
author–paper pairs. An **entry** is an author's first paper in a primary
category she has never used, as primary or cross-list, after at least four
earlier papers. There are 411,995 entries from 2017 on. In 62% of them a
co-author has published in that category before (an insider).

**The output cost of entry.** Papers written in the year before and the two
years after the first paper in the new field, against non-entry papers of
authors with the same earlier output in the same year. Entries without an
insider co-author; destination fields with at least 200 papers a year.

| Entry | Net output, in years of the author's own output | n |
|---|---|---|
| Far move (new broad field) | −0.67 (SE 0.03) | 30,595 |
| Near move (new category, same broad field) | −0.10 (SE 0.02) | 54,047 |
| Solo-authored entry | −0.96 (SE 0.06) | 6,973 |
| With an insider co-author (all destinations) | +0.01 | 152,977 |

**By growth of the destination field** (papers, previous three years):

| | Slow (0% a year) | Middle (12%) | Fast (80%) |
|---|---|---|---|
| All entries | −0.55 (0.03) | −0.26 (0.03) | −0.04 (0.03) |
| Far moves | −0.78 (0.05) | −0.63 (0.05) | −0.58 (0.04) |
| Near moves | −0.41 (0.04) | −0.10 (0.04) | +0.16 (0.03) |

**By size of the destination field:** −0.25, −0.21 and −0.39 for small, medium
and large fields (460, 1,400 and 4,400 papers a year). No clear relation.

**Other measurements.**

| Object | First cut |
|---|---|
| Gap before the first paper in a new field | 77 days at the median against 48 before other papers; within author, +0.23 log points (SE 0.004) |
| Reference list of the first paper in a field, same author | 6.2% shorter (SE 0.9) than in fields where she has 11 or more papers; 3.0% shorter for the next two papers; the gap closes after three to five papers |
| Citations of the first paper in a field, same author | 0.14 log points lower (SE 0.014); 0.06 lower for the next two; zero after three to five |
| Persistence | 59% of entries without an insider are never followed by another paper in that field; 26% are still active there after three years (38% with an insider) |
| Tenure in a home field | 89% still publish there five years after their first paper: an exit rate of 2.3% a year at the level of arXiv categories |
| Pedagogical papers | about 5,250 reviews or surveys, 320 tutorials and 180 sets of lecture notes a year |
| The 2026 rise in papers per author, fixed cohort | mathematics +0.12 log points, general relativity +0.10, high-energy physics +0.06, computer science +0.05, electrical engineering −0.03 |
| Does earlier use of language-model vocabulary predict the 2026 rise? | No: 0.02 (SE 0.02) |

One cut goes the other way. Within a field over time, the raw gap before an
entry is *longer* when the field's stock-to-flow ratio is lower (−0.17, SE
0.06). The gap is the cruder measure, and the variation within fields is
small (a standard deviation of 0.11), but this has to be resolved with finer
topics.

## 4. What the first cuts say about the model

1. **There is an entry cost, and it is modest.** Between a tenth of a year
   and a year of output, depending on the move. The current calibration has
   0.74 years for every newcomer.
2. **Newcomers do not catch up first and then produce.** Their first papers
   in a field have shorter reference lists and fewer citations, and both
   converge within three to five papers. Entry is gradual. A model in which a
   newcomer reads everything before writing anything is rejected by this.
3. **Growth, not size, makes a field cheap to enter.** This is what a cost
   that depends on the ratio of a literature's stock to its flow predicts,
   and it is the mechanism behind "fast-growing topics overshoot". A cost
   that rises with the size of the literature is not supported.
4. **Most entry happens through insiders, at no output cost.** A model with
   solo researchers misses the main channel. It also suggests where a real
   limit on entry may lie: in the time of insiders, not in reading speed.
5. **Most entries do not stick.** The model's "join and stay" describes a
   quarter of entries without an insider.
6. **The 2026 speed-up is broad, strongest in mathematics and theoretical
   physics, and not predicted by earlier adoption of language-model
   vocabulary.** There is variation across fields for a test of scale against
   speed, but no instrument at the level of the author.

## 5. Estimable objects and model primitives

| Model object | Data counterpart | Where |
|---|---|---|
| Entry cost, in years of output | output around the first paper in a new field, against matched non-movers | all fields, on disk |
| How fast a newcomer converges | reference lists and citations by number of papers in the field, same author | sample on disk; full records can be fetched |
| Initial coverage | distance of the move; overlap between the old and new fields' literatures | categories now; text or references later |
| Stock and flow of a literature | papers by field and year; growth | on disk |
| Exit rate from a field | length of an author's activity in a field | on disk |
| Rate at which a literature loses relevance | age of the references that papers cite, by field | INSPIRE and Semantic Scholar, to fetch |
| Insiders' premium | returns of the same author as insider and as newcomer | citations to fetch |
| Scale and speed | authors, and papers per author, by field and time | on disk |
| Volume elasticity of local density | Fact 1 | needs embeddings or the original panel |
| Reading time | none; reference lists are a proxy for coverage | — |
| Value of the frontier; founders' share of value | none; only shares of papers | — |

## 6. Identification strategies, in order of promise

**1. Movers, with dose–response.** The same author before and after her
first paper in a new field. The cost should rise with distance, fall with
the destination's growth and with an insider co-author. *Threat:* people move
when their old line of work dries up, so the dip may be a cause of the move
and not its cost. *Answers:* compare movers who leave the same field in the
same year for slow and for fast destinations, so that whatever pushed them is
common; use entries pulled by a dated event in the destination; test for a
dip two years before; use displacement from fields that shrink.

**2. Shifters of the entry cost.** The 2025 rule that took reviews out of
computer science is a shock to exactly this cost: compare the entry cost and
the newcomer's penalty in categories that relied on reviews, before and
after. More generally, the arrival of a review or of lecture notes in a topic
is an event with thousands of instances; lecture notes from schools whose
topics are fixed in advance are the cleanest.

**3. Shocks to a field's flow.** A discovery raises a field's flow before its
stock. The model predicts that entry is cheap at first and becomes costly as
the literature builds. Dated discoveries in physics give events; the growth
gradient in Section 3 is the cross-sectional version.

**4. Scale against speed in 2026.** Variation across fields exists
(mathematics against computer science). At the level of topic cells the
design is the one in `docs/paper_diagnosis.md`. It needs embeddings.

**5. A shock to the number of entrants.** The 2026 endorsement rule cut
first-time authors. It tests "more researchers open new ground" in reverse.

**6. Durability of knowledge across fields.** Where a literature loses
relevance slowly, the entry cost and insiders' premium should be larger. This
needs reference ages by field.

## 7. The model that serves these designs

* **State.** A researcher's coverage of a field's literature. It rises with
  what she reads and writes there and falls as the literature turns over.
* **Entry.** A newcomer starts with the coverage her old field gives her
  (distance), or with her co-author's (an insider). She writes from the
  start, at lower productivity, and converges.
* **Entry cost.** The output and returns she gives up while converging. It
  depends on distance, on the field's stock relative to its flow, and on
  access to insiders.
* **Free entry.** Insiders' premium equals the exit rate times the entry
  cost. Both sides are measurable.
* **Aggregate results** follow as before: more researchers open new ground;
  faster researchers thicken existing fields first; a tool's gain stays with
  those who are not on the entry margin.

Three changes to the proposal of the previous round follow from the data.
Catch-up is gradual, not a period of full-time reading. Distance and insiders
enter the entry cost. And the quantification should use the measured entry
cost as its sufficient statistic, not a fitted reading speed.

## 8. What to drop

The detection block, the speed limit as a result, readership units, and any
number that needs reading time, the value of the frontier or the founders'
share of value.

## 9. Next steps

1. Rebuild the mover design with finer topics, a pre-trend test, and origin
   field by year effects.
2. Fetch citations and reference lists for a sample of movers' full records.
3. Estimate the effect of the computer-science review rule on the entry cost.
4. Write the model of Section 7 and state its restrictions before the next
   estimates are seen.

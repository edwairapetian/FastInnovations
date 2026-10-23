"""Author-level LLM adoption and output (the drafting channel).

Adoption is measured from text, as in the excess-vocabulary literature: the
share of a submitter's abstracts that use at least one word whose frequency
jumped after ChatGPT (delve, showcase, underscore, intricate, meticulous,
pivotal, realm, notably, ...). Identity is the submitter's full-name key.

Sample: submitters with at least one paper in 2021-2022 and at least one in
2023. Adopters use a marker word in at least half of their 2023 abstracts;
non-adopters in none. Classifying on 2023 papers keeps the outcome window
(2024H1-2026H1) free of the classification condition. Outcome: papers
submitted per half-year, 2019H1-2026H1; 2023 is the classification period.
Event study relative to 2022H2 with submitter and half-year x field fixed
effects; standard errors clustered by submitter. Estimated for all fields and
for fields where AI is not the subject (excluding cs, stat, eess).
Writes empirics/results/llm_authors.json and a figure.
"""
import json
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, "empirics")
from names import submitter_key

MARK = re.compile(r"\b(?:delv(?:e|es|ed|ing)|showcas(?:e|es|ed|ing)|underscor(?:e|es|ed|ing)|"
                  r"intricate|intricacies|meticulous(?:ly)?|pivotal|realms?|notably|"
                  r"commendabl[ey]|garner(?:s|ed|ing)?|multifaceted|seamless(?:ly)?|"
                  r"invaluable|groundbreaking|tapestry|elucidat(?:e|es|ed|ing))\b", re.I)
M = pd.read_parquet("data/arxiv_meta.parquet", columns=["id", "v1", "submitter", "primary"])
T = pd.read_parquet("data/arxiv_text.parquet", columns=["id", "abstract"])
d = M[M.v1 >= "2019-01-01"].merge(T, on="id")
d["mark"] = d.abstract.str.contains(MARK)
d["skey"] = d.submitter.map(submitter_key)
d["y"] = d.v1.dt.year
d["half"] = d.y.astype(str) + "H" + np.where(d.v1.dt.month <= 6, "1", "2")
d["field"] = d.primary.str.split(".").str[0]
res = {"marker_share_by_year": d.groupby("y").mark.mean().round(4).to_dict()}

pre = d[d.y.isin([2021, 2022])].groupby("skey").size()
y24 = d[d.y == 2023].groupby("skey").agg(n=("id", "size"), m=("mark", "mean"))
elig = y24[(y24.n >= 1)].index.intersection(pre.index)
ad = y24.loc[elig]
group = pd.Series(np.select([ad.m >= 0.5, ad.m == 0], [1, 0], -1), index=ad.index)
group = group[group >= 0]
# modal field of the submitter in 2021-2022
mf = d[d.y.isin([2021, 2022]) & d.skey.isin(group.index)].groupby("skey").field.agg(
    lambda s: s.value_counts().index[0])
halves = sorted(d.half.unique())
halves = [h for h in halves if h <= "2026H1"]
panel = (d[d.skey.isin(group.index) & d.half.isin(halves)]
         .groupby(["skey", "half"]).size().unstack(fill_value=0)
         .reindex(columns=halves, fill_value=0).stack().rename("n").reset_index())
panel["adopt"] = panel.skey.map(group)
panel["field"] = panel.skey.map(mf)
res["n_adopters"] = int((group == 1).sum())
res["n_nonadopters"] = int((group == 0).sum())


def es(df, label):
    df = df.copy()
    df["ln"] = np.log1p(df.n)
    ref = "2022H2"
    rel = [h for h in halves if h != ref]
    X = pd.DataFrame({f"a_{h}": ((df.half == h) & (df.adopt == 1)).astype(float) for h in rel})
    df["fh"] = df.field + "_" + df.half
    Z = pd.concat([df[["ln"]], X], axis=1)
    for _ in range(4):
        Z = Z - Z.groupby(df.skey).transform("mean")
        Z = Z - Z.groupby(df.fh).transform("mean")
    f = sm.OLS(Z.ln, Z.drop(columns="ln")).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(df.skey)[0]})
    co = {h: [float(f.params[f"a_{h}"]), float(f.bse[f"a_{h}"])] for h in rel}
    co[ref] = [0.0, 0.0]
    post = df.half >= "2023H1"
    df["D"] = (post & (df.adopt == 1)).astype(float)
    Z2 = df[["ln", "D"]].copy()
    for _ in range(4):
        Z2 = Z2 - Z2.groupby(df.skey).transform("mean")
        Z2 = Z2 - Z2.groupby(df.fh).transform("mean")
    f2 = sm.OLS(Z2.ln, Z2[["D"]]).fit(cov_type="cluster",
                                       cov_kwds={"groups": pd.factorize(df.skey)[0]})
    # post-classification window (2024H1 on) vs the classification year 2023
    late = df.half >= "2024H1"
    df["D_late"] = (late & (df.adopt == 1)).astype(float)
    df["D_mid"] = ((df.half >= "2023H1") & ~late & (df.adopt == 1)).astype(float)
    Z3 = df[["ln", "D_mid", "D_late"]].copy()
    for _ in range(4):
        Z3 = Z3 - Z3.groupby(df.skey).transform("mean")
        Z3 = Z3 - Z3.groupby(df.fh).transform("mean")
    f3 = sm.OLS(Z3.ln, Z3[["D_mid", "D_late"]]).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(df.skey)[0]})
    pre_tr = [co[h][0] for h in halves if h < "2022H2"]
    return {"label": label, "n_submitters": int(df.skey.nunique()),
            "did_post2023": [float(f2.params["D"]), float(f2.bse["D"])],
            "did_2023_classif": [float(f3.params["D_mid"]), float(f3.bse["D_mid"])],
            "did_2024_26": [float(f3.params["D_late"]), float(f3.bse["D_late"])],
            "mean_papers_pre_adopters": float(df[(df.adopt == 1) & (df.half < "2023H1")].n.mean()),
            "mean_papers_pre_non": float(df[(df.adopt == 0) & (df.half < "2023H1")].n.mean()),
            "event_study": co, "max_abs_pre": float(np.max(np.abs(pre_tr)))}


res["all_fields"] = es(panel, "all fields")
nonai = panel[~panel.field.isin(["cs", "stat", "eess"])]
res["non_ai_fields"] = es(nonai, "excluding cs, stat, eess")
json.dump(res, open("empirics/results/llm_authors.json", "w"), indent=1)
for k in ("all_fields", "non_ai_fields"):
    r = res[k]
    print(k, r["n_submitters"], "post:", r["did_post2023"], "2023:", r["did_2023_classif"],
          "24-26:", r["did_2024_26"], "max|pre|", round(r["max_abs_pre"], 3),
          r["mean_papers_pre_adopters"], r["mean_papers_pre_non"])
print(res["marker_share_by_year"], res["n_adopters"], res["n_nonadopters"])
fig, ax = plt.subplots(figsize=(6.5, 3.2))
for k, col, off in (("all_fields", "#2166ac", -0.1), ("non_ai_fields", "#b2182b", 0.1)):
    co = res[k]["event_study"]
    xs = np.arange(len(halves))
    b = np.array([co[h][0] for h in halves])
    s = np.array([co[h][1] for h in halves])
    ax.errorbar(xs + off, b, yerr=1.96 * s, fmt="o", ms=3, color=col, lw=0.8,
                label=res[k]["label"])
ax.axhline(0, color="k", lw=0.5)
ax.axvline(halves.index("2022H2") + 0.5, color="k", ls=":", lw=0.8)
ax.set_xticks(np.arange(len(halves))[::2])
ax.set_xticklabels(halves[::2], fontsize=7)
ax.set_ylabel("log(1 + papers), adopters\nrelative to non-adopters")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("empirics/figures/llm_authors.pdf")

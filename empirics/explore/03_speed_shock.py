"""Exploratory: (a) what predicts which authors sped up in 2026, measured before 2026;
(b) how reference lists grow with experience in a field (a profile of catching up).
Uses the cached author panel of 02_entry_dynamics.py.
"""
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "results")
res = {}
MARK = re.compile(r"\b(?:delv(?:e|es|ed|ing)|showcas(?:e|es|ed|ing)|underscor(?:e|es|ed|ing)|"
                  r"intricate|intricacies|meticulous(?:ly)?|pivotal|realms?|notably|"
                  r"commendabl[ey]|garner(?:s|ed|ing)?|multifaceted|seamless(?:ly)?|"
                  r"invaluable|groundbreaking|tapestry|elucidat(?:e|es|ed|ing))\b", re.I)

ap = pd.read_parquet(os.path.join(ROOT, "data", "explore_author_panel.parquet"))
meta = pd.read_parquet(os.path.join(ROOT, "data", "arxiv_meta.parquet"), columns=["id", "v1", "n_authors"])
meta["pid"] = np.arange(len(meta))
meta["month"] = meta.v1.dt.month
ap = ap.merge(meta[["pid", "id", "month"]], on="pid")

# ---------------------------------------------------------------- (a) who sped up in 2026?
txt = pd.read_parquet(os.path.join(ROOT, "data", "arxiv_text.parquet"), columns=["id", "abstract"])
recent = meta[meta.v1.dt.year.between(2023, 2025)][["id"]].merge(txt, on="id")
recent["mark"] = recent.abstract.str.contains(MARK)
del txt
recent["mark"] = recent.mark.astype(float)
ap = ap.merge(recent[["id", "mark"]], on="id", how="left")
js = ap[ap.month <= 9]
cnt = js.groupby(["key", "year"]).size().unstack(fill_value=0)
for y in range(2019, 2027):
    if y not in cnt.columns:
        cnt[y] = 0
A = pd.DataFrame(index=cnt.index)
for y in (2022, 2023, 2024, 2025, 2026):
    A[f"n{y}"] = cnt[y]
coh = A[(A.n2022 > 0) & (A.n2023 > 0) & (A.n2024 > 0)].copy()
pre = ap[ap.year.between(2023, 2025)]
coh["mark_share"] = pre.groupby("key").mark.mean()
coh["solo_share"] = (pre.n_authors == 1).groupby(pre.key).mean()
coh["group"] = pre.groupby("key").group.agg(lambda s: s.mode().iloc[0])
coh = coh.dropna(subset=["mark_share", "group"])
coh["mark_share"] = coh.mark_share.astype(float)
coh["solo_share"] = coh.solo_share.astype(float)
coh["ln_pre"] = np.log(coh.n2022 + coh.n2023 + coh.n2024)
coh["d26"] = np.log1p(coh.n2026) - np.log1p(coh.n2025)
coh["d25"] = np.log1p(coh.n2025) - np.log1p(coh.n2024)
res["cohort_n"] = int(len(coh))
res["mean_d26"] = float(coh.d26.mean())
res["mean_d25"] = float(coh.d25.mean())
res["mark_share_mean"] = float(coh.mark_share.mean())
res["share_with_any_mark"] = float((coh.mark_share > 0).mean())


def reg(dep):
    X = pd.DataFrame({"mark_share": coh.mark_share, "solo_share": coh.solo_share, "ln_pre": coh.ln_pre}).astype(float)
    y = coh[dep] - coh.groupby("group")[dep].transform("mean")
    X = X - X.groupby(coh.group.values).transform("mean")
    b, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
    r = y.values - X.values @ b
    XtX = np.linalg.inv(X.values.T @ X.values)
    se = np.sqrt(np.diag(XtX @ ((X.values * r[:, None]).T @ (X.values * r[:, None])) @ XtX))
    return {c: [float(bb), float(ss)] for c, bb, ss in zip(X.columns, b, se)}


res["speedup_2026_on_predetermined"] = reg("d26")
res["placebo_2025"] = reg("d25")
print("cohort", len(coh), "mean d26 %.3f d25 %.3f" % (coh.d26.mean(), coh.d25.mean()))
print("2026:", res["speedup_2026_on_predetermined"])
print("2025 placebo:", res["placebo_2025"])
q = pd.cut(coh.mark_share, [-0.01, 0, 0.1, 0.25, 0.5, 1.0])
res["d26_by_mark_share"] = {str(k): [float(v.d26.mean()), float(v.d25.mean()), int(len(v))] for k, v in coh.groupby(q, observed=True)}
print("by marker share (d26, d25, n):", res["d26_by_mark_share"])
res["d26_by_group"] = {k: [float(v.d26.mean()), float(v.d25.mean()), int(len(v))] for k, v in coh.groupby("group") if len(v) > 800}
print("by field (d26, d25, n):", {k: [round(a, 3), round(b_, 3), n] for k, (a, b_, n) in res["d26_by_group"].items()})

# ---------------------------------------------------------------- (b) references and experience in the field
s2 = []
with open(os.path.join(ROOT, "data_raw", "s2_focal.jsonl")) as f:
    for line in f:
        d_ = json.loads(line)
        rec = d_.get("rec") or {}
        if rec.get("referenceCount"):
            s2.append((d_["arxiv"], rec["referenceCount"], rec.get("citationCount") or 0))
s2 = pd.DataFrame(s2, columns=["id", "refs", "cites"])
pp = ap.groupby("id").agg(max_prior=("prior_in_cat", "max"), min_prior=("prior_in_cat", "min"), primary=("primary", "first"),
                          year=("year", "first"), n_authors=("n_authors", "first"), kmin=("k", "min")).reset_index()
s2 = s2.merge(pp, on="id")
s2 = s2[s2.kmin >= 1]
s2["cellfe"] = s2.primary + "_" + s2.year.astype(str)
s2["lrefs"] = np.log(s2.refs)
s2["lcit"] = np.log1p(s2.cites)
bins = [-1, 0, 2, 5, 10, 1000]
labs = ["0 (all new)", "1-2", "3-5", "6-10", "11+"]
s2["exp"] = pd.cut(s2.max_prior, bins, labels=labs)
out = {}
for dep in ("lrefs", "lcit"):
    y = s2[dep] - s2.groupby("cellfe")[dep].transform("mean")
    D = pd.get_dummies(s2.exp).astype(float)[labs[:-1]]
    D["ln_team"] = np.log(s2.n_authors)
    D = D - D.groupby(s2.cellfe.values).transform("mean")
    b, *_ = np.linalg.lstsq(D.values, y.values, rcond=None)
    r = y.values - D.values @ b
    se = np.sqrt(np.diag(np.linalg.inv(D.values.T @ D.values) * (r @ r) / (len(r) - D.shape[1])))
    out[dep] = {c: [float(bb), float(ss)] for c, bb, ss in zip(D.columns, b, se)}
    print(dep, "relative to teams with 11+ prior papers in the field:", {k: [round(v[0], 3), round(v[1], 3)] for k, v in out[dep].items()})
res["experience_profile"] = dict(n=int(len(s2)), **out)
json.dump(res, open(os.path.join(OUT, "speed_shock.json"), "w"), indent=1)

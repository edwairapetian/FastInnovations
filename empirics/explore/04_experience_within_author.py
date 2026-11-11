"""Exploratory: within-author profile of reference lists and citations by experience in the field.

For authors with several papers in the Semantic Scholar sample, compare the same author's papers
in fields where she has little experience with her papers in fields where she has more, holding
fixed her total experience and the field-by-year cell.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "results")
ap = pd.read_parquet(os.path.join(ROOT, "data", "explore_author_panel.parquet"))
meta = pd.read_parquet(os.path.join(ROOT, "data", "arxiv_meta.parquet"), columns=["id"])
meta["pid"] = np.arange(len(meta))
ap = ap.merge(meta, on="pid")
s2 = []
with open(os.path.join(ROOT, "data_raw", "s2_focal.jsonl")) as f:
    for line in f:
        d_ = json.loads(line)
        rec = d_.get("rec") or {}
        if rec.get("referenceCount"):
            s2.append((d_["arxiv"], rec["referenceCount"], rec.get("citationCount") or 0))
s2 = pd.DataFrame(s2, columns=["id", "refs", "cites"])
d = ap.merge(s2, on="id")
d = d[d.groupby("key").id.transform("size") >= 3].copy()
d["lrefs"] = np.log(d.refs)
d["lcit"] = np.log1p(d.cites)
d["cellfe"] = d.primary + "_" + d.year.astype(str)
labs = ["0", "1-2", "3-5", "6-10"]
d["exp"] = pd.cut(d.prior_in_cat, [-1, 0, 2, 5, 10, 10000], labels=labs + ["11+"])
D = pd.get_dummies(d.exp).astype(float)[labs]
D["ln_team"] = np.log(d.n_authors)
D["ln_career"] = np.log1p(d.k)
res = dict(n_author_papers=int(len(d)), n_authors=int(d.key.nunique()), share_new_to_field=float((d.prior_in_cat == 0).mean()))
for dep in ("lrefs", "lcit"):
    y = d[dep].copy()
    X = D.copy()
    for _ in range(6):
        for g in (d.key, d.cellfe):
            y = y - y.groupby(g.values).transform("mean")
            X = X - X.groupby(g.values).transform("mean")
    b, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
    r = y.values - X.values @ b
    XtX = np.linalg.inv(X.values.T @ X.values)
    sc = pd.DataFrame(X.values * r[:, None]).groupby(d.key.values).sum().values
    se = np.sqrt(np.diag(XtX @ (sc.T @ sc) @ XtX))
    res[dep] = {c: [float(bb), float(ss)] for c, bb, ss in zip(X.columns, b, se)}
    print(dep, {k: [round(v[0], 3), round(v[1], 3)] for k, v in res[dep].items()})
print(res["n_author_papers"], res["n_authors"], round(res["share_new_to_field"], 3))
json.dump(res, open(os.path.join(OUT, "experience_within_author.json"), "w"), indent=1)

"""Exploratory: how long do researchers take to enter a new field?

For every author (name key with a full first name) we order her arXiv papers by the exact
submission time and measure the gap, in days, since her previous paper.  An *entry* is her
first paper whose primary category she has never used before, as primary or as a cross-list,
after at least four earlier papers.  The excess of the gap before an entry over her gaps
before other papers is a direct, text-free measure of the time it takes to get into a field.

Outputs a small JSON with the numbers quoted in docs/empirical_design.md.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "empirics"))
from names import author_keys  # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

df = pd.read_parquet(os.path.join(ROOT, "data", "arxiv_meta.parquet"),
                     columns=["authors", "n_authors", "primary", "categories", "v1"])
df = df[df.n_authors <= 30].reset_index(drop=True)
df["pid"] = np.arange(len(df))
df["t"] = (df.v1 - pd.Timestamp("2015-01-01", tz="UTC")).dt.total_seconds() / 86400.0
df["year"] = df.v1.dt.year
grp = df.primary.str.split(".").str[0]
BROAD = {"hep-th": "hep", "hep-ph": "hep", "hep-ex": "hep", "hep-lat": "hep", "nucl-th": "nucl", "nucl-ex": "nucl",
         "stat": "stat", "eess": "eess"}
df["group"] = grp.map(lambda g: BROAD.get(g, g))
print("papers", len(df), "categories", df.primary.nunique(), "groups", df.group.nunique(), flush=True)

keys = [author_keys(a, full=True) for a in df.authors]
ap = pd.DataFrame({"pid": np.repeat(df.pid.values, [len(k) for k in keys]),
                   "key": [k for ks in keys for k in ks]})
del keys
ap = ap[ap.key.str.contains(" ") & (ap.key.str.split().str[0].str.len() > 1)]
ap = ap.drop_duplicates(["pid", "key"])
ap = ap.merge(df[["pid", "t", "year", "primary", "group", "n_authors"]], on="pid")
npap = ap.groupby("key").pid.transform("size")
ap = ap[(npap >= 5) & (npap <= 300)].sort_values(["key", "t"]).reset_index(drop=True)
print("author-paper rows", len(ap), "authors", ap.key.nunique(), flush=True)

# order of the paper in the author's career, gap since her previous paper
ap["k"] = ap.groupby("key").cumcount()
ap["gap"] = ap.t - ap.groupby("key").t.shift(1)
# has she used this primary category before, as primary or as a cross-list?
cats = df[["pid", "categories"]].copy()
cats["cat"] = cats.categories.str.split()
cats = cats.explode("cat")[["pid", "cat"]]
ac = ap[["key", "pid", "t"]].merge(cats, on="pid")
first_any = ac.groupby(["key", "cat"]).t.min().rename("t_first_any").reset_index()
ap = ap.merge(first_any, left_on=["key", "primary"], right_on=["key", "cat"], how="left").drop(columns="cat")
ap["new_cat"] = (ap.t <= ap.t_first_any + 1e-9)
first_grp = ap.groupby(["key", "group"]).t.transform("min")
ap["new_group"] = ap.t <= first_grp + 1e-9
# number of her earlier papers with this primary category (for the incumbent flag)
ap["prior_in_cat"] = ap.groupby(["key", "primary"]).cumcount()
inc = ap.groupby("pid").prior_in_cat.max().rename("max_prior")
ap = ap.merge(inc, on="pid")
ap["inc_coauthor"] = ap.max_prior > 0          # some author of the paper has been here before

ev = ap[(ap.k >= 4) & ap.gap.notna() & (ap.year >= 2017)].copy()
ev["entry"] = ev.new_cat
# her usual gap: median of her gaps before non-entry papers
base = ev[~ev.entry].groupby("key").gap.median().rename("base_gap")
ev = ev.merge(base, on="key", how="inner")
ev["lgap"] = np.log1p(ev.gap)

res = {}
res["n_authors"] = int(ev.key.nunique())
res["n_author_papers"] = int(len(ev))
res["n_entries"] = int(ev.entry.sum())
res["n_entries_new_group"] = int((ev.entry & ev.new_group).sum())


def summarise(mask, label):
    e = ev[mask]
    o = ev[~ev.entry]
    out = dict(n=int(len(e)), median_gap=float(e.gap.median()), mean_gap=float(e.gap.mean()),
               median_ratio_to_base=float((e.gap / e.base_gap.clip(lower=1)).median()),
               mean_log_excess=float((e.lgap - np.log1p(e.base_gap)).mean()),
               other_median_gap=float(o.gap.median()), other_mean_gap=float(o.gap.mean()))
    res[label] = out
    print(label, out, flush=True)


summarise(ev.entry, "all_entries")
summarise(ev.entry & ~ev.inc_coauthor, "entries_no_incumbent_coauthor")
summarise(ev.entry & ev.inc_coauthor, "entries_with_incumbent_coauthor")
summarise(ev.entry & ev.new_group & ~ev.inc_coauthor, "entries_new_group_no_incumbent")
summarise(ev.entry & ~ev.new_group & ~ev.inc_coauthor, "entries_same_group_no_incumbent")
summarise(ev.entry & (ev.n_authors == 1), "entries_solo")
res["share_entries_with_incumbent_coauthor"] = float(ev[ev.entry].inc_coauthor.mean())

# within-author regression: log gap on entry dummies, author effects, year effects
ev["e_noinc"] = (ev.entry & ~ev.inc_coauthor).astype(float)
ev["e_inc"] = (ev.entry & ev.inc_coauthor).astype(float)
X = ev[["e_noinc", "e_inc"]].copy()
X["ln_team"] = np.log(ev.n_authors)
y = ev.lgap.copy()
gk = ev.key
for _ in range(3):                       # alternate demeaning by author and by year
    y = y - y.groupby(gk).transform("mean")
    X = X - X.groupby(gk).transform("mean")
    y = y - y.groupby(ev.year).transform("mean")
    X = X - X.groupby(ev.year).transform("mean")
beta, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
resid = y.values - X.values @ beta
# cluster-robust standard errors by author
XtX_inv = np.linalg.inv(X.values.T @ X.values)
score = pd.DataFrame(X.values * resid[:, None]).groupby(gk.values).sum().values
V = XtX_inv @ (score.T @ score) @ XtX_inv
se = np.sqrt(np.diag(V))
res["within_author_log_gap"] = {c: [float(b), float(s)] for c, b, s in zip(X.columns, beta, se)}
print("within-author log gap:", res["within_author_log_gap"], flush=True)

# destination: flow, stock (decay 0.25 a year, from 2015) and growth, by category and year
cy = df.groupby(["primary", "year"]).size().rename("P").reset_index()
cy = cy.sort_values(["primary", "year"])
st = []
for c, g in cy.groupby("primary"):
    S = 0.0
    for yv, pv in zip(g.year.values, g.P.values):
        S = S * np.exp(-0.25) + pv
        st.append((c, yv, S))
st = pd.DataFrame(st, columns=["primary", "year", "L"])
cy = cy.merge(st, on=["primary", "year"])
cy["P_lag"] = cy.groupby("primary").P.shift(1)
cy["L_lag"] = cy.groupby("primary").L.shift(1)
cy["g3"] = np.log(cy.P_lag / cy.groupby("primary").P.shift(4)) / 3.0
en = ev[ev.entry & ~ev.inc_coauthor & (ev.year >= 2020) & (ev.year <= 2025)].merge(
    cy[["primary", "year", "P_lag", "L_lag", "g3"]], on=["primary", "year"], how="inner").dropna()
en = en[en.P_lag >= 200]
en["excess"] = en.gap - en.base_gap
en["ln_SF"] = np.log(en.L_lag / en.P_lag)
en["ln_P"] = np.log(en.P_lag)
res["n_entries_for_destination_regression"] = int(len(en))
Z = np.column_stack([np.ones(len(en)), en.ln_SF, en.ln_P, np.log(en.n_authors), np.log1p(en.base_gap)])
for dep in ("lgap", "excess"):
    b, *_ = np.linalg.lstsq(Z, en[dep].values, rcond=None)
    r = en[dep].values - Z @ b
    ZtZ = np.linalg.inv(Z.T @ Z)
    sc = pd.DataFrame(Z * r[:, None]).groupby(en.primary.values).sum().values   # cluster by destination
    se = np.sqrt(np.diag(ZtZ @ (sc.T @ sc) @ ZtZ))
    res[f"destination_{dep}"] = dict(ln_stock_to_flow=[float(b[1]), float(se[1])], ln_flow=[float(b[2]), float(se[2])],
                                     n_clusters=int(en.primary.nunique()))
    print("destination", dep, res[f"destination_{dep}"], flush=True)
q = pd.qcut(en.ln_SF, 4, labels=False)
res["median_gap_by_stock_to_flow_quartile"] = [float(en.gap[q == i].median()) for i in range(4)]
res["median_excess_by_stock_to_flow_quartile"] = [float(en.excess[q == i].median()) for i in range(4)]
res["stock_to_flow_quartile_means"] = [float(np.exp(en.ln_SF[q == i].mean())) for i in range(4)]
print("by stock-to-flow quartile:", res["stock_to_flow_quartile_means"], res["median_gap_by_stock_to_flow_quartile"],
      res["median_excess_by_stock_to_flow_quartile"], flush=True)

json.dump(res, open(os.path.join(OUT, "entry_gaps.json"), "w"), indent=1)

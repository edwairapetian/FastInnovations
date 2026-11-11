"""Exploratory: output around entry into a new field, persistence after entry, the role of the
destination's stock-to-flow ratio within fields, pedagogical papers, and references as a
proxy for reading.  Builds on the author panel of 01_entry_gaps.py (rebuilt here and cached).
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "empirics"))
from names import author_keys  # noqa: E402

OUT = os.path.join(HERE, "results")
CACHE = os.path.join(ROOT, "data", "explore_author_panel.parquet")
res = {}

df = pd.read_parquet(os.path.join(ROOT, "data", "arxiv_meta.parquet"),
                     columns=["id", "authors", "n_authors", "primary", "categories", "title", "comments", "v1"])
df["pid"] = np.arange(len(df))
df["t"] = (df.v1 - pd.Timestamp("2015-01-01", tz="UTC")).dt.total_seconds() / 86400.0
df["year"] = df.v1.dt.year
BROAD = {"hep-th": "hep", "hep-ph": "hep", "hep-ex": "hep", "hep-lat": "hep", "nucl-th": "nucl", "nucl-ex": "nucl"}
df["group"] = df.primary.str.split(".").str[0].map(lambda g: BROAD.get(g, g))
TMAX = float(df.t.max())

if os.path.exists(CACHE):
    ap = pd.read_parquet(CACHE)
else:
    d = df[df.n_authors <= 30]
    keys = [author_keys(a, full=True) for a in d.authors]
    ap = pd.DataFrame({"pid": np.repeat(d.pid.values, [len(k) for k in keys]), "key": [k for ks in keys for k in ks]})
    ap = ap[ap.key.str.contains(" ") & (ap.key.str.split().str[0].str.len() > 1)].drop_duplicates(["pid", "key"])
    ap = ap.merge(d[["pid", "t", "year", "primary", "group", "n_authors"]], on="pid")
    n = ap.groupby("key").pid.transform("size")
    ap = ap[(n >= 5) & (n <= 300)].sort_values(["key", "t"]).reset_index(drop=True)
    ap["k"] = ap.groupby("key").cumcount()
    ap["gap"] = ap.t - ap.groupby("key").t.shift(1)
    cats = d[["pid", "categories"]].copy()
    cats["cat"] = cats.categories.str.split()
    cats = cats.explode("cat")[["pid", "cat"]]
    first_any = ap[["key", "pid", "t"]].merge(cats, on="pid").groupby(["key", "cat"]).t.min().rename("tf").reset_index()
    ap = ap.merge(first_any, left_on=["key", "primary"], right_on=["key", "cat"], how="left").drop(columns="cat")
    ap["new_cat"] = ap.t <= ap.tf + 1e-9
    ap["new_group"] = ap.t <= ap.groupby(["key", "group"]).t.transform("min") + 1e-9
    ap["prior_in_cat"] = ap.groupby(["key", "primary"]).cumcount()
    ap["inc_coauthor"] = ap.groupby("pid").prior_in_cat.transform("max") > 0
    ap = ap.drop(columns=["tf"])
    ap.to_parquet(CACHE)
print("panel", len(ap), ap.key.nunique(), flush=True)

# ---------------------------------------------------------------- 1. output around entry
ap = ap.sort_values(["key", "t"]).reset_index(drop=True)
aid = ap.key.astype("category").cat.codes.values.astype(np.int64)
BIG = 1e6
comp = aid * BIG + ap.t.values
edges = np.arange(-720, 721, 90)
ok = (ap.k >= 4) & (ap.t >= 760) & (ap.t <= TMAX - 730)
ev = ap[ok].copy()
ev["kind"] = np.where(ev.new_cat & ev.new_group & ~ev.inc_coauthor, "entry_new_group",
               np.where(ev.new_cat & ~ev.new_group & ~ev.inc_coauthor, "entry_same_group",
               np.where(ev.new_cat & ev.inc_coauthor, "entry_with_incumbent", "other")))
rng = np.random.default_rng(0)
keep = (ev.kind != "other") | (rng.random(len(ev)) < 0.12)
ev = ev[keep]
base = aid[ev.index.values] * BIG + ev.t.values
cnt = np.empty((len(ev), len(edges) - 1))
lo = np.searchsorted(comp, base + edges[0], side="left")
for j in range(len(edges) - 1):
    hi = np.searchsorted(comp, base + edges[j + 1], side="left")
    cnt[:, j] = hi - lo
    lo = hi
cnt[:, list(edges).index(0)] -= 1                      # the event paper itself
pre = cnt[:, :4].sum(axis=1)                           # papers in days -720..-360
ev["pre"] = np.minimum(pre, 12).astype(int)
ev["cell"] = ev.year.astype(str) + "_" + ev.pre.astype(str)
w_other = ev[ev.kind == "other"].groupby("cell").size()
prof = {}
for kind in ("entry_new_group", "entry_same_group", "entry_with_incumbent"):
    e = ev.kind == kind
    share = ev[e].groupby("cell").size() / e.sum()
    wgt = (ev.loc[ev.kind == "other", "cell"].map(share / w_other)).fillna(0).values
    c_other = (cnt[(ev.kind == "other").values] * wgt[:, None]).sum(axis=0) / wgt.sum()
    c_entry = cnt[e.values].mean(axis=0)
    rate = cnt[e.values][:, :4].sum(axis=1).mean() / 360.0     # papers per day in the far pre-period
    diff = c_entry - c_other
    prof[kind] = dict(n=int(e.sum()), bins=[int(x) for x in edges[:-1]], entry=[float(x) for x in c_entry],
                      matched_other=[float(x) for x in c_other], diff=[float(x) for x in diff],
                      pre_rate_per_year=float(rate * 365),
                      papers_lost_year_before=float(-diff[4:8].sum()), papers_lost_two_years_after=float(-diff[8:].sum()),
                      years_lost_before=float(-diff[4:8].sum() / (rate * 365)))
    print(kind, "n", int(e.sum()), "pre rate/yr %.2f" % (rate * 365),
          "diff by 90-day bin:", np.round(diff, 3), "years of output lost in the year before: %.3f" % prof[kind]["years_lost_before"], flush=True)
res["output_around_entry"] = prof

# ---------------------------------------------------------------- 2. persistence after entry
en = ap[(ap.k >= 4) & ap.new_cat & (ap.year >= 2017) & (ap.year <= 2021)].copy()
last = ap.groupby(["key", "primary"]).t.max().rename("t_last").reset_index()
ncat = ap.groupby(["key", "primary"]).size().rename("n_in_cat").reset_index()
en = en.merge(last, on=["key", "primary"]).merge(ncat, on=["key", "primary"])
en["stay_days"] = en.t_last - en.t
pers = {}
for lab, m in (("no_incumbent", ~en.inc_coauthor), ("with_incumbent", en.inc_coauthor), ("new_group_no_incumbent", en.new_group & ~en.inc_coauthor)):
    e = en[m]
    pers[lab] = dict(n=int(len(e)), share_never_again=float((e.n_in_cat == 1).mean()),
                     share_active_after_1y=float((e.stay_days > 365).mean()), share_active_after_3y=float((e.stay_days > 3 * 365).mean()),
                     mean_papers_in_field=float(e.n_in_cat.mean()))
    print("persistence", lab, pers[lab], flush=True)
res["persistence_after_entry"] = pers
# tenure in a home field: authors' main category, hazard of the last paper there (2017-2021 cohorts of activity)
main = ap.groupby(["key", "primary"]).agg(n=("pid", "size"), t0=("t", "min"), t1=("t", "max")).reset_index()
main = main[main.n >= 5]
main["span_years"] = (main.t1 - main.t0) / 365.0
obs = main[(main.t0 < 365 * 4)]                      # spells that started 2015-2018
alive5 = (obs.t1 > obs.t0 + 5 * 365).mean()
res["tenure"] = dict(n_spells=int(len(obs)), share_still_publishing_5y_after_first=float(alive5),
                     implied_annual_exit_rate=float(-np.log(alive5) / 5), median_span_years=float(obs.span_years.median()))
print("tenure", res["tenure"], flush=True)

# ---------------------------------------------------------------- 3. stock-to-flow within destination fields
cy = df.groupby(["primary", "year"]).size().rename("P").reset_index().sort_values(["primary", "year"])
rows = []
for c, g in cy.groupby("primary"):
    S = 0.0
    for yv, pv in zip(g.year.values, g.P.values):
        S = S * np.exp(-0.25) + pv
        rows.append((c, yv, S))
cy = cy.merge(pd.DataFrame(rows, columns=["primary", "year", "L"]), on=["primary", "year"])
cy["P_lag"] = cy.groupby("primary").P.shift(1)
cy["L_lag"] = cy.groupby("primary").L.shift(1)
base_gap = ap[(ap.k >= 4) & ~ap.new_cat & ap.gap.notna()].groupby("key").gap.median().rename("base_gap")
e2 = ap[(ap.k >= 4) & ap.new_cat & ~ap.inc_coauthor & ap.gap.notna() & (ap.year >= 2019) & (ap.year <= 2026)].merge(base_gap, on="key")
e2 = e2.merge(cy[["primary", "year", "P_lag", "L_lag"]], on=["primary", "year"]).dropna()
e2 = e2[e2.P_lag >= 200]
e2["lgap"] = np.log1p(e2.gap)
X = pd.DataFrame({"ln_SF": np.log(e2.L_lag / e2.P_lag), "ln_P": np.log(e2.P_lag), "ln_team": np.log(e2.n_authors), "ln_base": np.log1p(e2.base_gap)})
y = e2.lgap.copy()
for _ in range(4):
    for g in (e2.primary, e2.year):
        y = y - y.groupby(g.values).transform("mean")
        X = X - X.groupby(g.values).transform("mean")
b, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
r = y.values - X.values @ b
XtX = np.linalg.inv(X.values.T @ X.values)
sc = pd.DataFrame(X.values * r[:, None]).groupby(e2.primary.values).sum().values
se = np.sqrt(np.diag(XtX @ (sc.T @ sc) @ XtX))
res["destination_within_field"] = dict(n=int(len(e2)), clusters=int(e2.primary.nunique()),
                                       coef={c: [float(bb), float(ss)] for c, bb, ss in zip(X.columns, b, se)},
                                       sd_ln_SF_within=float(X.ln_SF.std()))
print("destination, field and year effects:", res["destination_within_field"], flush=True)

# ---------------------------------------------------------------- 4. pedagogical papers
txt = (df.title.fillna("") + " || " + df.comments.fillna("")).str.lower()
ped = {"lecture notes": r"\blectures?\b|\blecture notes\b", "review or survey": r"\breview\b|\bsurvey\b|\boverview\b",
       "tutorial or primer": r"\btutorial\b|\bprimer\b|\bpedagogical\b|\bintroduction to\b"}
res["pedagogical"] = {}
for lab, pat in ped.items():
    m = txt.str.contains(pat, regex=True)
    by = m.groupby(df.year).sum()
    res["pedagogical"][lab] = dict(total=int(m.sum()), per_year_2019_2025=float(by.loc[2019:2025].mean()))
    print("pedagogical", lab, int(m.sum()), "per year 2019-25: %.0f" % by.loc[2019:2025].mean(), flush=True)

# ---------------------------------------------------------------- 5. references as a proxy for reading
s2 = []
with open(os.path.join(ROOT, "data_raw", "s2_focal.jsonl")) as f:
    for line in f:
        d_ = json.loads(line)
        rec = d_.get("rec") or {}
        if rec.get("referenceCount") is not None:
            s2.append((d_["arxiv"], rec["referenceCount"], rec.get("citationCount")))
s2 = pd.DataFrame(s2, columns=["id", "refs", "cites"]).merge(df[["id", "pid", "primary", "year", "n_authors"]], on="id")
pp = ap.groupby("pid").agg(any_new=("new_cat", "max"), all_new=("new_cat", "min"), kmin=("k", "min")).reset_index()
s2 = s2.merge(pp, on="pid")
s2 = s2[(s2.refs > 0) & (s2.kmin >= 1)]
s2["lrefs"] = np.log(s2.refs)
s2["cellfe"] = s2.primary + "_" + s2.year.astype(str)
y = s2.lrefs - s2.groupby("cellfe").lrefs.transform("mean")
X = pd.DataFrame({"any_new": s2.any_new.astype(float), "all_new": s2.all_new.astype(float), "ln_team": np.log(s2.n_authors)})
X = X - X.groupby(s2.cellfe.values).transform("mean")
b, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
r = y.values - X.values @ b
se = np.sqrt(np.diag(np.linalg.inv(X.values.T @ X.values) * (r @ r) / (len(r) - 3)))
res["references"] = dict(n=int(len(s2)), mean_refs=float(s2.refs.mean()), coef={c: [float(bb), float(ss)] for c, bb, ss in zip(X.columns, b, se)})
print("references:", res["references"], flush=True)
json.dump(res, open(os.path.join(OUT, "entry_dynamics.json"), "w"), indent=1)

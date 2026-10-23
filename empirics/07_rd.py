"""Listing position and the fate of later overlapping work: RD estimates.

A paper submitted just after the daily cutoff opens the next listing; one
submitted just before closes the current one. We compare them.

Outcomes for focal paper i:
  cites           Semantic Scholar citation count (log 1+c; within category-year
                  percentile)
  nd95 / nd99     later papers in the category within 24 months whose TF-IDF
                  similarity to i exceeds the category's tau95 / tau99 overlap
                  threshold, excluding papers that share an author with i
  aware/unaware   those later near-duplicates that do / do not cite i
                  (Semantic Scholar reference lists)
Lemma 1 predicts that higher visibility of i (detection by later researchers)
lowers unaware near-duplicates by theta*dphi and raises aware ones by
q*theta*dphi, so the ratio of the two effects identifies q.

Specification: local linear in minutes from the cutoff, separate slopes,
triangular kernel, bandwidth h, donut excluding |run| < d, category x year and
weekday fixed effects, standard errors clustered by submission day.
Writes empirics/results/rd.json and figures.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, "empirics")
from names import author_keys, submitter_key

R = pd.read_parquet("data/rd_sample.parquet")
M = pd.read_parquet("data/arxiv_meta.parquet",
                    columns=["id", "v1", "authors", "title", "comments"])
T = pd.read_parquet("data/arxiv_text.parquet", columns=["id", "abstract"])
R = R.merge(M, on="id", how="left").merge(T, on="id", how="left")
R["abs_len"] = R.abstract.str.split().str.len()
R["title_len"] = R.title.str.split().str.len()
R["day"] = R.v1.dt.tz_convert("America/New_York").dt.date.astype(str)

# submitter experience: prior submissions by the same name key since 2015
allm = pd.read_parquet("data/arxiv_meta.parquet", columns=["id", "v1", "submitter"])
allm = allm.sort_values("v1")
allm["skey"] = allm.submitter.map(submitter_key)
allm["prior_subs"] = allm.groupby("skey").cumcount()
R = R.merge(allm[["id", "prior_subs"]], on="id", how="left")
R["log_prior"] = np.log1p(R.prior_subs)

# ---- near-duplicate outcomes -------------------------------------------------
P = pd.read_parquet("data/nd_pairs.parquet")
auth = dict(zip(M.id, M.authors))
P["self"] = [bool(set(author_keys(auth.get(i))) & set(author_keys(auth.get(j))))
             for i, j in zip(P.i, P.j)]
aw = {}
try:
    for line in open("data_raw/s2_j.jsonl"):
        r = json.loads(line)
        rec = r["rec"]
        if rec is None:
            continue
        refs = {(x.get("externalIds") or {}).get("ArXiv") for x in (rec.get("references") or [])}
        aw[r["arxiv"]] = refs
except FileNotFoundError:
    pass
P["has_refs"] = P.j.map(lambda j: j in aw and len(aw[j]) > 0)
P["aware"] = [j in aw and i in aw[j] for i, j in zip(P.i, P.j)]
# INSPIRE-HEP (curated references) for high-energy-physics categories
ins = {}
for f in ("data_raw/insp_focal.jsonl", "data_raw/insp_j.jsonl"):
    try:
        for line in open(f):
            r = json.loads(line)
            if r.get("recid"):
                ins.setdefault(r["arxiv"], {}).update(r)
    except FileNotFoundError:
        pass
P["has_refs_ins"] = P.j.map(lambda j: bool(ins.get(j, {}).get("refs")))
P["aware_ins"] = [bool(ins.get(j, {}).get("refs")) and ins.get(i, {}).get("recid") in set(ins[j]["refs"])
                  for i, j in zip(P.i, P.j)]
P["p99"] = P.sim >= P.tau99
oth = P[~P.self]
agg = pd.DataFrame(index=R.id)
agg["nd95"] = oth.groupby("i").size()
agg["nd99"] = oth[oth.p99].groupby("i").size()
agg["nd95_late"] = oth[oth.dt_days >= 90].groupby("i").size()
agg["self95"] = P[P.self].groupby("i").size()
ok = oth[oth.has_refs]
agg["nd95_ref"] = ok.groupby("i").size()
agg["aware95"] = ok[ok.aware].groupby("i").size()
agg["unaware95"] = ok[~ok.aware].groupby("i").size()
agg["aware99"] = ok[ok.aware & ok.p99].groupby("i").size()
agg["unaware99"] = ok[~ok.aware & ok.p99].groupby("i").size()
oi = oth[oth.has_refs_ins]
agg["nd95_ins"] = oi.groupby("i").size()
agg["aware95_ins"] = oi[oi.aware_ins].groupby("i").size()
agg["unaware95_ins"] = oi[~oi.aware_ins].groupby("i").size()
agg = agg.fillna(0)
R = R.merge(agg, left_on="id", right_index=True, how="left")
R["any95"] = (R.nd95 > 0).astype(float)

# ---- citations ----------------------------------------------------------------
cit = {}
try:
    for line in open("data_raw/s2_focal.jsonl"):
        r = json.loads(line)
        if r["rec"] is not None:
            cit[r["arxiv"]] = r["rec"].get("citationCount")
except FileNotFoundError:
    pass
R["cites"] = R.id.map(cit)
R["cites_ins"] = R.id.map(lambda i: ins.get(i, {}).get("cc_noself"))
R["lcites_ins"] = np.log1p(R.cites_ins)
HEP = ["hep-th", "hep-ph", "hep-ex", "hep-lat", "gr-qc", "nucl-th", "nucl-ex",
       "astro-ph.CO", "astro-ph.HE"]
R["hep"] = R.primary.isin(HEP)
for k in ("nd95_ins", "aware95_ins", "unaware95_ins"):
    R.loc[~R.hep, k] = np.nan
R["lcites"] = np.log1p(R.cites)
R["cite_pct"] = R.groupby(["primary", "year"]).cites.rank(pct=True)
R["top5"] = (R.pos <= 5).astype(float)
R["held_f"] = R.held.astype(float)
R["cy"] = R.primary + "_" + R.year.astype(str)


def demean(df, cols, fe):
    out = df[cols].copy()
    for _ in range(3):
        for f in fe:
            out = out - out.groupby(df[f]).transform("mean")
    return out


def rd(y, h=60, donut=0, df=R, covs=()):
    s = df[(df.run.abs() <= h) & (df.run.abs() >= donut) & df[y].notna()
           & df[list(covs)].notna().all(axis=1)].copy()
    s["wt"] = 1 - s.run.abs() / h
    s["A"] = s.after
    s["rA"] = s.run * s.after
    s["rB"] = s.run * (1 - s.after)
    cols = [y, "A", "rA", "rB", *covs]
    Z = demean(s, cols, ["cy", "weekday"])
    X = Z[["A", "rA", "rB", *covs]]
    f = sm.WLS(Z[y], X, weights=s.wt).fit(cov_type="cluster",
                                         cov_kwds={"groups": pd.factorize(s.day)[0]})
    base = s.loc[s.after == 0, y].mean()
    return {"b": float(f.params["A"]), "se": float(f.bse["A"]), "n": int(len(s)),
            "mean_before": float(base)}


res = {"sample": {"n": int(len(R)), "with_cites": int(R.cites.notna().sum()),
                  "pairs_95_nonself": int(len(oth)), "pairs_with_refs": int(len(ok)),
                  "aware_share_95": float(ok.aware.mean()) if len(ok) else None,
                  "aware_share_99": float(ok[ok.p99].aware.mean()) if len(ok) else None,
                  "self_share_95": float(P.self.mean()),
                  "aware_share_95_ins": float(oth[oth.has_refs_ins].aware_ins.mean()),
                  "pairs_with_refs_ins": int(oth.has_refs_ins.sum())}}
first = ["pos_frac", "top5", "held_f"]
bal = ["n_authors", "abs_len", "title_len", "log_prior", "n_list"]
outs = ["lcites", "cite_pct", "nd95", "nd99", "any95", "nd95_late", "self95",
        "nd95_ref", "aware95", "unaware95", "aware99", "unaware99",
        "lcites_ins", "nd95_ins", "aware95_ins", "unaware95_ins"]
for h, dn in ((60, 0), (60, 5), (120, 5), (30, 5), (60, 10), (180, 10)):
    key = f"h{h}_d{dn}"
    res[key] = {}
    for y in first + bal + outs:
        if R[y].notna().sum() > 1000:
            res[key][y] = rd(y, h, dn)
# covariate-adjusted main spec
res["h60_d5_cov"] = {y: rd(y, 60, 5, covs=("n_authors", "abs_len", "log_prior"))
                     for y in outs if R[y].notna().sum() > 1000}
# Lemma 1 ratio q = d aware / (-d unaware)
for k in ("h60_d5", "h120_d5", "h60_d5_cov"):
    for suf in ("95", "95_ins"):
        a, u = res[k].get("aware" + suf), res[k].get("unaware" + suf)
        if a and u and u["b"] != 0:
            res[k]["q_ratio_" + suf] = -a["b"] / u["b"]
json.dump(res, open("empirics/results/rd.json", "w"), indent=1)
R.to_parquet("data/rd_analysis.parquet", index=False)
for k in ("h60_d5", "h120_d5", "h60_d5_cov"):
    print(k)
    for y, v in res[k].items():
        if isinstance(v, dict):
            print(f"  {y:14s} {v['b']:+.4f} ({v['se']:.4f})  mean0 {v['mean_before']:.4f}  n {v['n']}")
print(res["sample"])

# ---- figures --------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for ax, y, lab in zip(axes, ["top5", "lcites_ins", "log_prior"],
                      ["Among the first five in the listing", "log(1 + citations), HEP (INSPIRE)",
                       "log(1 + submitter's prior submissions)"]):
    s = R[(R.run.abs() < 120) & R[y].notna()].copy()
    s[y + "_r"] = s[y] - s.groupby("cy")[y].transform("mean") + s[y].mean()
    s["bin"] = (s.run // 5) * 5 + 2.5
    b = s.groupby("bin")[y + "_r"].mean()
    ax.scatter(b.index, b.values, s=6, color="#2166ac")
    ax.axvline(0, color="k", lw=0.6)
    ax.axvspan(-5, 5, color="#dddddd", alpha=0.6, lw=0)
    ax.set_title(lab, fontsize=9)
    ax.set_xlabel("Minutes from cutoff", fontsize=8)
fig.tight_layout()
fig.savefig("empirics/figures/rd_listing.pdf")

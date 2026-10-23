"""Listing rank as a readership shock: high-energy physics, 2017-2022.

A paper's position in its category's daily listing is one plus the number of
papers in the category submitted earlier in the same batch. Holding the
submission time (minutes since the cutoff that opened the batch), the category,
the year, the weekday and the size of the listing fixed, position varies with
how many other people happened to submit earlier. Papers submitted in the first
15 minutes after the cutoff (where authors race for the top) are excluded.

Outcomes (INSPIRE-HEP):
  lcites        log(1 + citations excluding self-citations)
  N_q           later papers in the category within 24 months, by authors who
                do not overlap with i, whose TF-IDF similarity to i is at least
                tau_q (q = 50, 75, 90 percentile of nearest-neighbour similarity)
  A_q / U_q     those that cite / do not cite i (INSPIRE reference lists)
Lemma 1: dA = q theta dphi and dU = -theta dphi, so -dA/dU = q, the share of
detected overlaps that proceed as follow-ons.

Writes empirics/results/rank_design.json and a figure.
"""
import glob
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

S = pd.read_parquet("data/hep_sample.parquet")
M = pd.read_parquet("data/arxiv_meta.parquet", columns=["id", "authors"])
S = S.merge(M, on="id", how="left")
T = pd.read_parquet("data/arxiv_text.parquet", columns=["id", "abstract"])
S = S.merge(T[T.id.isin(set(S.id))], on="id", how="left")
S["abs_len"] = S.abstract.str.split().str.len()
S = S.drop(columns="abstract")
allm = pd.read_parquet("data/arxiv_meta.parquet", columns=["id", "v1", "submitter"]).sort_values("v1")
allm["prior"] = allm.groupby(allm.submitter.map(submitter_key)).cumcount()
S = S.merge(allm[["id", "prior"]], on="id", how="left")
S["log_prior"] = np.log1p(S.prior)
S["lna"] = np.log(S.n_authors.clip(lower=1))
ins = {}
for f in sorted(glob.glob("data_raw/insp_*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        if r.get("recid"):
            ins.setdefault(r["arxiv"], {}).update(r)
S["cites"] = S.id.map(lambda i: ins.get(i, {}).get("cc_noself"))
S["lcites"] = np.log1p(S.cites)
S["cite_pct"] = S.groupby(["primary", "year"]).cites.rank(pct=True)

P = pd.read_parquet("data/hep_pairs.parquet")
auth = dict(zip(M.id, M.authors))
ak = {}
def keys(x):
    if x not in ak:
        ak[x] = set(author_keys(auth.get(x)))
    return ak[x]
P = P[P.i.isin(set(S.id))]
P["self"] = [bool(keys(i) & keys(j)) for i, j in zip(P.i, P.j)]
P["has_refs"] = P.j.map(lambda j: bool(ins.get(j, {}).get("refs")))
P["aware"] = [P_has and ins.get(i, {}).get("recid") in set(ins[j]["refs"])
              for i, j, P_has in zip(P.i, P.j, P.has_refs)]
oth = P[~P.self]
agg = pd.DataFrame(index=S.id)
for q in (50, 75, 90):
    o = oth[oth.sim >= oth[f"tau{q}"]]
    agg[f"N{q}"] = o.groupby("i").size()
    oo = o[o.has_refs]
    agg[f"A{q}"] = oo[oo.aware].groupby("i").size()
    agg[f"U{q}"] = oo[~oo.aware].groupby("i").size()
    agg[f"S{q}"] = P[P.self & (P.sim >= P[f"tau{q}"])].groupby("i").size()
agg = agg.fillna(0)
S = S.merge(agg, left_on="id", right_index=True, how="left")
S["lpos"] = np.log(S.pos)
S["lnl"] = np.log(S.n_list)
S["mon"] = (S.batch_dow == 0).astype(int)
S["tbin"] = np.where(S.tsc < 1440, S.tsc // 15, 96 + (S.tsc - 1440) // 60).astype(int)
S["tb"] = S.mon.astype(str) + "_" + S.tbin.astype(str)
S["cy"] = S.primary + "_" + S.year.astype(str)
S["blk"] = S.block.astype(str)


def demean(df, cols, fe, it=6):
    Z = df[cols].astype(float).copy()
    for _ in range(it):
        for f in fe:
            Z = Z - Z.groupby(df[f]).transform("mean")
    return Z


def est(y, df, x="lpos", ctrl=("lnl",), fe=("cy", "tb")):
    d = df[df[y].notna()]
    xs = [x] if isinstance(x, str) else list(x)
    Z = demean(d, [y, *xs, *ctrl], list(fe))
    f = sm.OLS(Z[y], Z[[*xs, *ctrl]]).fit(cov_type="cluster",
                                          cov_kwds={"groups": pd.factorize(d.blk)[0]})
    if isinstance(x, str):
        return {"b": float(f.params[x]), "se": float(f.bse[x]), "n": int(len(d)),
                "mean": float(d[y].mean())}
    return {k: [float(f.params[k]), float(f.bse[k])] for k in xs} | {"n": int(len(d)),
            "mean": float(d[y].mean())}


base = S[(S.tsc >= 15)]
res = {"n": int(len(base)), "share_with_cites": float(base.cites.notna().mean()),
       "resid_sd_lpos": float(demean(base, ["lpos", "lnl"], ["cy", "tb"]).lpos.std()),
       "raw_sd_lpos": float(base.lpos.std()),
       "pairs_nonself_50": int((oth.sim >= oth.tau50).sum()),
       "aware_share_nonself": {q: float(oth[(oth.sim >= oth[f"tau{q}"]) & oth.has_refs].aware.mean())
                               for q in (50, 75, 90)},
       "self_share_50": float(P.self.mean())}
outs = ["lcites", "cite_pct"] + [f"{k}{q}" for q in (50, 75, 90) for k in ("N", "A", "U", "S")]
specs = {
    "main": dict(df=base),
    "weekday_batches": dict(df=base[base.mon == 0]),
    "first_10h": dict(df=base[base.tsc <= 600]),
    "fine_bins": dict(df=base.assign(tb=base.mon.astype(str) + "_" + (base.tsc // 5).astype(int).astype(str))),
    "top10_indicator": dict(df=base.assign(top10=(base.pos <= 10).astype(float)), x="top10"),
    "batch_fe": dict(df=base, fe=("cy", "tb", "blk")),
    "first_6h_batch_fe": dict(df=base[base.tsc <= 360], fe=("cy", "tb", "blk")),
    "add_covariates": dict(df=base.dropna(subset=["abs_len", "log_prior"]),
                           ctrl=("lnl", "lna", "abs_len", "log_prior")),
}
for name, sp in specs.items():
    res[name] = {y: est(y, sp["df"], sp.get("x", "lpos"), sp.get("ctrl", ("lnl",)),
                        sp.get("fe", ("cy", "tb"))) for y in outs}
for name in specs:
    for q in (50, 75, 90):
        a, u = res[name][f"A{q}"], res[name][f"U{q}"]
        res[name][f"q_ratio_{q}"] = -a["b"] / u["b"] if u["b"] != 0 else None
# position bins (reference: position 16 or lower), batch fixed effects
for k, (lo, hi) in {"p1": (1, 1), "p2_3": (2, 3), "p4_7": (4, 7), "p8_15": (8, 15)}.items():
    base[k] = base.pos.between(lo, hi).astype(float)
bins = ("p1", "p2_3", "p4_7", "p8_15")
res["bins_batch_fe"] = {y: est(y, base, bins, ("lnl",), ("cy", "tb", "blk"))
                        for y in ["lcites", "cite_pct", "N50", "A50", "U50", "S50", "N75", "A75", "U75", "S75"]}
# balance: predetermined characteristics should not move with rank
res["balance"] = {y: est(y, base, "lpos") for y in ["lna", "abs_len", "log_prior"]}
res["balance_batch_fe"] = {y: est(y, base, "lpos", ("lnl",), ("cy", "tb", "blk"))
                           for y in ["lna", "abs_len", "log_prior"]}
json.dump(res, open("empirics/results/rank_design.json", "w"), indent=1)
print({k: v for k, v in res.items() if not isinstance(v, dict) or k in ("aware_share_nonself",)})
for name in specs:
    print(name)
    for y in outs:
        r = res[name][y]
        print(f"  {y:9s} {r['b']:+.4f} ({r['se']:.4f}) mean {r['mean']:.4f} n {r['n']}")
    print("  q ratios", {q: res[name][f"q_ratio_{q}"] for q in (50, 75, 90)})
print("balance", res["balance"])
print("balance_batch_fe", res["balance_batch_fe"])
for y, r in res["bins_batch_fe"].items():
    print("bins", y, {k: (round(v[0], 4), round(v[1], 4)) for k, v in r.items() if isinstance(v, list)},
          round(r["mean"], 3))

# binned scatter: residualised outcome on residualised log position
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for ax, y, lab in zip(axes, ["lcites", "A75", "U75"],
                      ["log(1 + citations)", "Aware overlap (cite i), tau75",
                       "Unaware overlap, tau75"]):
    d = base[base[y].notna()]
    Z = demean(d, [y, "lpos", "lnl"], ["cy", "tb"])
    bx = sm.OLS(Z.lpos, Z[["lnl"]]).fit().resid
    by = sm.OLS(Z[y], Z[["lnl"]]).fit().resid
    q = pd.qcut(bx, 20, labels=False, duplicates="drop")
    gx, gy = bx.groupby(q).mean(), by.groupby(q).mean() + d[y].mean()
    ax.scatter(gx, gy, s=10, color="#2166ac")
    ax.set_title(lab, fontsize=9)
    ax.set_xlabel("log listing position (residual)", fontsize=8)
fig.tight_layout()
fig.savefig("empirics/figures/rank_design.pdf")

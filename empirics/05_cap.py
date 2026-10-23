"""The October 2026 submission cap: at most two new submissions per submitter per
calendar month, announced and effective on 1 October 2026 (arXiv blog,
"Fair Moderation, Equitable Access, and AI: arXiv's Updated Rate Limit Policy").

The snapshot covers submissions to 15 October 2026 (14:00 ET). All comparisons
use days 1-15 of each month and papers announced in their regular batch (not
held), so that the October window is comparable with earlier months.

1. Mechanical exposure: the share of papers that were a submitter's third or
   later submission within days 1-15 of the month.
2. First stage: that share in October 2026.
3. Volume: papers per weekday, Oct 1-15 vs Sep 1-15, 2026 against the same
   change in 2024 and 2025; category dose-response in exposure.
4. Rotation: papers by heavy submitters' name keys submitted by a co-author.
Writes empirics/results/cap.json.
"""
import json
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, "empirics")
from names import author_keys, submitter_key

L = pd.read_parquet("data/listing.parquet",
                    columns=["id", "primary", "submitter", "held", "block"])
M = pd.read_parquet("data/arxiv_meta.parquet", columns=["id", "v1", "authors"])
d = L.merge(M, on="id")
d = d[d.v1 >= "2023-06-01"].copy()
et = d.v1.dt.tz_convert("America/New_York")
d["ym"] = et.dt.strftime("%Y-%m")
d["dom"] = et.dt.day
d["date"] = et.dt.date
d["wd"] = et.dt.dayofweek < 5
d["skey"] = d.submitter.map(submitter_key)
d["archive"] = d.primary.str.split(".").str[0].replace(
    {"hep-th": "physics", "hep-ph": "physics", "hep-ex": "physics", "hep-lat": "physics",
     "astro-ph": "physics", "cond-mat": "physics", "gr-qc": "physics", "nucl-th": "physics",
     "nucl-ex": "physics", "quant-ph": "physics", "nlin": "physics", "math-ph": "physics"})
h = d[d.dom <= 15].sort_values("v1").copy()
h["rank_m"] = h.groupby(["ym", "skey"]).cumcount() + 1
h["excess"] = h.rank_m >= 3
# name collisions: common names whose half-month submissions span several
# archives belong to different people; "clean" excess keeps names whose
# submissions in the half-month fall in a single archive
h["n_arch"] = h.groupby(["ym", "skey"]).archive.transform("nunique")
h["excess_clean"] = h.excess & (h.n_arch == 1) & (h.submitter.str.upper() != h.submitter)
res = {}
by_m = h.groupby("ym").agg(n=("id", "size"), excess=("excess", "mean"),
                           excess_clean=("excess_clean", "mean"))
res["excess_share_by_month_days1_15"] = by_m.excess.round(4).to_dict()
res["excess_clean_share_by_month_days1_15"] = by_m.excess_clean.round(4).to_dict()
pre = h[(h.ym >= "2026-01") & (h.ym <= "2026-09")]
res["excess_share_2026_jan_sep"] = float(pre.excess.mean())
res["excess_share_2026_oct"] = float(h[h.ym == "2026-10"].excess.mean())
res["max_rank_oct2026"] = int(h[h.ym == "2026-10"].rank_m.max())
res["excess_clean_2026_jan_sep"] = float(pre.excess_clean.mean())
res["excess_clean_2026_oct"] = float(h[h.ym == "2026-10"].excess_clean.mean())

# volume: regular-batch papers per weekday, days 1-15
v = h[~h.held & h.wd].groupby(["ym"]).agg(n=("id", "size"), days=("date", "nunique"))
v["per_day"] = v.n / v.days
res["papers_per_weekday_days1_15"] = v.per_day.round(1).to_dict()
for y in (2023, 2024, 2025, 2026):
    if f"{y}-10" in v.index and f"{y}-09" in v.index:
        res[f"dlog_oct_sep_{y}"] = float(np.log(v.loc[f"{y}-10", "per_day"])
                                         - np.log(v.loc[f"{y}-09", "per_day"]))

# category dose-response: exposure from Jan-Sep 2026; outcome dlog Oct-Sep 2026
# minus dlog Oct-Sep 2025
exp_c = pre.groupby("primary").agg(expo=("excess_clean", "mean"), npre=("id", "size"))
w = h[~h.held & h.wd & h.ym.isin(["2025-09", "2025-10", "2026-09", "2026-10"])]
cnt = w.groupby(["primary", "ym"]).size().unstack(fill_value=0)
days = w.groupby("ym").date.nunique()
rate = cnt / days
ok = (cnt.min(axis=1) >= 20)
r = rate[ok]
dd = pd.DataFrame({"dlog26": np.log(r["2026-10"]) - np.log(r["2026-09"]),
                   "dlog25": np.log(r["2025-10"]) - np.log(r["2025-09"])})
dd["ddlog"] = dd.dlog26 - dd.dlog25
dd = dd.join(exp_c, how="inner")
fit = smf.wls("ddlog ~ expo", data=dd, weights=dd.npre).fit(cov_type="HC1")
fit26 = smf.wls("dlog26 ~ expo", data=dd, weights=dd.npre).fit(cov_type="HC1")
res["dose_response"] = {"n_categories": int(len(dd)),
                        "b_expo_DiD": float(fit.params["expo"]), "se_DiD": float(fit.bse["expo"]),
                        "b_expo_2026only": float(fit26.params["expo"]),
                        "se_2026only": float(fit26.bse["expo"]),
                        "mean_expo": float(np.average(dd.expo, weights=dd.npre)),
                        "p90_expo": float(dd.expo.quantile(0.9))}
dd.reset_index().to_csv("empirics/results/cap_categories.csv", index=False)

# rotation: heavy submitters (>=3 submissions within days 1-15 of some month
# in Jan-Sep of the year); their papers as submitter vs as non-submitting
# author (initial key) in days 1-15 of October vs Jan-Sep average
h["akeys"] = h.authors.map(lambda a: [k for k in author_keys(a, full=True)
                                       if len(k.split()[0]) > 1])
rot = {}
for y in (2025, 2026):
    pre_y = h[(h.ym >= f"{y}-01") & (h.ym <= f"{y}-09")]
    heavy = set(pre_y.loc[pre_y.excess_clean, "skey"].dropna())
    ex = h[(h.ym >= f"{y}-01") & (h.ym <= f"{y}-10") & ~h.held]
    as_sub = ex[ex.skey.isin(heavy)].groupby("ym").size()
    ak = ex[["ym", "akeys", "skey"]].explode("akeys")
    as_coauth = ak[ak.akeys.isin(heavy) & (ak.akeys != ak.skey)].groupby("ym").size()
    rot[y] = {"n_heavy": len(heavy),
              "as_submitter_pre_mean": float(as_sub.loc[f"{y}-01":f"{y}-09"].mean()),
              "as_submitter_oct": float(as_sub.get(f"{y}-10", 0)),
              "as_coauthor_pre_mean": float(as_coauth.loc[f"{y}-01":f"{y}-09"].mean()),
              "as_coauthor_oct": float(as_coauth.get(f"{y}-10", 0))}
    rot[y]["dlog_submitter"] = float(np.log(rot[y]["as_submitter_oct"] / rot[y]["as_submitter_pre_mean"]))
    rot[y]["dlog_coauthor"] = float(np.log(rot[y]["as_coauthor_oct"] / rot[y]["as_coauthor_pre_mean"]))
    tot = ex.groupby("ym").size()
    rot[y]["dlog_all_papers"] = float(np.log(tot[f"{y}-10"] / tot.loc[f"{y}-01":f"{y}-09"].mean()))
res["rotation"] = rot
json.dump(res, open("empirics/results/cap.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if "by_month" not in k}, indent=1))
print(by_m.tail(16))

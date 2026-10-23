"""Two arXiv moderation shocks.

Endorsement (arXiv blog, 10 Dec 2025 for Mathematics; 21 Jan 2026 for all
sections): an institutional e-mail address no longer suffices for automatic
endorsement of a new submitter. Outcome: the share of papers whose submitter
has never submitted before ("first-time submitters") and the share whose
submitter also never appears as an author ("no track record"), from name keys.
Design: triple difference, Mathematics vs other sections, policy season
(Oct 2025-Apr 2026) vs the same calendar weeks a year earlier.

CS review and position papers (arXiv blog, 31 Oct 2025): these must have been
accepted after peer review before they are posted in Computer Science.
Outcome: share of papers whose title marks a review, survey, overview,
tutorial or position paper. Design: CS vs other archives, monthly, 2023-2026.

Writes empirics/results/entry_policies.json and figures.
"""
import json
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, "empirics")
from names import author_keys, submitter_key

L = pd.read_parquet("data/listing.parquet", columns=["id", "primary", "submitter", "block"])
M = pd.read_parquet("data/arxiv_meta.parquet",
                    columns=["id", "v1", "authors", "title", "categories"])
d = L.merge(M, on="id").sort_values("v1").reset_index(drop=True)
d["skey"] = d.submitter.map(submitter_key)
d["ikey"] = d.submitter.map(lambda s: submitter_key(s, full=False))
seen_sub, seen_auth = set(), set()
new_sub = np.zeros(len(d), bool)
no_track = np.zeros(len(d), bool)
for k, (s, i, a) in enumerate(zip(d.skey, d.ikey, d.authors)):
    if s is not None and s not in seen_sub:
        new_sub[k] = True
        no_track[k] = i not in seen_auth
    seen_sub.add(s)
    seen_auth.update(author_keys(a))
d["new_sub"] = new_sub
d["no_track"] = no_track
d["math"] = d.primary.str.startswith("math.") & (d.primary != "math.MP")
d["cs"] = d.primary.str.startswith("cs.")
res = {}

# ---- endorsement: weekly cells by announcement week -------------------------
e = d[(d.block >= "2024-09-02") & (d.block < "2026-10-12")].copy()
e["week"] = pd.to_datetime(e.block).dt.to_period("W-SUN").dt.start_time
e["season"] = np.where(e.week >= "2025-09-01", 1, 0)
# calendar position within season: weeks since 1 Sep of season start
e["wk_in"] = ((e.week - pd.to_datetime(np.where(e.season == 1, "2025-09-01", "2024-09-02")))
              .dt.days // 7)
e["grp"] = np.where(e.math, "math", "other")
cell = e.groupby(["grp", "season", "week", "wk_in"]).agg(
    n=("id", "size"), new_sub=("new_sub", "mean"), no_track=("no_track", "mean"),
    n_new=("new_sub", "sum")).reset_index()
cell["lnew"] = np.log(cell.n_new.clip(lower=1))
ref_math = (pd.Timestamp("2025-12-08") - pd.Timestamp("2025-09-01")).days // 7
ref_all = (pd.Timestamp("2026-01-19") - pd.Timestamp("2025-09-01")).days // 7
cell["post_math"] = (cell.wk_in >= ref_math).astype(int)
cell["post_all"] = (cell.wk_in >= ref_all).astype(int)
cell["M"] = (cell.grp == "math").astype(int)
cell["S"] = cell.season
win = cell[cell.wk_in < ref_all + 14].copy()       # through end of April
# year-on-year differences remove each group's seasonality; Newey-West SEs
yoy = win.pivot_table(index=["grp", "wk_in"], columns="S",
                      values=["new_sub", "no_track", "lnew", "n"]).reset_index()
out = {}
for y in ("new_sub", "no_track", "lnew"):
    out[y] = {}
    for g, ref in (("math", ref_math), ("other", ref_all)):
        z = yoy[yoy.grp == g].sort_values("wk_in")
        dy = (z[(y, 1)] - z[(y, 0)]).to_numpy()
        post = (z.wk_in >= ref).astype(float).to_numpy()
        f = smf.ols("dy ~ post", data=pd.DataFrame({"dy": dy, "post": post})).fit(
            cov_type="HAC", cov_kwds={"maxlags": 4})
        out[y][f"yoy_did_{g}"] = [float(f.params["post"]), float(f.bse["post"])]
        out[y][f"pre_mean_{g}"] = float(z.loc[z.wk_in < ref, (y, 1)].mean())
    # staggered window only: math minus other, weeks before 21 Jan
    zm = yoy[yoy.grp == "math"].set_index("wk_in")
    zo = yoy[yoy.grp == "other"].set_index("wk_in")
    dd = ((zm[(y, 1)] - zm[(y, 0)]) - (zo[(y, 1)] - zo[(y, 0)])).loc[:ref_all - 1]
    f = smf.ols("dd ~ post", data=pd.DataFrame({"dd": dd.to_numpy(),
                "post": (dd.index >= ref_math).astype(float)})).fit(
        cov_type="HAC", cov_kwds={"maxlags": 4})
    out[y]["staggered_math_vs_other"] = [float(f.params["post"]), float(f.bse["post"])]
res["endorsement"] = out
# simple period means for the text
per = e.assign(period=np.select(
    [e.wk_in < ref_math, e.wk_in < ref_all, e.wk_in < ref_all + 14],
    ["P0_sep_dec9", "P1_dec10_jan20", "P2_jan21_apr"], "P3_may_oct"))
res["endorsement_period_means"] = (per.groupby(["season", "grp", "period"])
                                   [["new_sub", "no_track"]].mean().round(4)
                                   .reset_index().to_dict("records"))
# figure: YoY difference in first-time submitter share, by group
piv = cell.pivot_table(index="wk_in", columns=["grp", "season"], values="new_sub")
fig, ax = plt.subplots(figsize=(7, 3.4))
for g, c in (("math", "#b2182b"), ("other", "#2166ac")):
    yy = (piv[(g, 1)] - piv[(g, 0)]).rolling(3, center=True, min_periods=1).mean() * 100
    ax.plot(yy.index, yy.values, color=c, label={"math": "Mathematics", "other": "Other sections"}[g])
ax.axvline(ref_math, color="#b2182b", ls=":", lw=1)
ax.axvline(ref_all, color="#2166ac", ls=":", lw=1)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("Weeks since 1 September (2025-26 season)")
ax.set_ylabel("First-time submitter share,\nchange on a year earlier (pp)")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("empirics/figures/endorsement.pdf")

# ---- CS reviews -------------------------------------------------------------
rx = re.compile(r"\b(?:survey|review|overview|tutorial|position paper|position:|primer|"
                r"roadmap|perspective|state of the art|systematic literature)\b", re.I)
c = d[(d.v1 >= "2023-01-01")].copy()
c["rev"] = c.title.str.contains(rx)
c["ym"] = c.v1.dt.tz_convert("America/New_York").dt.strftime("%Y-%m")
c["crosscs"] = ~c.cs & c.categories.str.contains(r"\bcs\.")
c["grp"] = np.select([c.cs, c.crosscs], ["cs", "noncs_crosslisted_cs"], "noncs")
mon = c.groupby(["grp", "ym"]).agg(n=("id", "size"), rev=("rev", "mean"),
                                   nrev=("rev", "sum")).reset_index()
mon["post"] = (mon.ym >= "2025-11").astype(int)
mon["CS"] = (mon.grp == "cs").astype(int)
mon["t"] = mon.groupby("grp").cumcount()
m2 = mon[mon.grp != "noncs_crosslisted_cs"]
f = smf.wls("rev ~ C(grp) + C(ym) + CS:post", data=m2, weights=m2.n).fit(cov_type="HC1")
f2 = smf.wls("rev ~ C(grp) + C(ym) + CS:post + CS:t", data=m2, weights=m2.n).fit(cov_type="HC1")
mon3 = mon[mon.grp != "cs"].assign(X=lambda x: (x.grp == "noncs_crosslisted_cs").astype(int))
f3 = smf.wls("rev ~ C(grp) + C(ym) + X:post", data=mon3, weights=mon3.n).fit(cov_type="HC1")
res["cs_reviews"] = {
    "did": [float(f.params["CS:post"]), float(f.bse["CS:post"])],
    "did_with_cs_trend": [float(f2.params["CS:post"]), float(f2.bse["CS:post"])],
    "leakage_noncs_primary_crosslisted_cs": [float(f3.params["X:post"]), float(f3.bse["X:post"])],
    "cs_pre_mean_2025_jan_oct": float(c[c.cs & (c.ym >= "2025-01") & (c.ym <= "2025-10")].rev.mean()),
    "cs_post_mean_2025_11_2026_10": float(c[c.cs & (c.ym >= "2025-11")].rev.mean()),
    "noncs_pre": float(c[(c.grp == "noncs") & (c.ym >= "2025-01") & (c.ym <= "2025-10")].rev.mean()),
    "noncs_post": float(c[(c.grp == "noncs") & (c.ym >= "2025-11")].rev.mean()),
    "cs_reviews_per_month_pre": float(c[c.cs & (c.ym >= "2025-01") & (c.ym <= "2025-10")].groupby("ym").rev.sum().mean()),
    "cs_reviews_per_month_post": float(c[c.cs & (c.ym >= "2025-11") & (c.ym <= "2026-09")].groupby("ym").rev.sum().mean()),
}
# dose-response within CS: categories that relied more on arXiv reviews lose
# more of them (first stage); outcome: share of papers with a first-time author
# (full-name key never seen before), Nov-Oct years
seen = set()
first_auth = np.zeros(len(d), bool)
for k, a in enumerate(d.authors):
    ks = [x for x in author_keys(a, full=True) if len(x.split()[0]) > 1]
    first_auth[k] = any(x not in seen for x in ks)
    seen.update(ks)
d["first_auth"] = first_auth
c = c.join(d["first_auth"])
c["yr"] = np.select([c.ym.between("2023-11", "2024-10"), c.ym.between("2024-11", "2025-10"),
                     c.ym.between("2025-11", "2026-10")], ["y24", "y25", "y26"], "")
cc = c[c.cs & (c.yr != "")]
cat = cc.groupby(["primary", "yr"]).agg(n=("id", "size"), rev=("rev", "mean"),
                                       fa=("first_auth", "mean")).unstack()
cat = cat[cat["n"].min(axis=1) >= 300]
dr = pd.DataFrame({"expo": cat[("rev", "y25")], "w": cat[("n", "y25")],
                   "d_rev": cat[("rev", "y26")] - cat[("rev", "y25")],
                   "d_fa": cat[("fa", "y26")] - cat[("fa", "y25")],
                   "expo_pl": cat[("rev", "y24")],
                   "d_fa_pl": cat[("fa", "y25")] - cat[("fa", "y24")],
                   "d_rev_pl": cat[("rev", "y25")] - cat[("rev", "y24")]})
fr = {k: smf.wls(f"{k} ~ {e}", data=dr, weights=dr.w).fit(cov_type="HC1")
      for k, e in (("d_rev", "expo"), ("d_fa", "expo"), ("d_fa_pl", "expo_pl"), ("d_rev_pl", "expo_pl"))}
res["cs_reviews"]["dose_response"] = {
    "n_cs_categories": int(len(dr)),
    "first_stage_d_rev_on_expo": [float(fr["d_rev"].params["expo"]), float(fr["d_rev"].bse["expo"])],
    "first_stage_placebo_prev_year": [float(fr["d_rev_pl"].params["expo_pl"]), float(fr["d_rev_pl"].bse["expo_pl"])],
    "reduced_form_first_author_share": [float(fr["d_fa"].params["expo"]), float(fr["d_fa"].bse["expo"])],
    "reduced_form_placebo_prev_year": [float(fr["d_fa_pl"].params["expo_pl"]), float(fr["d_fa_pl"].bse["expo_pl"])],
    "expo_mean": float(np.average(dr.expo, weights=dr.w)), "expo_sd": float(dr.expo.std())}
dr.to_csv("empirics/results/cs_review_categories.csv")
piv = mon.pivot(index="ym", columns="grp", values="rev") * 100
fig, ax = plt.subplots(figsize=(7, 3.2))
for g, col, lab in (("cs", "#b2182b", "Computer Science"), ("noncs", "#2166ac", "Other archives"),
                    ("noncs_crosslisted_cs", "#7f7f7f", "Other archives, cross-listed to CS")):
    ax.plot(range(len(piv)), piv[g].values, color=col, label=lab)
ax.axvline(list(piv.index).index("2025-11") - 0.5, color="k", ls=":", lw=1)
ax.set_xticks(range(0, len(piv), 6))
ax.set_xticklabels(piv.index[::6], fontsize=7)
ax.set_ylabel("Review-type titles (% of papers)")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("empirics/figures/cs_reviews.pdf")
json.dump(res, open("empirics/results/entry_policies.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "endorsement_period_means"}, indent=1))
print(pd.DataFrame(res["endorsement_period_means"]))

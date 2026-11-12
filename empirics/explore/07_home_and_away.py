"""Exploratory: do researchers write at home or away, and has that changed?

Home is the author's most frequent primary category over the three previous calendar years
(at least three papers there).  Each paper she writes in year t is at home, in another
category of the same broad field, or in another broad field.  January-September of each year.
Also: how the paper compares with papers by people at home there (team, references).
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ap = pd.read_parquet(os.path.join(ROOT, "data", "explore_author_panel.parquet"))
meta = pd.read_parquet(os.path.join(ROOT, "data", "arxiv_meta.parquet"), columns=["v1", "categories"])
meta["pid"] = np.arange(len(meta))
meta["month"] = meta.v1.dt.month
ap = ap.merge(meta[["pid", "month"]], on="pid")

# home category by rolling three-year window
cnt = ap.groupby(["key", "year", "primary"]).size().rename("n").reset_index()
homes = []
for y in range(2018, 2027):
    w = cnt[(cnt.year >= y - 3) & (cnt.year <= y - 1)].groupby(["key", "primary"]).n.sum().reset_index()
    w = w.sort_values(["key", "n"], ascending=[True, False])
    top = w.groupby("key").head(1)
    tot = w.groupby("key").n.sum().rename("tot")
    top = top.merge(tot, on="key")
    top = top[top.n >= 3]
    top["year"] = y
    top["home_share_prev"] = top.n / top.tot
    homes.append(top[["key", "year", "primary", "home_share_prev"]].rename(columns={"primary": "home"}))
homes = pd.concat(homes)
BROAD = {"hep-th": "hep", "hep-ph": "hep", "hep-ex": "hep", "hep-lat": "hep", "nucl-th": "nucl", "nucl-ex": "nucl"}
homes["home_group"] = homes.home.str.split(".").str[0].map(lambda g: BROAD.get(g, g))
d = ap[(ap.month <= 9) & (ap.year >= 2018)].merge(homes, on=["key", "year"])
d["where"] = np.where(d.primary == d.home, "home", np.where(d.group == d.home_group, "near", "far"))
# is some co-author at home in the paper's category?
at_home_here = d.assign(h=(d.primary == d.home)).groupby("pid").h.transform("max")
d["with_local"] = at_home_here & (d["where"] != "home")
rows = {}
for y, g in d.groupby("year"):
    aw = g[g["where"] != "home"]
    rows[int(y)] = dict(
        n=int(len(g)), authors=int(g.key.nunique()),
        share_home=float((g["where"] == "home").mean()), share_near=float((g["where"] == "near").mean()),
        share_far=float((g["where"] == "far").mean()),
        away_with_local_coauthor=float(aw.with_local.mean()), away_solo=float((aw.n_authors == 1).mean()),
        home_solo=float((g[g["where"] == "home"].n_authors == 1).mean()),
        papers_home_per_author=float((g["where"] == "home").groupby(g.key).sum().mean()),
        papers_away_per_author=float((g["where"] != "home").groupby(g.key).sum().mean()),
        home_share_prev=float(g.groupby("key").home_share_prev.first().mean()))
out = pd.DataFrame(rows).T
pd.set_option("display.width", 250)
print(out.round(3).to_string())
json.dump(rows, open(os.path.join(HERE, "results", "home_and_away.json"), "w"), indent=1)

# by broad home field: is the shift towards neighbouring categories general?
byf = {}
for grp, gg in d.groupby("home_group"):
    if gg.key.nunique() < 3000:
        continue
    t = gg.groupby("year")["where"].value_counts(normalize=True).unstack(fill_value=0)
    pa = gg.groupby("year").apply(lambda z: pd.Series({"home_pa": (z["where"] == "home").groupby(z.key).sum().mean(),
                                                       "away_pa": (z["where"] != "home").groupby(z.key).sum().mean()}))
    byf[grp] = {int(y): dict(home=float(t.loc[y, "home"]), near=float(t.loc[y].get("near", 0)), far=float(t.loc[y].get("far", 0)),
                             home_pa=float(pa.loc[y, "home_pa"]), away_pa=float(pa.loc[y, "away_pa"])) for y in (2019, 2022, 2024, 2026) if y in t.index}
    print(grp, " ".join(f"{y}: home {v['home']:.2f} near {v['near']:.2f} far {v['far']:.2f} | per author home {v['home_pa']:.2f} away {v['away_pa']:.2f}  " for y, v in byf[grp].items()))
json.dump(dict(by_year=rows, by_field=byf), open(os.path.join(HERE, "results", "home_and_away.json"), "w"), indent=1)

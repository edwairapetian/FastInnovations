"""Exploratory: net output cost of entering a field (papers over the year before and the two years
after the first paper there, against matched non-entry papers), by the growth of the destination
field and by its size.  Tests whether fast-growing fields are cheaper to enter.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "results")
ap = pd.read_parquet(os.path.join(ROOT, "data", "explore_author_panel.parquet")).sort_values(["key", "t"]).reset_index(drop=True)
meta = pd.read_parquet(os.path.join(ROOT, "data", "arxiv_meta.parquet"), columns=["primary", "v1"])
meta["year"] = meta.v1.dt.year
cy = meta.groupby(["primary", "year"]).size().rename("P").reset_index().sort_values(["primary", "year"])
cy["P1"] = cy.groupby("primary").P.shift(1)
cy["P4"] = cy.groupby("primary").P.shift(4)
cy["g3"] = np.log(cy.P1 / cy.P4) / 3
TMAX = float(ap.t.max())
aid = ap.key.astype("category").cat.codes.values.astype(np.int64)
comp = aid * 1e6 + ap.t.values
edges = np.arange(-720, 721, 90)
ev = ap[(ap.k >= 4) & (ap.t >= 760) & (ap.t <= TMAX - 730)].copy()
ev["entry"] = ev.new_cat & ~ev.inc_coauthor
rng = np.random.default_rng(1)
ev = ev[ev.entry | (~ev.new_cat & (rng.random(len(ev)) < 0.12))]
ev = ev.merge(cy[["primary", "year", "P1", "g3"]], on=["primary", "year"], how="left")
idx = ap.reset_index().merge(ev[["key", "pid"]], on=["key", "pid"])["index"].values
base = aid[idx] * 1e6 + ap.t.values[idx]
order = ap.iloc[idx][["key", "pid"]].reset_index(drop=True)
ev = order.merge(ev, on=["key", "pid"])
cnt = np.empty((len(ev), len(edges) - 1))
lo = np.searchsorted(comp, base + edges[0], side="left")
for j in range(len(edges) - 1):
    hi = np.searchsorted(comp, base + edges[j + 1], side="left")
    cnt[:, j] = hi - lo
    lo = hi
cnt[:, list(edges).index(0)] -= 1
ev["pre"] = np.minimum(cnt[:, :4].sum(axis=1), 12).astype(int)
ev["cell"] = ev.year.astype(str) + "_" + ev.pre.astype(str)
other = (~ev.entry).values
w_other = ev[other].groupby("cell").size()


def net(mask):
    share = ev[mask].groupby("cell").size() / mask.sum()
    wgt = ev.loc[other, "cell"].map(share / w_other).fillna(0).values
    c_other = (cnt[other] * wgt[:, None]).sum(axis=0) / wgt.sum()
    diff = cnt[mask].mean(axis=0) - c_other
    rate = cnt[mask][:, :4].sum(axis=1).mean() / 360 * 365
    # bootstrap-free standard error of the net loss from the entry sample (controls are many)
    tot = cnt[mask][:, 4:].sum(axis=1)
    se = tot.std() / np.sqrt(mask.sum()) / rate
    return dict(n=int(mask.sum()), net_years=float(diff[4:].sum() / rate), se=float(se),
                before=float(diff[4:8].sum() / rate), after=float(diff[8:].sum() / rate))


res = {}
e = ev.entry.values & ev.g3.notna().values & (ev.P1 >= 200).values
g = ev.g3.values
qs = np.nanquantile(g[e], [1 / 3, 2 / 3])
for lab, m in (("slow destination", e & (g <= qs[0])), ("middle", e & (g > qs[0]) & (g <= qs[1])), ("fast destination", e & (g > qs[1]))):
    res[lab] = net(m)
    res[lab]["mean_growth"] = float(np.nanmean(g[m]))
    print(lab, res[lab], flush=True)
sz = np.log(ev.P1.values)
qz = np.nanquantile(sz[e], [1 / 3, 2 / 3])
for lab, m in (("small destination", e & (sz <= qz[0])), ("medium", e & (sz > qz[0]) & (sz <= qz[1])), ("large destination", e & (sz > qz[1]))):
    res[lab] = net(m)
    res[lab]["mean_papers_per_year"] = float(np.exp(np.nanmean(sz[m])))
    print(lab, res[lab], flush=True)
for lab, m in (("far move", e & ev.new_group.values), ("near move", e & ~ev.new_group.values)):
    res[lab] = net(m)
    print(lab, res[lab], flush=True)
ng = ev.new_group.values
for dist, dm in (("far", ng), ("near", ~ng)):
    for lab, m in (("slow", g <= qs[0]), ("middle", (g > qs[0]) & (g <= qs[1])), ("fast", g > qs[1])):
        res[f"{dist} move, {lab} destination"] = net(e & dm & m)
        print(dist, lab, res[f"{dist} move, {lab} destination"], flush=True)
solo = (ev.n_authors.values == 1)
res["solo entry"] = net(e & solo)
print("solo", res["solo entry"], flush=True)
json.dump(res, open(os.path.join(OUT, "entry_cost_by_destination.json"), "w"), indent=1)

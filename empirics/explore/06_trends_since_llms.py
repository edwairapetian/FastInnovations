"""Exploratory: how have speed, solo work and movement across fields changed since language models?

January-September of each year, so that 2026 is comparable.  Uses the cached author panel.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ap = pd.read_parquet(os.path.join(ROOT, "data", "explore_author_panel.parquet"))
meta = pd.read_parquet(os.path.join(ROOT, "data", "arxiv_meta.parquet"), columns=["v1", "n_authors", "primary"])
meta["pid"] = np.arange(len(meta))
meta["month"] = meta.v1.dt.month
meta["year"] = meta.v1.dt.year
ap = ap.merge(meta[["pid", "month"]], on="pid")
js = ap[(ap.month <= 9) & (ap.year >= 2018)]
m = meta[(meta.month <= 9) & (meta.year >= 2018)]
rows = {}
for y, g in js.groupby("year"):
    est = g[g.k >= 4]                                   # established authors
    ent = est[est.new_cat]
    mp = m[m.year == y]
    rows[int(y)] = dict(
        solo_share_of_papers=float((mp.n_authors == 1).mean()),
        mean_team=float(mp.n_authors[mp.n_authors <= 30].mean()),
        entry_rate=float(est.new_cat.mean()),                              # share of established authors' papers that open a new field for them
        far_entry_rate=float((est.new_cat & est.new_group).mean()),
        entries_without_insider=float((~ent.inc_coauthor).mean()),
        entries_solo=float((ent.n_authors == 1).mean()),
        gap_entry_median=float(ent.gap.median()),
        gap_other_median=float(est[~est.new_cat].gap.median()),
        fields_per_author=float(est.groupby("key").primary.nunique().mean()),
        papers_per_author=float(est.groupby("key").size().mean()),
        n_entries=int(len(ent)))
out = pd.DataFrame(rows).T
pd.set_option("display.width", 250)
print(out.round(3).to_string())
json.dump(rows, open(os.path.join(HERE, "results", "trends_since_llms.json"), "w"), indent=1)

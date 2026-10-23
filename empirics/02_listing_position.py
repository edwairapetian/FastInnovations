"""Reconstruct arXiv daily announcement batches and each paper's position in its
primary category's "new submissions" listing.

arXiv assigns identifiers at announcement, in listing order. The version-1
timestamp is the submission time; papers that are held for moderation keep
their submission time but are announced later. So we
  1. assign each paper the deadline it was submitted against (14:00 ET on
     weekdays from 2017; 16:00 ET before), which gives the batch it would join
     absent a hold;
  2. take the smallest identifier of each such batch as the start of that
     announcement block, and place every paper in the block its identifier
     falls in;
  3. rank papers by identifier within block x primary category.
Output: data/listing.parquet.
"""
import numpy as np
import pandas as pd

m = pd.read_parquet("data/arxiv_meta.parquet",
                    columns=["id", "v1", "primary", "submitter", "n_authors"])
m = m[m.v1 >= "2015-01-05"].copy()
et = m.v1.dt.tz_convert("America/New_York")
cut = np.where(et < pd.Timestamp("2017-01-01", tz="America/New_York"), 960, 840)
mins = (et.dt.hour * 60 + et.dt.minute + et.dt.second / 60).to_numpy()
day = et.dt.tz_localize(None).dt.normalize()
dow = day.dt.dayofweek.to_numpy()
after = mins >= cut
dl = day + pd.to_timedelta(after.astype(int), unit="D")
d2 = dl.dt.dayofweek.to_numpy()
dl = dl + pd.to_timedelta(np.select([d2 == 5, d2 == 6], [2, 1], 0), unit="D")
m["deadline"] = dl.to_numpy()
m["cutoff_min"] = cut
m["weekday"] = dow
# signed minutes to the nearest weekday cutoff (same calendar day), weekdays only
m["run"] = np.where(dow < 5, mins - cut, np.nan)
m["tod"] = mins
m["idn"] = (m.id.str[:4].astype(np.int64) * 100000
            + m.id.str[5:].astype(np.int64))

starts = m.groupby("deadline").idn.min().sort_values()
mono = starts.index.to_series().is_monotonic_increasing
print("block starts monotone in deadline:", mono)
starts = starts[starts.index.to_series().diff().fillna(pd.Timedelta(1, "D")) > pd.Timedelta(0)]
blk = np.searchsorted(starts.to_numpy(), m.idn.to_numpy(), side="right") - 1
m["block"] = starts.index.to_numpy()[np.clip(blk, 0, None)]
m["held"] = m["block"] > m["deadline"]
m = m.sort_values("idn")
g = m.groupby(["block", "primary"])
m["pos"] = g.cumcount() + 1
m["n_list"] = g.idn.transform("size")
m["pos_frac"] = np.where(m.n_list > 1, (m.pos - 1) / (m.n_list - 1).clip(lower=1), 0.0)
m["n_block"] = m.groupby("block").idn.transform("size")
m.drop(columns=["v1"]).to_parquet("data/listing.parquet", index=False)
print(len(m), "held share", m.held.mean())
rk = m[~m.held].groupby("block").apply(lambda x: x.idn.rank().corr(x.tod.rank()
        + 0 * x.idn), include_groups=False)
print("rank corr id vs time-of-day (not held):", rk.describe().round(3).to_dict())

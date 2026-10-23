"""Population sample for the listing-rank design in high-energy physics.

All papers with a primary category covered by INSPIRE-HEP (hep-th, hep-ph,
hep-ex, hep-lat, gr-qc, nucl-th, nucl-ex, astro-ph.CO, astro-ph.HE), announced
in their regular batch (not held), deadlines 2017-01-09 to 2022-12-16 outside
holiday periods. For each paper: listing position, list size and the time
elapsed since the cutoff that opened its batch. Output: data/hep_sample.parquet.
"""
import numpy as np
import pandas as pd

HEP = ["hep-th", "hep-ph", "hep-ex", "hep-lat", "gr-qc", "nucl-th", "nucl-ex",
       "astro-ph.CO", "astro-ph.HE"]
L = pd.read_parquet("data/listing.parquet")
M = pd.read_parquet("data/arxiv_meta.parquet", columns=["id", "v1"])
L = L[L.primary.isin(HEP) & ~L.held & (L.deadline >= "2017-01-09")
      & (L.deadline <= "2022-12-16")].merge(M, on="id")
dl = pd.to_datetime(L.deadline)
md = dl.dt.month * 100 + dl.dt.day
first = pd.to_datetime(dl.dt.year.astype(str) + "-11-01")
thx = first + pd.to_timedelta((3 - first.dt.dayofweek) % 7 + 21, unit="D")
L = L[~((md >= 1217) | (md <= 108)) & ~((dl.dt.month == 11) & ((dl - thx).dt.days.abs() <= 4))].copy()
# the batch opens at the previous weekday's 14:00 ET cutoff
dl = pd.to_datetime(L.deadline)
prev = dl - pd.to_timedelta(np.where(dl.dt.dayofweek == 0, 3, 1), unit="D")
open_t = (prev + pd.Timedelta(hours=14)).dt.tz_localize("America/New_York", ambiguous="NaT",
                                                           nonexistent="shift_forward")
L["tsc"] = (L.v1.dt.tz_convert("America/New_York") - open_t).dt.total_seconds() / 60
L = L[L.tsc.notna() & (L.tsc >= 0)]
L["year"] = dl.loc[L.index].dt.year
L["batch_dow"] = pd.to_datetime(L.deadline).dt.dayofweek
L.drop(columns=["v1"]).to_parquet("data/hep_sample.parquet", index=False)
print(len(L), L.groupby("year").size().to_dict(), L.tsc.describe().round(0).to_dict())
open("data_raw/hep_all_ids.txt", "w").write("\n".join(L.id))

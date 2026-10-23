"""Regression-discontinuity sample around the daily submission cutoff.

Weekday submissions within 180 minutes of the cutoff, deadlines from
2017-01-09 (first full week of the 14:00 ET cutoff) to 2022-12-16, excluding
holiday periods when announcements are skipped or merged (Dec 17-Jan 8 and the
week of US Thanksgiving). Output: data/rd_sample.parquet.
"""
import numpy as np
import pandas as pd

L = pd.read_parquet("data/listing.parquet")
L = L[(L.weekday < 5) & (L.run.abs() <= 180)
      & (L.deadline >= "2017-01-09") & (L.deadline <= "2022-12-16")].copy()
dl = pd.to_datetime(L.deadline)
md = dl.dt.month * 100 + dl.dt.day
xmas = (md >= 1217) | (md <= 108)
# US Thanksgiving: 4th Thursday of November; drop Mon-Fri of that week
nov = dl.dt.month == 11
first = pd.to_datetime(dl.dt.year.astype(str) + "-11-01")
thx = first + pd.to_timedelta((3 - first.dt.dayofweek) % 7 + 21, unit="D")
thxweek = nov & ((dl - thx).dt.days.abs() <= 4)
L = L[~xmas & ~thxweek]
L["after"] = (L.run >= 0).astype(int)
L["year"] = dl.dt.year
L.to_parquet("data/rd_sample.parquet", index=False)
print(len(L), L.groupby("year").size().to_dict())
print("within 120:", (L.run.abs() <= 120).sum(), "within 60:", (L.run.abs() <= 60).sum())

"""Later overlapping papers for the HEP population sample, at a low similarity
threshold (the median nearest-neighbour similarity, tau50), so that overlap by
other authors is common enough to study. Outputs data/hep_pairs.parquet,
data/hep_tau.parquet and the list of later papers for INSPIRE."""
import sys

import pandas as pd

sys.path.insert(0, "empirics")
from neardup import run

S = pd.read_parquet("data/hep_sample.parquet", columns=["id", "primary"])
P, T = run(set(S.id), cats=set(S.primary.unique()), keep_q=50)
P.to_parquet("data/hep_pairs.parquet", index=False)
T.to_parquet("data/hep_tau.parquet", index=False)
open("data_raw/hep_pairs_j.txt", "w").write("\n".join(sorted(set(P.j))))
print(len(P), "pairs;", P.j.nunique(), "later papers")

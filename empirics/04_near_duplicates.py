"""Later near-duplicates of each paper in the RD sample.

Within each primary category, papers are represented by TF-IDF vectors of title
and abstract (vocabulary fitted on the category's 2016H2-2024 papers). For a
random sample of papers we compute the cosine similarity to the nearest earlier
paper in the category within 24 months; its 99th and 95th percentiles define
the category's overlap thresholds tau99 and tau95, as in the overlap flag of
the descriptive evidence. For every focal paper i we record each later paper j
in the category, submitted within 730 days, with cosine >= tau95.

Outputs: data/nd_pairs.parquet (i, j, sim, dt_days, tau99, tau95),
         data/nd_tau.parquet  (category thresholds).
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

WIN = np.timedelta64(730, "D")
rng = np.random.default_rng(0)
meta = pd.read_parquet("data/arxiv_meta.parquet", columns=["id", "v1", "primary"])
meta = meta[(meta.v1 >= "2016-07-01") & (meta.v1 < "2025-01-01")]
text = pd.read_parquet("data/arxiv_text.parquet")
df = meta.merge(text, on="id")
del text
focal = set(pd.read_parquet("data/rd_sample.parquet", columns=["id"]).id)
pairs, taus = [], []
cats = df.primary.value_counts()
for c, nc in cats.items():
    d = df[df.primary == c].sort_values("v1").reset_index(drop=True)
    isf = d.id.isin(focal).to_numpy()
    if isf.sum() == 0 or nc < 200:
        continue
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.3,
                          stop_words="english", dtype=np.float32)
    X = vec.fit_transform(d.title + " " + d.abstract).tocsr()
    t = d.v1.dt.tz_localize(None).to_numpy()
    # thresholds from nearest earlier neighbour similarity
    elig = np.where(t >= np.datetime64("2018-01-01"))[0]
    samp = rng.choice(elig, size=min(3000, len(elig)), replace=False) if len(elig) else []
    nn = []
    for s in np.array_split(np.sort(samp), max(1, len(samp) // 300)):
        if len(s) == 0:
            continue
        lo = np.searchsorted(t, t[s[0]] - WIN)
        hi = s[-1]
        if hi <= lo:
            continue
        S = (X[s] @ X[lo:hi].T).toarray()
        tt = t[lo:hi]
        mask = (tt[None, :] < t[s][:, None]) & (tt[None, :] >= (t[s] - WIN)[:, None])
        S[~mask] = 0
        nn.append(S.max(axis=1))
    nn = np.concatenate(nn) if nn else np.array([np.nan])
    tau99, tau95 = np.nanquantile(nn, 0.99), np.nanquantile(nn, 0.95)
    taus.append(dict(primary=c, n=nc, tau99=tau99, tau95=tau95, tau50=np.nanmedian(nn)))
    fi = np.where(isf)[0]
    for b in np.array_split(fi, max(1, len(fi) // 400)):
        lo = b[0] + 1
        hi = np.searchsorted(t, t[b[-1]] + WIN, side="right")
        if hi <= lo:
            continue
        S = (X[b] @ X[lo:hi].T).toarray()
        tt = t[lo:hi]
        mask = (tt[None, :] > t[b][:, None]) & (tt[None, :] <= (t[b] + WIN)[:, None])
        S[~mask] = 0
        r, k = np.nonzero(S >= tau95)
        if len(r):
            pairs.append(pd.DataFrame({
                "i": d.id.to_numpy()[b[r]], "j": d.id.to_numpy()[lo + k],
                "sim": S[r, k],
                "dt_days": ((tt[k] - t[b[r]]) / np.timedelta64(1, "D")).astype(np.float32),
                "primary": c}))
    print(c, nc, int(isf.sum()), round(tau99, 3), round(tau95, 3), flush=True)
P = pd.concat(pairs, ignore_index=True)
T = pd.DataFrame(taus)
P = P.merge(T[["primary", "tau99", "tau95"]], on="primary")
P.to_parquet("data/nd_pairs.parquet", index=False)
T.to_parquet("data/nd_tau.parquet", index=False)
print(len(P), "pairs;", (P.sim >= P.tau99).sum(), "above tau99")

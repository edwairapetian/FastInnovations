"""Later overlapping papers by TF-IDF similarity within a primary category.

For a category, papers are TF-IDF vectors of title and abstract (vocabulary
fitted on the category's 2016H2-2024 papers). Thresholds come from the
distribution of each paper's similarity to its nearest earlier paper in the
category within 24 months (a random sample of 3,000 papers from 2018 on):
tau_q is its q-th percentile, so tau99 flags the closest 1% of papers. For each
focal paper we keep every later paper in the category, within 730 days, whose
similarity is at least tau_keep.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

WIN = np.timedelta64(730, "D")
QS = (50, 75, 90, 95, 99)


def load_corpus():
    meta = pd.read_parquet("data/arxiv_meta.parquet", columns=["id", "v1", "primary"])
    meta = meta[(meta.v1 >= "2016-07-01") & (meta.v1 < "2025-01-01")]
    text = pd.read_parquet("data/arxiv_text.parquet")
    return meta.merge(text, on="id")


def category_pairs(d, focal, keep_q=95, seed=0):
    """d: one category's papers (id, v1, title, abstract). Returns (pairs, taus)."""
    rng = np.random.default_rng(seed)
    d = d.sort_values("v1").reset_index(drop=True)
    isf = d.id.isin(focal).to_numpy()
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.3,
                          stop_words="english", dtype=np.float32)
    X = vec.fit_transform(d.title + " " + d.abstract).tocsr()
    t = d.v1.dt.tz_localize(None).to_numpy()
    elig = np.where(t >= np.datetime64("2018-01-01"))[0]
    samp = np.sort(rng.choice(elig, size=min(3000, len(elig)), replace=False))
    nn = []
    for s in np.array_split(samp, max(1, len(samp) // 300)):
        lo = np.searchsorted(t, t[s[0]] - WIN)
        hi = s[-1]
        if len(s) == 0 or hi <= lo:
            continue
        S = (X[s] @ X[lo:hi].T).toarray()
        tt = t[lo:hi]
        S[~((tt[None, :] < t[s][:, None]) & (tt[None, :] >= (t[s] - WIN)[:, None]))] = 0
        nn.append(S.max(axis=1))
    nn = np.concatenate(nn)
    taus = {f"tau{q}": float(np.quantile(nn, q / 100)) for q in QS}
    keep = taus[f"tau{keep_q}"]
    out = []
    fi = np.where(isf)[0]
    for b in np.array_split(fi, max(1, len(fi) // 400)):
        if len(b) == 0:
            continue
        lo = b[0] + 1
        hi = np.searchsorted(t, t[b[-1]] + WIN, side="right")
        if hi <= lo:
            continue
        S = (X[b] @ X[lo:hi].T).toarray()
        tt = t[lo:hi]
        S[~((tt[None, :] > t[b][:, None]) & (tt[None, :] <= (t[b] + WIN)[:, None]))] = 0
        r, k = np.nonzero(S >= keep)
        if len(r):
            out.append(pd.DataFrame({
                "i": d.id.to_numpy()[b[r]], "j": d.id.to_numpy()[lo + k],
                "sim": S[r, k],
                "dt_days": ((tt[k] - t[b[r]]) / np.timedelta64(1, "D")).astype(np.float32)}))
    P = pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["i", "j", "sim", "dt_days"])
    return P, taus


def run(focal, cats=None, keep_q=95, min_n=200):
    df = load_corpus()
    pairs, taus = [], []
    for c, nc in df.primary.value_counts().items():
        if (cats is not None and c not in cats) or nc < min_n:
            continue
        d = df[df.primary == c]
        if not d.id.isin(focal).any():
            continue
        P, tau = category_pairs(d, focal, keep_q)
        P["primary"] = c
        pairs.append(P)
        taus.append(dict(primary=c, n=int(nc), **tau))
        print(c, nc, len(P), {k: round(v, 3) for k, v in tau.items()}, flush=True)
    T = pd.DataFrame(taus)
    P = pd.concat(pairs, ignore_index=True).merge(T.drop(columns="n"), on="primary")
    return P, T

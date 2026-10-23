"""Parse the arXiv metadata snapshot (Kaggle Cornell-University/arxiv, file dated
2026-10-18) into two parquet files for papers with new-style ids from 2015 on.

data/arxiv_meta.parquet  : id, yymm, submitter, n_authors, authors ("Last, First"
                            joined by ";"), primary, categories, title, comments,
                            v1 (UTC submission time of version 1), journal_ref,
                            doi, n_versions
data/arxiv_text.parquet  : id, title, abstract (for text similarity)

The primary category is the first entry of `categories`, as in the OAI feed.
"""
import json
import sys

import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "data_raw/arxiv-metadata-oai-snapshot.json"
meta, text = [], []
with open(SRC) as f:
    for line in f:
        d = json.loads(line)
        pid = d["id"]
        if "/" in pid or int(pid[:4]) < 1501:
            continue
        cats = d.get("categories") or ""
        vers = d.get("versions") or []
        ap = d.get("authors_parsed") or []
        title = " ".join((d.get("title") or "").split())
        meta.append(dict(
            id=pid, yymm=int(pid[:4]), submitter=d.get("submitter"),
            n_authors=len(ap),
            authors=";".join(", ".join(x for x in a[:2] if x) for a in ap),
            primary=cats.split()[0] if cats else None, categories=cats,
            title=title, comments=d.get("comments"),
            v1=vers[0]["created"] if vers else None,
            journal_ref=d.get("journal-ref"), doi=d.get("doi"),
            n_versions=len(vers)))
        text.append(dict(id=pid, title=title,
                         abstract=" ".join((d.get("abstract") or "").split())))
m = pd.DataFrame(meta)
m["v1"] = pd.to_datetime(m["v1"], format="%a, %d %b %Y %H:%M:%S %Z", utc=True)
m.to_parquet("data/arxiv_meta.parquet", index=False)
pd.DataFrame(text).to_parquet("data/arxiv_text.parquet", index=False)
print(len(m), m["v1"].min(), m["v1"].max())
print(m.groupby(m["v1"].dt.year).size())

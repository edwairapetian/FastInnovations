"""Fetch Semantic Scholar records for arXiv ids with the public batch endpoint.

usage: python empirics/s2_fetch.py IDS.txt OUT.jsonl FIELDS
Ids already in OUT are skipped, so the script can be restarted. Requests that
are rate limited (HTTP 429) are retried with exponential backoff.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

ids_path, out_path, fields = sys.argv[1], sys.argv[2], sys.argv[3]
URL = "https://api.semanticscholar.org/graph/v1/paper/batch?fields=" + fields
ids = [x.strip() for x in open(ids_path) if x.strip()]
done = set()
if os.path.exists(out_path):
    for line in open(out_path):
        done.add(json.loads(line)["arxiv"])
todo = [i for i in ids if i not in done]
print(len(ids), "ids;", len(todo), "to fetch", flush=True)
B = 500
with open(out_path, "a") as out:
    for k in range(0, len(todo), B):
        chunk = todo[k:k + B]
        body = json.dumps({"ids": ["ARXIV:" + i for i in chunk]}).encode()
        wait = 2
        while True:
            req = urllib.request.Request(URL, data=body,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    res = json.load(r)
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(wait)
                    wait = min(wait * 2, 64)
                    continue
                raise
            except Exception:
                time.sleep(wait)
                wait = min(wait * 2, 64)
        for i, rec in zip(chunk, res):
            out.write(json.dumps({"arxiv": i, "rec": rec}) + "\n")
        out.flush()
        print(k + len(chunk), "/", len(todo), flush=True)
        time.sleep(1)

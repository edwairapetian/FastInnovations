"""Fetch INSPIRE-HEP records (record id, citation counts, reference record ids)
for arXiv ids. usage: python empirics/inspire_fetch.py IDS.txt OUT.jsonl [refs]
Restartable; INSPIRE allows 15 requests per 5 seconds.
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

ids_path, out_path = sys.argv[1], sys.argv[2]
want_refs = len(sys.argv) > 3
fields = "control_number,arxiv_eprints.value,citation_count,citation_count_without_self_citations"
if want_refs:
    fields += ",references.record"
ids = [x.strip() for x in open(ids_path) if x.strip()]
done = set()
if os.path.exists(out_path):
    for line in open(out_path):
        done.add(json.loads(line)["arxiv"])
todo = [i for i in ids if i not in done]
print(len(ids), "ids;", len(todo), "to fetch", flush=True)
B = 40
with open(out_path, "a") as out:
    for k in range(0, len(todo), B):
        chunk = todo[k:k + B]
        q = " or ".join(f"arxiv:{i}" for i in chunk)
        url = ("https://inspirehep.net/api/literature?size=100&fields=" + fields
               + "&q=" + urllib.parse.quote(q))
        wait = 2
        while True:
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    res = json.load(r)
                break
            except Exception as e:
                time.sleep(wait)
                wait = min(2 * wait, 60)
        found = set()
        for h in res["hits"]["hits"]:
            m = h["metadata"]
            for e in m.get("arxiv_eprints", []):
                a = e.get("value")
                if a in chunk and a not in found:
                    found.add(a)
                    rec = {"arxiv": a, "recid": m.get("control_number"),
                           "cc": m.get("citation_count"),
                           "cc_noself": m.get("citation_count_without_self_citations")}
                    if want_refs:
                        rec["refs"] = [int(r["record"]["$ref"].rsplit("/", 1)[1])
                                       for r in m.get("references", []) if "record" in r]
                    out.write(json.dumps(rec) + "\n")
        for a in chunk:
            if a not in found:
                out.write(json.dumps({"arxiv": a, "recid": None}) + "\n")
        out.flush()
        if (k // B) % 25 == 0:
            print(k + len(chunk), "/", len(todo), flush=True)
        time.sleep(0.4)

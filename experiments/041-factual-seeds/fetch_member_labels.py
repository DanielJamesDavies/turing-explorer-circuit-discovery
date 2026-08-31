"""Neuronpedia labels for every member of the concept-run circuits,
cached to labels.jsonl (resumable; misses cached too).

  python3 fetch_member_labels.py
"""
import json
import os
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
GEMMA = HERE.parent / "038-transcoder-compare-gemma"
CACHE = HERE / "labels.jsonl"
URL = ("https://www.neuronpedia.org/api/feature/gemma-2-2b/"
       "%d-gemmascope-transcoder-16k/%d")

want = set()
for line in open(GEMMA / os.environ.get("MEMFILE","ours_gtc_fact_members.jsonl")):
    r = json.loads(line)
    if r["arm"] != "triamp400":
        continue
    want.add((int(r["layer"]), int(r["latent"])))
    for l, d in r["members"].items():
        want |= {(int(l), int(f)) for f in d}

have = set()
if CACHE.exists():
    for line in open(CACHE):
        r = json.loads(line)
        have.add((r["layer"], r["feat"]))

todo = sorted(want - have)
print("%d wanted, %d cached, %d to fetch" % (len(want), len(have), len(todo)),
      flush=True)
with open(CACHE, "a") as out:
    for i, (l, f) in enumerate(todo):
        label = "(fetch failed)"
        try:
            with urllib.request.urlopen(URL % (l, f), timeout=25) as resp:
                d = json.load(resp)
            e = d.get("explanations") or []
            label = (e[0].get("description", "").strip()
                     if e else "(no explanation)")
        except Exception as ex:
            label = "(fetch failed: %s)" % type(ex).__name__
        out.write(json.dumps({"layer": l, "feat": f, "label": label}) + "\n")
        out.flush()
        if (i + 1) % 40 == 0:
            print("  %d/%d" % (i + 1, len(todo)), flush=True)
        time.sleep(0.35)
print("DONE", flush=True)

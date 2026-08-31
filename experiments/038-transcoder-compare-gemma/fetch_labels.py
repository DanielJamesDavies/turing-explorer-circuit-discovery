"""Fetch Neuronpedia auto-interp labels for every feature in our six
Gemma tri-amp circuits (plus the seed features themselves), cached to
neuronpedia_labels.jsonl -- one {layer, feat, label} row per feature,
"(no explanation)" recorded so misses are cached too. Polite: 0.4 s
between requests, resumable (reruns skip cached rows).

  python3 fetch_labels.py     # needs network; ~300 requests first run
"""
import json
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
CACHE = HERE / "neuronpedia_labels.jsonl"
URL = ("https://www.neuronpedia.org/api/feature/gemma-2-2b/"
       "%d-gemmascope-transcoder-16k/%d")

want = set()
sizes = {}
for line in open(HERE / "ours_gtc_members.jsonl"):
    r = json.loads(line)
    if r["arm"] != "triamp400":
        continue
    key = (int(r["layer"]), int(r["latent"]))
    want.add(key)                                          # the seed itself
    mem = {(int(l), int(f)) for l, d in r["members"].items() for f in d}
    want |= mem
    sizes[key] = len(mem)

# ALSO their matched-size head, so the echo-share control has a base rate
# computed on the same label source rather than on our members only.
for line in open(HERE / "theirs_gtc_nodes.jsonl"):
    r = json.loads(line)
    key = (int(r["layer"]), int(r["latent"]))
    if key in sizes:
        want |= {(int(l), int(f))
                 for l, f, *_ in r["ranking"][:sizes[key]]}

have = set()
if CACHE.exists():
    for line in open(CACHE):
        r = json.loads(line)
        have.add((r["layer"], r["feat"]))

todo = sorted(want - have)
print("%d features wanted, %d cached, %d to fetch" %
      (len(want), len(have), len(todo)), flush=True)
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
        if (i + 1) % 25 == 0:
            print("  %d/%d" % (i + 1, len(todo)), flush=True)
        time.sleep(0.4)
print("DONE: %d rows total" % (len(have) + len(todo)), flush=True)

"""Targeted Neuronpedia search for SPECIFIC concept features -- the
"general relativity -> Einstein" test. Unlike find_factual.py (broad
knowledge categories), this asks whether a 16k-wide transcoder even
carries features for named entities and specific theories, at a range
of depths.

  python3 find_concept.py "<query>" [more queries...]   (default set below)
"""
import json
import os
import sys
import time
import urllib.request

API = "https://www.neuronpedia.org/api/explanation/search"
LAYERS = [int(x) for x in os.environ.get("LAYERS", "6,8,10,12,14").split(",")]
DEFAULT = [
    "Albert Einstein",
    "theory of relativity and spacetime",
    "physicists and physics theories",
    "quantum mechanics",
    "famous scientists names",
    "the year 1905 and early twentieth century",
    "Nobel Prize winners",
    "mathematical physics equations",
]


def search(layer, q):
    body = json.dumps({"modelId": "gemma-2-2b",
                       "layers": ["%d-gemmascope-transcoder-16k" % layer],
                       "query": q}).encode()
    req = urllib.request.Request(API, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.load(r).get("results", [])


def main():
    queries = sys.argv[1:] or DEFAULT
    rows, seen = [], set()
    for q in queries:
        print("\n### %s" % q, flush=True)
        for layer in LAYERS:
            try:
                res = search(layer, q)
            except Exception as e:
                print("  ! L%d %s" % (layer, type(e).__name__), flush=True)
                continue
            for x in res[:6]:
                idx = int(x["index"])
                lab = (x.get("description") or "").strip()
                if (layer, idx) not in seen:
                    seen.add((layer, idx))
                    rows.append({"layer": layer, "feat": idx,
                                 "label": lab, "query": q})
                print("  L%-3d %-6d %s" % (layer, idx, lab[:88]), flush=True)
            time.sleep(0.4)
    with open(os.environ.get("OUT_FILE", "concept_candidates.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("\n%d unique -> concept_candidates.jsonl" % len(rows))


if __name__ == "__main__":
    main()

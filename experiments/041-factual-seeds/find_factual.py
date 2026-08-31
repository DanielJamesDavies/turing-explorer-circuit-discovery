"""Search Neuronpedia for FACTUAL/INFORMATIONAL GemmaScope transcoder
features -- candidates for a knowledge-extraction circuit, as opposed
to the syntactic/lexical seeds the activation-density scan happened to
pick.

Queries cover several knowledge kinds (dates, wars, places, science,
people, institutions). Candidates are written to candidates.jsonl for
the activation screen (screen_factual.py) to filter down to features
that actually fire on our cached wikitext windows.

  python3 find_factual.py    # network; writes candidates.jsonl
"""
import json
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
API = "https://www.neuronpedia.org/api/explanation/search"
# Deep enough that a circuit has room to assemble a fact, shallow enough
# that fitting stays affordable (UP = every transcoder layer below).
LAYERS = [8, 10, 12]
QUERIES = [
    "specific years and historical dates",
    "historical events and wars",
    "names of countries and capital cities",
    "scientific units and measurements",
    "famous people and historical figures",
    "geographic locations, rivers and mountains",
    "chemical elements and compounds",
    "sports teams and competitions",
    "musical works, composers and albums",
    "government institutions and legal bodies",
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
    seen, rows = set(), []
    for layer in LAYERS:
        for q in QUERIES:
            try:
                res = search(layer, q)
            except Exception as e:
                print("  ! %d %s: %s" % (layer, q[:28], type(e).__name__),
                      flush=True)
                continue
            for x in res:
                idx = int(x.get("index"))
                key = (layer, idx)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"layer": layer, "feat": idx,
                             "label": (x.get("description") or "").strip(),
                             "query": q})
            print("L%-3d %-42s %d results" % (layer, q[:42], len(res)),
                  flush=True)
            time.sleep(0.5)
    with open(HERE / "candidates.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("\n%d unique candidates -> candidates.jsonl" % len(rows))


if __name__ == "__main__":
    main()

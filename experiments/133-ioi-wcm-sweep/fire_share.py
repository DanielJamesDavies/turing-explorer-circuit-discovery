"""133 (CPU): which IOI circuit members never fire on the held-out (disjoint-name) items? From 081's member profiles
(engine.profile: first 256 held-out items, n_fire = items with the member active at a real token).

  .venv/bin/python experiments/133-ioi-wcm-sweep/fire_share.py [profiles.jsonl.gz ...]
"""
import gzip
import json
import sys
from collections import defaultdict

paths = sys.argv[1:] or ["experiments/081-task-circuits/out_v2/gpt2-small/profiles.jsonl.gz",
                         "experiments/081-task-circuits/out_v4/gpt2-small/profiles.jsonl.gz"]
for p in paths:
    for line in gzip.open(p, "rt"):
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if r.get("task") != "ioi":
            continue
        M = r["members"]
        n = len(M)
        dead = [m for m in M if m["n_fire"] == 0]
        by = defaultdict(lambda: [0, 0])
        for m in M:
            k = "L%02d-%s" % (m["layer"] // 4 * 4, m["kind"])
            by[m["kind"]][0] += 1
            by[m["kind"]][1] += m["n_fire"] == 0
        print("%s %s lam %s: %d members, %.1f%% never fire on 256 held-out items | by kind: %s" % (
            p.split("/")[-3], r["arm"], r["lam"], n, 100.0 * len(dead) / max(n, 1),
            ", ".join("%s %d (%.0f%% dead)" % (k, v[0], 100.0 * v[1] / max(v[0], 1)) for k, v in sorted(by.items()))))

"""Print the field layout of 081's task files (read-only helper)."""
import json
from pathlib import Path

T = Path(__file__).resolve().parents[1] / "081-task-circuits" / "tasks" / "gpt2-small"
for n in ("ioi", "greater_than"):
    t = json.load(open(T / ("%s.json" % n), encoding="utf-8"))
    print(n, list(t.keys()), {k: len(v) for k, v in t["split"].items()})
    r = t["rows"][0]
    print({k: (v if len(str(v)) < 200 else str(v)[:200]) for k, v in r.items()})

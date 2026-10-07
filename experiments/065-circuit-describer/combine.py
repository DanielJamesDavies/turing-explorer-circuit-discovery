"""Merge describe.py run files into results/descriptions.jsonl, the one file other steps read.

Each describe.py run writes only its own results/runs/<time>_<pid>.jsonl, so runs never share a file. This merges the
current descriptions.jsonl with every run file: one record per circuit, the latest successful description (a failed
call is kept only when a circuit has no success yet, so it shows up as still to do). The new file is written to a
temporary file and swapped in with an atomic replace, so a reader never sees a half-written file. Run files are kept
as the history; re-running combine is safe.

    python experiments/065-circuit-describer/combine.py
"""
from __future__ import annotations

import collections
import json
import os
import sys

from describe import COMBINED, RUNS, read_records


def better(a: dict | None, b: dict) -> bool:
    """Whether record b should replace a: a success beats a failure; otherwise the later one wins."""
    if a is None:
        return True
    if bool(b.get("result")) != bool(a.get("result")):
        return bool(b.get("result"))
    return b.get("time", "") >= a.get("time", "")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sources = ([COMBINED] if COMBINED.exists() else []) + sorted(RUNS.glob("*.jsonl"))
    best: dict[str, dict] = {}
    n_in = 0
    for path in sources:
        for rec in read_records(path):
            n_in += 1
            if better(best.get(rec["key"]), rec):
                best[rec["key"]] = rec
    tmp = COMBINED.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for key in sorted(best):
            fh.write(json.dumps(best[key], ensure_ascii=False) + "\n")
    os.replace(tmp, COMBINED)
    ok = [r for r in best.values() if r.get("result")]
    print(f"{len(sources)} files, {n_in} records -> {len(best)} circuits "
          f"({len(ok)} described, {len(best) - len(ok)} still failed) in {COMBINED.name}")
    print("prompt versions:", dict(collections.Counter(r.get("prompt_version") for r in ok)))


if __name__ == "__main__":
    main()

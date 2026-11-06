"""T2's verify gate and quarantine (Daniel, 2026-10-26), from the verify rows build_records.py MODE=verify writes.

  gate    the EARLY gate: every target of the gate list (targets_tl8k_gate64.txt: 64 spread over the 35 cells, verified
          before the rest of the 8k is built) must have an ok verify row. Any failure or missing row: exit 1, and the
          chain stops the TuringLLM track (a builder bug costs minutes, not the full build).
  apply   past the gate nothing stops: a target of the full list whose verify row is not ok (failed, error, missing) is
          QUARANTINED. Writes into OUT:
            quarantine.tsv        key, reason (the first failures), must-include subsets it belongs to
            targets_tl8k.txt      the full list minus the quarantined, list order (every later stage's 8k list)
            targets_tl8k_<n>.txt  each subset minus the quarantined (sweep190, nulls126, case6, gallery9, stable128, ...)
            quarantine.json       counts, the must-include hits, the md5 of every derived list
          and prints the count, flagging LOUDLY every quarantined must-include target (sweep / nulls / case / gallery).
          A gate target that is not ok in the full verify is still a failure (exit 1): the gate's promise is broken.

  python experiments/094-tl-v2-records/quarantine.py gate  --verify DIR --gate LIST
  python experiments/094-tl-v2-records/quarantine.py apply --verify DIR --targets LIST --out OUT [--gate LIST]
         [--subset NAME=LIST ...]
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

MUST = ("sweep190", "nulls126", "case6", "gallery9")      # downstream stages reference these by name


def read_list(p):
    return [t for t in Path(p).read_text().split() if t]


def verify_rows(d):
    """{seed: row} over DIR/verify.shard*.jsonl (the last row per seed wins; truncated lines skipped)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(str(d), "verify.shard*.jsonl"))):
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if "seed" in r:
                out[r["seed"]] = r
    return out


def reason(r):
    if r is None:
        return "missing: no verify row"
    if r.get("missing"):
        return "missing: no record"
    if r.get("ok"):
        return None
    f = r.get("failures") or ["not ok"]
    return "; ".join(str(x) for x in f[:3])[:300]


def gate(args):
    keys = read_list(args.gate)
    rows = verify_rows(args.verify)
    bad = [(k, reason(rows.get(k))) for k in keys if reason(rows.get(k)) is not None]
    print("  T2 early gate: %d targets, %d ok, %d failed or missing" % (len(keys), len(keys) - len(bad), len(bad)))
    for k, why in bad[:20]:
        print("    GATE FAIL %s: %s" % (k, why))
    return 1 if bad or not keys else 0


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def apply(args):
    keys = read_list(args.targets)
    rows = verify_rows(args.verify)
    subsets = OrderedDict()
    for s in args.subset or []:
        name, _, path = s.partition("=")
        subsets[name] = read_list(path)
    q = OrderedDict((k, reason(rows.get(k))) for k in keys if reason(rows.get(k)) is not None)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    member = {k: [n for n, ks in subsets.items() if k in set(ks)] for k in q}
    with open(out / "quarantine.tsv.tmp", "w") as fh:
        fh.write("key\treason\tsubsets\n")
        for k, why in q.items():
            fh.write("%s\t%s\t%s\n" % (k, why.replace("\t", " "), ",".join(member[k])))
    os.replace(out / "quarantine.tsv.tmp", out / "quarantine.tsv")
    files = OrderedDict()
    for name, ks in [("targets_tl8k.txt", keys)] + [("targets_tl8k_%s.txt" % n, v) for n, v in subsets.items()]:
        p = out / name
        p.with_suffix(".tmp").write_text("".join("%s\n" % k for k in ks if k not in q))
        os.replace(p.with_suffix(".tmp"), p)
        files[name] = dict(n=sum(k not in q for k in ks), of=len(ks), md5=md5(p))
    hits = OrderedDict((n, [k for k in subsets[n] if k in q]) for n in subsets if n in MUST)
    gate_bad = [k for k in (read_list(args.gate) if args.gate else []) if k in q]
    summ = OrderedDict(n_targets=len(keys), n_verify_rows=len(rows), n_quarantined=len(q),
                       quarantined=list(q), must_include_hits=hits, gate_targets_quarantined=gate_bad, files=files)
    (out / "quarantine.json").write_text(json.dumps(summ, indent=1) + "\n")
    print("  T2 quarantine: %d of %d targets quarantined (verify failed or missing); every later stage reads %s"
          % (len(q), len(keys), out / "targets_tl8k.txt"))
    for k, why in list(q.items())[:20]:
        print("    quarantined %s: %s%s" % (k, why, (" [MUST-INCLUDE: %s]" % ",".join(m for m in member[k] if m in MUST))
                                           if any(m in MUST for m in member[k]) else ""))
    if len(q) > 20:
        print("    ... %d more in %s" % (len(q) - 20, out / "quarantine.tsv"))
    for n, ks in hits.items():
        if ks:
            print("  !!! WARNING: %d MUST-INCLUDE target(s) of %s quarantined: %s (the %s stage runs without them)"
                  % (len(ks), n, " ".join(ks), n))
    if gate_bad:
        print("  T2: gate targets not ok in the full verify: %s" % " ".join(gate_bad))
        return 1
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("gate", "apply"))
    ap.add_argument("--verify", required=True)
    ap.add_argument("--gate")
    ap.add_argument("--targets")
    ap.add_argument("--out")
    ap.add_argument("--subset", action="append")
    a = ap.parse_args(argv)
    if a.mode == "gate":
        if not a.gate:
            ap.error("gate needs --gate")
        return gate(a)
    if not (a.targets and a.out):
        ap.error("apply needs --targets and --out")
    return apply(a)


if __name__ == "__main__":
    sys.exit(main())

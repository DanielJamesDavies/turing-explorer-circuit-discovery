"""Experiment 081, stage 0 (CPU, tokenizer only; no model, no download): build every task's prompts for MODEL.

  originals   originals.GENERATORS (IOI, greater-than, the four SFC agreement sets, gendered pronoun, ROME
              known_1000), from data/external/<source>/ (each with a SOURCE.md)
  induction   tasks.induction (generated: repeated random tokens)
  BLiMP       blimp.survey over every local subset under MODEL's tokenizer, blimp.select (capped), one task each;
              skipped with a message when data/external/blimp is absent (NO_BLIMP=1 skips it on purpose)
Full datasets, natural split (2026-10-21): SFC's own train / test files (SFC_SPLIT=official), 75 / 25 by group for the
fixed datasets without one (BLiMP subsets, ROME known_1000, Mathwin), and N_TRAIN / N_HELD items from disjoint list
halves for the generated tasks (IOI, greater-than, induction); every held-out set capped at HELDOUT_CAP.
Writes tasks/<MODEL>/<name>.json, summary.json and blimp_survey.json; prints the table. Deterministic.

  MODEL=gpt2-small PYTHONPATH=src python -X utf8 experiments/081-task-circuits/make_tasks.py
  env: MODEL (gpt2-small)  TASKS (comma list; default all)  N_TRAIN (1600) / N_HELD (5000) (generated tasks)
       HELDOUT_CAP (5000; 0 = no cap)  SFC_SPLIT (official | prefix)  NO_BLIMP=1  BLIMP  BLIMP_QUAL_MIN (0.8)
       OUT_TASKS (tasks/<MODEL>)
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blimp as BL  # noqa: E402
import originals as O  # noqa: E402
import task_common as C  # noqa: E402
import tasks as T  # noqa: E402

OUT = Path(os.environ.get("OUT_TASKS", str(C.TASK_DIR)))
N_TRAIN = int(os.environ.get("N_TRAIN", C.N_TRAIN_GEN))
N_HELD = int(os.environ.get("N_HELD", C.N_HELD_GEN))
GENERATORS = dict(O.GENERATORS, induction=T.induction)
GENERATED = ("ioi", "greater_than", "induction")          # tasks generated from lists: sizes are chosen, not read
WANT = [t for t in os.environ.get("TASKS", ",".join(GENERATORS)).split(",") if t]


def write(task):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("%s.json" % task["name"])).write_text(json.dumps(task, separators=(",", ":")) + "\n")


def summary_row(t):
    lens = [len(r["tokens"]) for r in t["rows"]]
    c = t["counts"]
    drops = {k: v for k, v in c.items() if (k.startswith("dropped_") or k.startswith("not_qualifying_")) and v}
    sm = t["split_meta"]
    return dict(name=t["name"], family=t["family"], kept=c["kept"], train=len(t["split"]["train"]),
                held=len(t["split"]["held"]), held_uncapped=sm["n_held_uncapped"], held_cap=sm["held_cap"],
                split_rule=sm["rule"], overlap=t.get("overlap"), lists_split=t.get("lists_split"),
                groups=t["n_groups"], len_min=min(lens) if lens else None,
                len_max=max(lens) if lens else None, answer_mode=t["answer_mode"], contrast=t["contrast"],
                multi_good=c.get("multi_token_good", 0), multi_bad=c.get("multi_token_bad", 0), drops=drops,
                metric=t["metric"], expect=t.get("expect"))


def main():
    tok = T.Tok(C.MODEL)
    rows = []
    for name in WANT:
        try:
            kw = dict(n_train=N_TRAIN, n_held=N_HELD) if name in GENERATED else {}
            t = GENERATORS[name](tok, **kw)
        except FileNotFoundError as e:
            print("%s skipped: %s" % (name, e))
            continue
        if not t["rows"]:                                    # e.g. greater-than on TuringLLM (digit-level years)
            print("%s: no usable item under %s's tokenizer (%s); not written" % (name, C.MODEL, t["counts"]))
            old = OUT / ("%s.json" % name)
            if old.exists():
                old.unlink()
            continue
        write(t)
        rows.append(summary_row(t))
    survey, chosen = None, []
    if os.environ.get("NO_BLIMP") != "1" and not os.environ.get("TASKS"):
        try:
            path = Path(os.environ["BLIMP"]) if os.environ.get("BLIMP") else None
            BL.blimp_dir(path)
        except FileNotFoundError as e:
            print("BLiMP skipped: %s" % e)
        else:
            survey = BL.survey(tok, path)
            chosen = BL.select(survey)
            for old in OUT.glob(C.BLIMP_PREFIX + "*.json"):        # a previous selection's files
                stem = old.name.split(".")[0]
                if stem[len(C.BLIMP_PREFIX):] not in chosen:
                    old.unlink()
            (OUT / "blimp_survey.json").write_text(json.dumps(dict(
                source=BL.BLIMP_SOURCE, checksum=BL.checksum(path), model=C.MODEL, qual_min=BL.QUAL_MIN,
                per_term=BL.PER_TERM, max_subsets=BL.MAX_SUBSETS, min_rows=BL.MIN_ROWS, min_prefixes=BL.MIN_PREFIXES,
                selected=chosen, subsets=survey), indent=1) + "\n")
            for uid in chosen:
                t = BL.subset_task(tok, uid, BL.qualify(tok, BL.read_subset(uid, path)))
                write(t)
                rows.append(summary_row(t))
    if not os.environ.get("TASKS"):
        (OUT / "summary.json").write_text(json.dumps(rows, indent=1) + "\n")

    print("MODEL %s | held-out cap %s | generated tasks: %d training / %d held-out | SFC split %s" % (
        C.MODEL, C.HELDOUT_CAP or "none", N_TRAIN, N_HELD, O.SFC_SPLIT))
    print("%-44s %-12s %6s %6s %5s %8s %6s %7s %-5s %-9s %7s  %s" % (
        "task", "family", "kept", "train", "held", "uncapped", "groups", "tokens", "mode", "contrast", "multi", "drops"))
    for r in rows:
        print("%-44s %-12s %6d %6d %5d %8d %6d %3s-%-3s %-5s %-9s %3d/%-3d  %s" % (
            r["name"], r["family"], r["kept"], r["train"], r["held"], r["held_uncapped"], r["groups"], r["len_min"],
            r["len_max"], r["answer_mode"], r["contrast"], r["multi_good"], r["multi_bad"], r["drops"]))
    for r in rows:
        if r.get("overlap"):
            o = r["overlap"]
            print("  %s: file records %s; held-out items whose prompt is a training prompt %d, identical to a training "
                  "item %d (of %d before the cap)" % (r["name"], o["file_records"], o["held_prefix_in_train"],
                                                     o["held_item_in_train"], r["held_uncapped"]))
        if r.get("lists_split"):
            print("  %s: lists split train / held: %s" % (r["name"], r["lists_split"]))
    if survey is not None:
        q = sorted((u for u, s in survey.items() if s["qualifies"]), key=lambda u: -survey[u]["n_qual"])
        print("\nBLiMP under %s's tokenizer: %d subsets, %d qualify (>= %.0f%% of pairs with one single-token critical "
              "word, one-prefix paradigm)" % (C.MODEL, len(survey), len(q), 100 * BL.QUAL_MIN))
        print("%-52s %-30s %6s %6s %8s %7s" % ("subset", "phenomenon", "qual%", "pairs", "prefixes", "chosen"))
        for u in q:
            s = survey[u]
            print("%-52s %-30s %5.0f%% %6d %8d %7s" % (u, s["term"], 100 * s["share"], s["n_qual"], s["n_prefixes"],
                                                     "yes" if u in chosen else ""))
        near = sorted((u for u, s in survey.items() if not s["qualifies"] and s["share"] >= 0.4),
                      key=lambda u: -survey[u]["share"])
        print("not qualifying (40-80%%): %s" % ", ".join("%s %.0f%%" % (u, 100 * survey[u]["share"]) for u in near))
        print("selected (%d): %s" % (len(chosen), ", ".join(chosen)))
    print("\nwrote %d task files -> %s" % (len(rows), OUT))


if __name__ == "__main__":
    main()

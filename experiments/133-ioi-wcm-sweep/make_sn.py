"""133 (CPU, tokenizer only): a SAME-NAMES held-out set for IOI.

081's IOI held-out items use the other half of Wang et al.'s name / place / object lists (disjoint from training).
This writes N_SN fresh items from the TRAINING half of the lists (same generator, originals.ioi, same seed: the
training side is simply filled further; its first 1,600 rows are 081's training items, checked prompt by prompt),
none of which is a training item (the Builder de-duplicates by (io, s, place, obj, template, order)).

Output: experiments/133-ioi-wcm-sweep/tasks/ioi_sn.json = 081's ioi task (rows unchanged, ids unchanged) + the SN
rows appended, with split["held_sn"] = their ids. split["train"] / split["held"] are 081's.

  PYTHONPATH=src .venv/bin/python experiments/133-ioi-wcm-sweep/make_sn.py
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E81 = HERE.parent / "081-task-circuits"
sys.path.insert(0, str(E81))
import task_common as C  # noqa: E402
import tasks as T  # noqa: E402
import originals as O  # noqa: E402

N_SN = int(os.environ.get("N_SN", 1000))


def main():
    base = C.load_task("ioi")
    tok = T.Tok(C.MODEL)
    n_tr = len(base["split"]["train"])
    gen = O.ioi(tok, n_train=n_tr + N_SN, n_held=0)
    tr_prompts = {base["rows"][i]["prompt"] for i in base["split"]["train"]}
    all_prompts = {r["prompt"] for r in base["rows"]}
    first = [r["prompt"] for r in gen["rows"][:n_tr]]
    same = sum(p in tr_prompts for p in first)
    print("generator check: %d / %d of the first rows are 081 training prompts" % (same, n_tr))
    assert same == n_tr, "the regenerated training side does not reproduce 081's training items"
    held_names = set()
    for i in base["split"]["held"]:
        m = base["rows"][i]["meta"]
        held_names |= {m["io"], m["s"]}
    tr_names = set()
    for i in base["split"]["train"]:
        m = base["rows"][i]["meta"]
        tr_names |= {m["io"], m["s"], *m["abc"]}
    new = []
    for r in gen["rows"][n_tr:]:
        if r["prompt"] in all_prompts:
            continue
        m = r["meta"]
        assert {m["io"], m["s"]} <= tr_names and not ({m["io"], m["s"]} & held_names)
        new.append(r)
    new = new[:N_SN]
    rows = list(base["rows"])
    ids = []
    for r in new:
        r = dict(r, id=len(rows))
        ids.append(r["id"])
        rows.append(r)
    out = dict(base, rows=rows)
    out["split"] = dict(base["split"], held_sn=ids)
    out["sn_meta"] = dict(n_sn=len(ids), rule="fresh items from the TRAINING half of the name / place / object lists "
                                             "(originals.ioi, training side filled to %d); no training prompt" %
                                             (n_tr + N_SN))
    p = HERE / "tasks" / "ioi_sn.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    lens = [len(r["tokens"]) for r in new]
    print("wrote %s: %d SN rows (len %d-%d), %d train names, %d held names" % (p, len(ids), min(lens), max(lens),
                                                                             len(tr_names), len(held_names)))


if __name__ == "__main__":
    main()

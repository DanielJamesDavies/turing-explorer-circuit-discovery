"""Experiment 081, stage 2: the competence probe (GPU, full model, no circuit).

Per task (every tasks/<MODEL>/*.json, or TASKS=a,b): the task metric of the full model on every item's prompt (summed
teacher-forced log-probs for multi-token answers) and, where the dataset has its own patch prompts, on the item's
counterfactual (scored for the clean item's good / bad answers). The pre-stated rule (task_common.competence, README)
is applied to the HELD-OUT items: COMPETENT iff mean metric > 0, >= 70% of items have metric > 0, and (original-contrast
tasks only) mean(clean) - mean(counterfactual) > 0. Only competent tasks are calibrated and fitted.

Generated-contrast tasks need their pool (prepare.py) only for fitting, not here.

Writes OUT_PROBE/competence.json and OUT_PROBE/rows.jsonl.gz; prints the table. REPORT=1 reprints it (CPU).

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python -X utf8 experiments/081-task-circuits/probe.py
  env: MODEL (gpt2-small)  TASKS  OUT_PROBE (experiments/081-task-circuits/probe/<MODEL>)  REPORT=1
"""
import gzip
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import task_common as C  # noqa: E402

OUT_PROBE = Path(os.environ.get("OUT_PROBE", str(HERE / "probe" / C.MODEL)))


def task_names():
    want = [t for t in os.environ.get("TASKS", "").split(",") if t]
    return want or [p.stem for p in C.task_files()]


def summarise(task, vals):
    """The verdict on held-out items (and the same numbers on training items, descriptive). vals = Engine.probe's
    {"clean": {task: {values, correct}, obj: ..., top1}, "cf": same or None}, items ordered train then held."""
    n_tr = len(task["split"]["train"])
    out = dict(name=task["name"], family=task["family"], expect=task.get("expect"), metric=task["metric"]["task"],
               contrast=task["contrast"], answer_mode=task.get("answer_mode"))
    for part, sl in (("held", slice(n_tr, None)), ("train", slice(0, n_tr))):
        clean = vals["clean"][C.PRIMARY]["values"][sl]
        cf = vals["cf"][C.PRIMARY]["values"][sl] if vals.get("cf") else None
        comp = C.competence(clean, cf)
        top1 = vals["clean"].get("top1")
        top1 = top1[sl] if top1 else None
        obj = vals["clean"][C.OBJ]["values"][sl]
        out[part] = dict(comp, top1=(sum(top1) / max(1, len(top1))) if top1 else None,
                         obj_mean=sum(obj) / max(1, len(obj)))
    out["competent"] = out["held"]["competent"]
    return out


def print_table(res):
    print("%-44s %-12s %-12s %5s %9s %8s %8s %7s %6s  %s" % ("task", "family", "metric", "n", "mean", "correct",
                                                            "cf gap", "top-1", "expect", "verdict"))
    for r in res.values():
        h = r["held"]
        gap = "%8.3f" % h["cf_gap"] if h.get("cf_gap") is not None else "     n/a"
        top1 = "%6.0f%%" % (100 * h["top1"]) if h.get("top1") is not None else "      -"
        fails = [k for k, ok in (("mean", h["crit_mean"]), ("70%", h["crit_frac"]), ("cf", h["crit_cf"])) if ok is False]
        print("%-44s %-12s %-12s %5d %9.3f %7.0f%% %s %s %6s  %s" % (
            r["name"], r["family"], r["metric"], h["n"], h["mean_clean"], 100 * h["frac_correct"], gap, top1,
            {True: "yes", False: "no", None: "-"}[r["expect"]],
            "COMPETENT" if r["competent"] else "not competent (%s)" % ", ".join(fails)))
    print("rule (held-out items): mean > %g, share correct >= %.0f%%, and for original-contrast tasks mean(clean) - "
          "mean(cf) > %g" % (C.COMPETENCE["min_mean"], 100 * C.COMPETENCE["min_frac_correct"],
                             C.COMPETENCE["min_cf_gap"]))


def main():
    import engine as E
    OUT_PROBE.mkdir(parents=True, exist_ok=True)
    eng = E.Engine(C.MODEL)
    res = {}
    with gzip.open(OUT_PROBE / "rows.jsonl.gz", "wt") as fh:
        for name in task_names():
            t0 = time.time()
            task = C.load_task(name)
            B = eng.batch(task) if task["contrast"] == "original" else _batch_no_pool(eng, task)
            vals = eng.probe(B, task)
            res[name] = summarise(task, vals)
            fh.write(json.dumps(dict(task=name, items=B["items"], **vals)) + "\n")
            print("%-44s %s  (%.1fs)" % (name, "COMPETENT" if res[name]["competent"] else "not competent",
                                         time.time() - t0), flush=True)
    (OUT_PROBE / "competence.json").write_text(json.dumps(dict(rule=C.COMPETENCE, model=C.MODEL, generated=time.strftime(
        "%Y-%m-%d %H:%M"), tasks=res), indent=1) + "\n")
    print()
    print_table(res)


def _batch_no_pool(eng, task):
    """A generated-contrast task's batch for the probe: the C prompts are not needed, so a one-prompt stand-in pool
    (the first training prompt) satisfies batch()'s contract without reading prepare.py's output."""
    return eng.batch(task, contrast={"pool": [task["split"]["train"][0]]})


def competent_tasks(path: Path = OUT_PROBE / "competence.json"):
    """The competent task names from a finished probe; refuses without one (no silent default)."""
    if not Path(path).exists():
        raise FileNotFoundError("%s does not exist: run probe.py first (only competent tasks are fitted)" % path)
    d = json.loads(Path(path).read_text())["tasks"]
    return [k for k, v in d.items() if v["competent"]]


if __name__ == "__main__":
    if os.environ.get("REPORT") == "1":
        print_table(json.loads((OUT_PROBE / "competence.json").read_text())["tasks"])
    else:
        main()

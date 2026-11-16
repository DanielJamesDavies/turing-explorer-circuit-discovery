"""117 stage 2 (GPU): two WCM v4 circuits per target, both through 062's production driver.Runner (lambda 1e-3, 400
steps, v4 = Z + A + C + R_C, rank-keep 1e-1, geometric lr x3.15 with warm-up, censored term):
  corpus  the production circuit: the target's own corpus record (production data-pass stores, OpenWebText 4M windows,
          build_record with the group-aware split), fitted exactly as the 2026-11-07 pod run fitted its 512 targets
  task    the same method on a task record (lib117.task_record: task prompts + corpus filler, anchors at END / S2,
          verified-silent counterfactual contrast)
Each circuit is scored with the production scorer (eval.wcm_eval.score_circuit through Runner.score) on
  data=text  the corpus record's held-out strongest and mid-band rows (the paper's held-out text)
  data=task  the task record's held-out prompts (disjoint names / nouns): strongest folds 0-3 (16 each) and mid fold 0

Resumable (circuits and records cached; eval rows keyed). Stops starting new targets after BUDGET_MIN minutes.
  TARGETS=out/targets.txt BUDGET_MIN=80 PYTHONPATH=src python -X utf8 -u experiments/117-known-circuits-gpt2/fit117.py
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lib117 as L  # noqa: E402

L.setup_env()
import torch  # noqa: E402
import driver  # noqa: E402  (062's; configured from the env set above)

BUDGET = float(os.environ.get("BUDGET_MIN", "80")) * 60
TARGETS = Path(os.environ.get("TARGETS", str(L.OUT / "targets.txt")))


def pathway_of(key, sel):
    for p, d in sel["pathways"].items():
        if any(c["key"] == key for c in d["chosen"]):
            return p
    raise KeyError(key)


def main():
    t_start = time.time()
    sel = json.loads((L.OUT / "targets.json").read_text())
    targets = [t for t in TARGETS.read_text().split() if t]
    OUT = Path(os.environ["OUT"])
    R = driver.Runner()
    M = R.method(driver.GAMMA, driver.LAM, True)
    loader = R.G["M0"].probe_builder.loader
    n_corpus = int(sum(loader._shard_sequence_counts))
    inf, bank = R.G["inference"], R.G["bank"]
    d = OUT / "dan215"
    d.mkdir(parents=True, exist_ok=True)
    ev_path, st_path, tr_path = d / "eval.jsonl", d / "status.jsonl", d / "train.jsonl"
    done = set()
    if ev_path.exists():
        for line in open(ev_path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "error" not in r:
                done.add((r["seed"], r["circuit"], r["data"], r["held"], r.get("fold", 0)))
    fe, fs, fr = open(ev_path, "a"), open(st_path, "a"), open(tr_path, "a")
    tasks = {}
    print("[117] %d targets, corpus %d windows, out %s" % (len(targets), n_corpus, OUT), flush=True)
    for key in targets:
        if time.time() - t_start > BUDGET:
            print("[117] budget reached; stopping before %s" % key, flush=True)
            break
        pw = L.PATHWAYS[pathway_of(key, sel)]
        st = dict(seed=key, pathway=pathway_of(key, sel))
        t0 = time.time()
        try:
            # ---- corpus (production) circuit
            rec_c = R.contexts(key)
            circs = {}
            if rec_c.get("strong") is None:
                st["corpus_skip"] = "no_contrast" if rec_c.get("no_contrast") else "too_few_contexts"
            else:
                arm_c, _ = R.train_arm(rec_c)
                c, t_fit = R.fit(M, rec_c, key, arm_c, OUT / "main" / "circuits" / ("%s.pt" % key), train_fh=fr,
                                 label=dict(circuit="corpus"))
                st["t_fit_corpus"] = round(t_fit, 1)
                if c is not None:
                    circs["corpus"] = c
                    st["n_corpus"] = driver.n_nodes(c)
            # ---- task circuit
            tp = OUT / "task_ctx" / ("%s.pt" % key)
            if tp.exists():
                rec_t = torch.load(tp, weights_only=False)[key]
            else:
                if pw["task"] not in tasks:
                    tasks[pw["task"]] = L.load_task(pw["task"])
                ent = next(c for c in sel["pathways"][st["pathway"]]["chosen"] if c["key"] == key)
                rec_t = L.task_record(inf, bank, loader, n_corpus, key, tasks[pw["task"]], pw["role"],
                                      io_token=ent.get("io_token"))
                tp.parent.mkdir(parents=True, exist_ok=True)
                torch.save({key: rec_t}, tp)
            st.update(task_fire_train=rec_t.get("fire_rate_train"), task_fire_held=rec_t.get("fire_rate_held"),
                      task_contrast=rec_t.get("contrast_source"), task_silent_share=rec_t.get("contrast_silent_share"))
            if rec_t.get("strong") is None:
                st["task_skip"] = "no_contrast" if rec_t.get("no_contrast") else "too_few_contexts"
            else:
                arm_t, _ = R.train_arm(rec_t)
                c, t_fit = R.fit(M, rec_t, key, arm_t, OUT / "task" / "circuits" / ("%s.pt" % key), train_fh=fr,
                                 label=dict(circuit="task"))
                st["t_fit_task"] = round(t_fit, 1)
                if c is not None:
                    circs["task"] = c
                    st["n_task"] = driver.n_nodes(c)
            # ---- scores
            t1 = time.time()
            jobs = []
            for cname in circs:
                if rec_c.get("strong") is not None:
                    for held in ("strong", "mid"):
                        if rec_c.get(held) is not None:
                            jobs.append((cname, "text", held, 0, rec_c))
                if rec_t.get("strong") is not None:
                    for k in range(4):
                        rf = L.fold(rec_t, "strong", k)
                        if rf is not None:
                            jobs.append((cname, "task", "strong", k, rf))
                    if rec_t.get("mid") is not None:
                        jobs.append((cname, "task", "mid", 0, L.fold(rec_t, "mid", 0)))
            for cname, data, held, k, rec in jobs:
                if (key, cname, data, held, k) in done:
                    continue
                row = R.score(circs[cname], rec, held)
                row.update(seed=key, circuit=cname, data=data, held=held, fold=k, pathway=st["pathway"])
                fe.write(json.dumps(row) + "\n"); fe.flush()
            st["t_eval"] = round(time.time() - t1, 1)
            st["ok"] = True
        except Exception as e:  # noqa: BLE001
            st["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
            traceback.print_exc()
            torch.cuda.empty_cache()
        st["t_total"] = round(time.time() - t0, 1)
        fs.write(json.dumps(st) + "\n"); fs.flush()
        print("[117] %s %s" % (key, json.dumps(st)), flush=True)


if __name__ == "__main__":
    main()

"""Experiment 081, stage 4: the task-circuit fits (GPU, resumable), per task x arm x lambda.

Per competent task (probe/<MODEL>/competence.json; TASKS overrides, FORCE_TASKS=1 to fit a non-competent one):
  W, U    for every lambda of the run config (config_main_<MODEL>.json, from calibrate.py ADOPT=1): fit
          (engine.Engine.fit: WCM with the output endpoint, the task's contrastive objective, every SAE site
          maskable, Z + C + A with gamma 0.25, A = the clean training prompts' mean, C = the patch prompts' mean, or
          the generated contrast pool's for tasks without patch prompts (prepare.py)), save the circuit
          (circuits/<task>/<arm>/<lambda>.json: {site: {latent: alpha}} + fit stats), score it on the held-out prompts
          with its coefficients applied (eval.wcm_eval.score_task: task metric and objective metric, Z / A_sp / C_sp,
          necessity, accuracy recovered by the circuit alone)
  profile every W circuit: per member, activation by token role and attribution at every position (engine.profile)
          -> profiles.jsonl.gz (report.py's structure check and T5)
  RND     for every W lambda with a non-empty circuit: N_DRAWS (5) random circuits matched to W's per-site member
          counts, drawn from the latents active on the training prompts (active/<task>.json.gz), coefficients fitted
          with support restricted, gates open, no penalty (062 random_null.py), scored identically
  W_ZA    generated-contrast tasks only, at every W lambda: the SENSITIVITY arm, W trained on Z + A with no C term
          (the generated contrast set is Daniel's latent rule transferred to outputs); scored identically, so its rows
          carry C too, and report.py reads it under both the full rule and the Z + A rule
One eval row per (task, arm, lambda, draw) in eval.jsonl (draw -1 for W / U / W_ZA); an error writes an error row and
the run moves on; error rows are retried on resume. Circuits on disk are reloaded, not refitted.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python -X utf8 experiments/081-task-circuits/run.py
  env: MODEL (gpt2-small)  OUT (out/<MODEL>)  CONFIG (config_main_<MODEL>.json)  TASKS (default: the competent
       ones)  ARMS (W,U,RND,W_ZA)  N_DRAWS (the config's, else 5)  NO_PROFILE=1  LIMIT (first N tasks)  FORCE_TASKS=1
       HELDOUT_N (score on the first N held-out items only; smoke.sh)  PROFILE_N (held-out items per profile; 256)
Training (2026-10-21): every fit draws its steps x batch presentations from the task's whole training pool, without
replacement within an epoch (engine.presentations); A / C are the whole training / contrast pools' means, computed
once per task (engine.means).
"""
import gzip
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import task_common as C  # noqa: E402

OUT = Path(os.environ.get("OUT", str(HERE / "out" / C.MODEL)))
os.environ["OUT"] = str(OUT)
CONFIG = Path(os.environ.get("CONFIG", str(HERE / ("config_main_%s.json" % C.MODEL))))
ARMS = [a for a in os.environ.get("ARMS", "W,U,RND,W_ZA").split(",") if a]
LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
for _a in ARMS:
    assert _a in C.ARMS, "unknown arm %r (task_common.ARMS)" % _a


def log(s):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), s), flush=True)


def task_list():
    want = [t for t in os.environ.get("TASKS", "").split(",") if t]
    from probe import competent_tasks
    if want and os.environ.get("FORCE_TASKS") == "1":
        names = want
    else:
        ok = competent_tasks()
        names = [t for t in want if t in ok] if want else ok
        skipped = [t for t in want if t not in ok]
        if skipped:
            log("not competent (FORCE_TASKS=1 to fit anyway): %s" % ", ".join(skipped))
    return names[:LIMIT] if LIMIT else names


def done_keys(path):
    out = set()
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if "error" not in r:
                out.add((r["task"], r["arm"], r["lam"], int(r.get("draw", -1))))
    return out


def scalars(row):
    return {k: v for k, v in row.items() if isinstance(v, (int, float, str, bool)) or v is None or k == "per_kind"}


def circuit_path(task, arm, lt, draw=-1):
    return OUT / "circuits" / task / arm / (("%s.json" % lt) if draw < 0 else ("%s.d%d.json" % (lt, draw)))


def save_circuit(path, alphas, prov, extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = {k: prov.get(k) for k in ("loss_initial", "loss_final", "holdout_data_loss", "n_kept", "mean_m_kept",
                                     "amp_stats", "logit_scale", "logit_metric", "support_n", "steps",
                                     "floors_valid_masked", "floors_precomputed", "present")}
    path.write_text(json.dumps(dict(alphas=C.alphas_to_json(alphas), fit=keep, **extra)) + "\n")


def load_circuit(path):
    d = json.loads(path.read_text())
    return C.alphas_from_json(d["alphas"]), d


def active_pool(eng, B, task):
    p = OUT / "active" / ("%s.json.gz" % task)
    if p.exists():
        d = json.loads(gzip.open(p, "rt").read())
        return {(int(k.split("/")[0]), k.split("/")[1]): v for k, v in d.items()}
    live = eng.active_latents(B)
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt") as fh:
        fh.write(json.dumps({"%d/%s" % s: v for s, v in live.items()}))
    return live


def main():
    import random

    import torch

    import engine as E
    cfg = C.load_run_config(CONFIG)
    steps, norm = int(cfg["steps"]), str(cfg["logit_norm"])
    n_draws = int(os.environ.get("N_DRAWS", cfg.get("n_draws", C.N_DRAWS)))
    names = task_list()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run_config.json").write_text(json.dumps(dict(cfg, config_path=str(CONFIG), arms_run=ARMS, tasks=names,
                                                         n_draws=n_draws), indent=1))
    p_eval = OUT / "eval.jsonl"
    done = done_keys(p_eval)
    prof_done = set()
    if (OUT / "profiles.jsonl.gz").exists():
        with gzip.open(OUT / "profiles.jsonl.gz", "rt") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    prof_done.add((r["task"], r["arm"], r["lam"]))
                except Exception:  # noqa: BLE001
                    pass
    fe, ft, fs = open(p_eval, "a"), open(OUT / "train.jsonl", "a"), open(OUT / "status.jsonl", "a")
    fm = gzip.open(OUT / "members.jsonl.gz", "at")
    fp = gzip.open(OUT / "profiles.jsonl.gz", "at")
    log("%d tasks (%s) | arms %s | W %s lr %g | U %s lr %g | %d steps | logit_norm %s | %d draws | out %s" % (
        len(names), ",".join(names), ",".join(ARMS), cfg["arms"]["W"]["lams"], cfg["arms"]["W"]["lr"],
        cfg["arms"]["U"]["lams"], cfg["arms"]["U"]["lr"], steps, norm, n_draws, OUT))
    eng = E.Engine(C.MODEL)

    def one(task, B, arm, lam, lr, draw=-1, support=None, extra=None):
        """Fit (or reload) and score one circuit; returns its alphas (None on failure)."""
        lt = C.lam_tag(lam)
        key = (task, arm, lt, draw)
        path = circuit_path(task, arm, lt, draw)
        row = dict(task=task, family=B["family"], arm=arm, lam=lt, lam_value=lam, lr=lr, steps=steps, draw=draw,
                   logit_norm=norm, weighted=bool(C.ARMS[arm]["weighted"]), **(extra or {}))
        alphas = None
        try:
            if path.exists():
                alphas, saved = load_circuit(path)
                row.update(t_fit=saved.get("t_fit"), fresh_fit=False)
            else:
                alphas, prov, t_fit, peak = eng.fit(B, arm, lam, lr, steps, norm, support=support)
                save_circuit(path, alphas, prov, dict(task=task, arm=arm, lam=lam, lr=lr, steps=steps, draw=draw,
                                                      t_fit=round(t_fit, 1)))
                row.update(t_fit=round(t_fit, 1), peak_mb_fit=peak, fresh_fit=True)
                tr = dict(task=task, arm=arm, lam=lt, draw=draw, **{k: prov.get(k) for k in (
                    "loss_initial", "loss_final", "holdout_data_loss", "n_kept", "mean_m_kept", "amp_stats",
                    "logit_scale", "dual_norm_zero", "dual_norm_floor", "n_train_pos", "n_holdout_pos", "present")})
                curve = prov.get("curve") or {}
                tr["curve_loss"] = [float("%.4g" % v) for v in (curve.get("loss") or [])]
                tr["curve_n"] = curve.get("n_members")
                ft.write(json.dumps(tr) + "\n"); ft.flush()
            if key not in done:
                t = time.time()
                row.update(scalars(eng.score(B, alphas)), t_score=round(time.time() - t, 1))
                fm.write(json.dumps(dict(task=task, arm=arm, lam=lt, draw=draw,
                                         members=C.alphas_to_json(alphas))) + "\n"); fm.flush()
        except Exception as e:  # noqa: BLE001
            row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            alphas = None
        if key not in done:
            fe.write(json.dumps(row) + "\n"); fe.flush()
            if "error" not in row:
                done.add(key)
            log("%-34s %-3s lam %-7s d%-2d n=%-6s fit %6ss  Z %s A %s C %s nec %s pass %s acc %s %s" % (
                task, arm, lt, draw, row.get("n"), row.get("t_fit"), _f(row.get("task_free0")),
                _f(row.get("task_freeM_topk")), _f(row.get("task_freeN_topk")), _f(row.get("task_nec")),
                row.get("task_pass"), _f(row.get("task_acc_free0")), row.get("error") or ""))
        return alphas

    for n_, task in enumerate(names):
        st = dict(task=task)
        t_all = time.time()
        try:
            T = C.load_task(task)
            generated = T["contrast"] == "generated"
            B = eng.batch(T, C.load_contrast(task) if generated else None)
            st.update(n_train=B["n_train"], n_held=B["n_held"], width=B["width"], c_source=B["c_source"],
                      n_c=int(B["c_tokens"].shape[0]), n_held_uncapped=T.get("split_meta", {}).get("n_held_uncapped"),
                      heldout_n=C.heldout_n() or None)
            t_m = time.time()
            eng.means(B)                                          # A / C over the whole pools, once per task
            st["t_means"] = round(time.time() - t_m, 1)
            w_alphas = {}
            for arm in ("W", "U"):
                if arm not in ARMS:
                    continue
                a = cfg["arms"][arm]
                for lam in a["lams"]:
                    al = one(task, B, arm, float(lam), float(a["lr"]))
                    if arm == "W" and al is not None:
                        w_alphas[C.lam_tag(float(lam))] = (float(lam), al)
            if "W_ZA" in ARMS and generated:                     # the no-C sensitivity arm, at W's lambdas
                for lam in cfg["arms"]["W"]["lams"]:
                    one(task, B, "W_ZA", float(lam), float(cfg["arms"]["W"]["lr"]))
            if os.environ.get("NO_PROFILE") != "1":
                for lt, (lam, al) in w_alphas.items():
                    if (task, "W", lt) in prof_done or not al:
                        continue
                    t = time.time()
                    prof = eng.profile(B, al)
                    fp.write(json.dumps(dict(task=task, arm="W", lam=lt, lam_value=lam, members=prof)) + "\n")
                    fp.flush()
                    prof_done.add((task, "W", lt))
                    log("%-34s profile W lam %s: %d members (%.0fs)" % (task, lt, len(prof), time.time() - t))
            if "RND" in ARMS and w_alphas:
                live = active_pool(eng, B, task)
                st["n_active"] = sum(len(v) for v in live.values())
                for lt, (lam, al) in w_alphas.items():
                    if not al:
                        continue
                    counts = C.site_counts(al)
                    rng = random.Random("%s-%s-null" % (task, lt))
                    for d in range(n_draws):
                        sup, short = C.draw_support(counts, live, rng)   # drawn before any skip: reruns reproduce
                        if (task, "RND", lt, d) in done:
                            continue
                        one(task, B, "RND", lam, float(cfg["arms"]["W"]["lr"]), draw=d, support=sup,
                            extra=dict(n_ref=sum(counts.values()), n_drawn=sum(len(v) for v in sup.values()),
                                       short_sites=short))
            st["ok"] = True
        except Exception as e:  # noqa: BLE001
            st["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
            traceback.print_exc()
        st["t_total"] = round(time.time() - t_all, 1)
        fs.write(json.dumps(st) + "\n"); fs.flush()
        log("%2d/%d %-34s %s (%.0fs)" % (n_ + 1, len(names), task, st.get("error") or "ok", st["t_total"]))
    fm.close()
    fp.close()
    log("DONE")


def _f(v):
    return "%.2f" % v if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v else "-"


if __name__ == "__main__":
    main()

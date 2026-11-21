"""Exp 132 shared setup: exp 081's engine (GPT-2 small + OAI v5 32k SAEs, 48 sites), the IOI task with 081's split, the
081 A / C means, and 081's scorer (eval.wcm_eval.score_task) called with an explicit ablation scope.

Nothing here changes 081's code or src: `score(eng, B, alphas, scope)` is engine.Engine.score with `sites=scope`
instead of `set(eng.sites)` (scope = every site reproduces engine.score exactly).
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
D081 = ROOT / "experiments" / "081-task-circuits"
os.environ.setdefault("MODEL", "gpt2-small")
os.environ.setdefault("OUT", str(HERE / "out"))          # engine.py's default OUT; nothing is written there by 081 code
sys.path.insert(0, str(D081))
OUT = Path(os.environ.get("OUT132", str(HERE / "out")))     # OUT132=.../out_smoke for HELDOUT_N smoke runs
OUT.mkdir(parents=True, exist_ok=True)

import task_common as C  # noqa: E402

EVAL_BS = int(os.environ.get("EVAL_BS", 64))
KINDS = ("attn", "resid_mid", "mlp", "resid")
TASK = "ioi"


def log(s):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), s), flush=True)


def setup():
    """(engine, B) for IOI: 081's batch (1,600 training items then 5,000 held-out) and its A / C means."""
    import engine as E
    E.EVAL_BS = EVAL_BS
    eng = E.Engine(C.MODEL)
    T = C.load_task(TASK)
    if os.environ.get("HELD_SET") == "sn":
        # exp 133's SAME-NAMES held-out set (133/make_sn.py; read only): fresh items from the TRAINING halves of the
        # name / place / object lists, none a training item. Training rows, C pool and hence the A / C means are 081's.
        T = json.loads((ROOT / "experiments" / "133-ioi-wcm-sweep" / "tasks" / "ioi_sn.json").read_text())
        T = dict(T, split=dict(train=list(T["split"]["train"]), held=list(T["split"]["held_sn"])))
        log("HELD_SET=sn: %d same-name held-out items available" % len(T["split"]["held"]))
    B = eng.batch(T)
    t = time.time()
    eng.means(B)
    log("IOI: %d train / %d held-out items, C pool %d, width %d; means %.0fs"
        % (B["n_items_train"], B["n_items_held"], int(B["c_tokens"].shape[0]), B["width"], time.time() - t))
    return eng, B, T


def scope_of(eng, kinds):
    return {s for s in eng.sites if s[1] in kinds}


def score(eng, B, alphas, scope=None):
    """081's scorer on the held-out items (engine.Engine.score with an explicit scope)."""
    from eval import wcm_eval as W
    sites = set(eng.sites) if scope is None else set(scope)
    M = eng.means(B)
    return W.score_task(eng.inf, eng.bank, alphas, B["tokens"], B["anchors"], B["n_train"], B["c_tokens"],
                        eng.metrics(B), valid=B["valid"], cf_valid=B["c_valid"], sites=sites, batch_size=EVAL_BS,
                        vacuous_abs=B["vac"], means_A=M["A"], means_C=M["C"])


KEYS = ("n", "n_sites_scope", "n_out_of_scope", "task_full", "task_e0", "task_eM_topk", "task_eN_topk",
        "task_free0_raw", "task_freeM_topk_raw", "task_freeN_topk_raw", "task_free0", "task_freeM_topk",
        "task_freeN_topk", "task_nec", "task_acc_full", "task_acc_e0", "task_acc_free0", "task_acc_freeM_topk",
        "task_acc_freeN_topk", "task_pass", "task_vacuous", "all_sites_ablated", "per_kind", "secs")


def slim(row):
    return {k: row.get(k) for k in KEYS}


def append(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def f2(v):
    return "%.3f" % v if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v else "-"

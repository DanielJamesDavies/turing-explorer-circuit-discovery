"""106 shared harness: build the 062 driver's Runner exactly as production does (WCM_VERSION=v3, CTX_DIR records,
explicit split), fit one target with optional prototype patches, and record timing / memory / final parameters.

Nothing here edits src/ or the driver: the driver is imported (it reads its env at import, so set_env() runs first) and
prototypes are installed by fastlm.install() as in-memory replacements of module attributes.

  import h106; h106.set_env(out_dir); R, M = h106.runner(); row = h106.fit_one(R, M, key, out_dir, label)
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CTX = REPO / "experiments" / "062-h100-protocol-v1" / "out_full" / "out" / "ctx"
REF104 = REPO / "experiments" / "104-v3-cross-model" / "out" / "bitident" / "main" / "circuits"
TARGETS3 = ("2.attn.11852", "6.mlp.36361", "9.resid.22205")      # 104's bit-identity panel, one per depth band
# throw-away fits (profiling) go outside the repo, so nothing in the project is ever deleted
SCRATCH = Path(os.environ.get("SCRATCH_106", "/mnt/c/Users/danie/AppData/Local/Temp/claude/"
                              "X--Projects-AIs-Turing-Publication-3-Implementation-2/"
                              "560c3a52-c1b3-451b-b658-362a5c4fbee7/scratchpad/106"))


def set_env(out_dir, steps=400):
    """The production v3 settings of 104's run_bitident.sh (MODE=main, explicit split from CTX_DIR, ORDER=list).
    FAMILY=gpt2-small | pythia-70m (env) switches to 104 part B's public setting (its records, SAE_PRELOAD=0)."""
    fam = os.environ.get("FAMILY", "turingllm")
    ctx = CTX if fam == "turingllm" else REPO / "experiments" / "104-v3-cross-model" / "out" / fam / "rec" / "ctx"
    os.environ.update(MODE="main", WCM_VERSION="v3", OUT=str(out_dir), CTX_DIR=str(ctx), CURVE_STRIDE="20",
                      ORDER="list", SKIP_SPEC="1", STEPS=str(int(steps)), HF_HUB_OFFLINE="1")
    if fam != "turingllm":
        os.environ.update(DATA=str(REPO / "data_public" / fam), STORES=str(REPO / "outputs_public" / fam),
                          SAE_PRELOAD="0")
    for p in (str(REPO / "src"), str(REPO / "experiments" / "062-h100-protocol-v1"), str(HERE)):
        if p not in sys.path:
            sys.path.insert(0, p)


_OPT = {"last": None, "n": 0, "t": []}
_SYNC_STEPS = [False]


def _install_opt_hook():
    import torch
    from torch.optim.optimizer import register_optimizer_step_post_hook

    def post(opt, args, kwargs):
        _OPT["last"] = opt
        _OPT["n"] += 1
        if _SYNC_STEPS[0]:
            torch.cuda.synchronize()
            _OPT["t"].append(time.perf_counter())
    register_optimizer_step_post_hook(post)


def runner():
    import torch
    import driver
    R = driver.Runner()
    M = R.method(driver.GAMMA, driver.LAM, True)
    _install_opt_hook()
    return R, M, driver


def final_params(opt):
    """The engine's parameters after the last step: thetas (one per site, in sites order) then amplitudes."""
    if opt is None:
        return None
    return [p.detach().float().cpu() for g in opt.param_groups for p in g["params"]]


def sig(c):
    out = []
    for n in c.nodes.values():
        if n.metadata.get("role") == "seed":
            continue
        f = n.metadata["feature_id"]
        out.append((int(f.layer), str(f.kind), int(f.index), float(n.metadata.get("amplitude", 1.0))))
    return sorted(out)


def fit_one(R, M, key, out_dir, label, sync_steps=False, save_params=True, score=False):
    """Fit `key` into out_dir/circuits/<key>.pt (resumable: an existing circuit + row is skipped). Returns the row."""
    import torch
    out_dir = Path(out_dir)
    rows_p = out_dir / "fits.jsonl"
    if rows_p.exists():
        for line in open(rows_p):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if r.get("key") == key and r.get("label") == label and "error" not in r:
                return r
    rec = R.contexts(key)
    arm, _thin = R.train_arm(rec)
    path = out_dir / "circuits" / ("%s.pt" % key)
    if path.exists():                                  # no row for it: a killed fit; set it aside (never deleted)
        path.rename(path.with_suffix(".stale%d.pt" % int(time.time())))
    _OPT["last"], _OPT["n"], _OPT["t"] = None, 0, []
    _SYNC_STEPS[0] = bool(sync_steps)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    m0 = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    c, t_fit = R.fit(M, rec, key, arm, path, train_fh=None)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    row = dict(key=key, label=label, t_fit=round(t1 - t0, 3), n=None if c is None else len(sig(c)),
               steps=_OPT["n"], peak_alloc_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1),
               peak_reserved_mb=round(torch.cuda.max_memory_reserved() / 2**20, 1),
               base_alloc_mb=round(m0 / 2**20, 1))
    if _OPT["t"]:
        st = [b - a for a, b in zip(_OPT["t"], _OPT["t"][1:])]
        st_sorted = sorted(st)
        row.update(step_s_median=round(st_sorted[len(st) // 2], 5), loop_s=round(_OPT["t"][-1] - _OPT["t"][0], 3),
                   pre_loop_s=None)
    prov = getattr(M, "last_mask_provenance", None) or {}
    curve = prov.get("curve") or {}
    row["loss_final"] = prov.get("loss_final")
    row["loss_curve"] = curve.get("loss")
    row["n_members_curve_last"] = (curve.get("n_members") or [None])[-1]
    if save_params:
        ps = final_params(_OPT["last"])
        if ps is not None:
            (out_dir / "params").mkdir(parents=True, exist_ok=True)
            torch.save(ps, out_dir / "params" / ("%s.pt" % key))
    if score and c is not None:
        t2 = time.perf_counter()
        sr = R.score(c, rec, "strong")
        row["t_eval_strong"] = round(time.perf_counter() - t2, 3)
        row["score"] = {k: v for k, v in sr.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    with open(rows_p, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


def compare_to(path_a, path_b):
    """Circuit identity of two saved circuits: exact (members + 4-dp amplitudes), member Jaccard, max |alpha diff|."""
    import torch
    a, b = sig(torch.load(path_a, weights_only=False)), sig(torch.load(path_b, weights_only=False))
    ka, kb = {x[:3] for x in a}, {x[:3] for x in b}
    da = {x[:3]: x[3] for x in a}
    db = {x[:3]: x[3] for x in b}
    common = ka & kb
    return dict(identical=a == b, members_identical=ka == kb, n_a=len(ka), n_b=len(kb),
                only_a=len(ka - kb), only_b=len(kb - ka), jaccard=round(len(common) / max(len(ka | kb), 1), 5),
                max_alpha_diff_common=max((abs(da[k] - db[k]) for k in common), default=0.0))


def compare_params(pa, pb):
    """Raw final parameters (fp32) of two fits: max |theta diff|, max |psi diff|, n exactly equal tensors."""
    import torch
    A, B = torch.load(pa), torch.load(pb)
    if len(A) != len(B):
        return dict(error="param count %d vs %d" % (len(A), len(B)))
    n = len(A) // 2
    dt = max(float((x - y).abs().max()) for x, y in zip(A[:n], B[:n]))
    dp = max(float((x - y).abs().max()) for x, y in zip(A[n:], B[n:])) if n else 0.0
    eq = sum(bool(torch.equal(x, y)) for x, y in zip(A, B))
    return dict(max_theta_diff=dt, max_psi_diff=dp, n_equal_tensors=eq, n_tensors=len(A))

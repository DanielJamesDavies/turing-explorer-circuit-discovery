"""PROTOCOL-V1 H100 DRIVER: fits and scores weighted circuits under the final protocol, one GPU per process.

Protocol v1 (DAN-75 spec, 2026-09-25):
  - contexts   32 strongest + 16 mid-band training contexts, stratified rotating split (ranks 1-2 always train),
               held-out 16 strongest (primary) + 16 mid-band (reported); 64 close contrast contexts, verified
               silent, stratified by similarity (48 train / 16 held-out). Built per target with 059's builder.
               Thin targets (no mid-band pool) fall back to the 48 strongest, flagged thin.
  - method     WCM[Z+C+A], gamma_C = gamma_A = 0.25, lambda = 1e-3, rank-keep 3e-3, ablation values from the
               training contexts, AdamW lr 0.05 / wd 0.05, 400 steps, batch 4.
The contexts reach the production engine through the same injection as 059 / 061 (protocol_harness), which
reproduced 059 arm B exactly; DAN-78 will move this into the production path.

MODE=main   per target: contexts -> fit -> eval pass (held-out strongest and mid-band) -> specificity (056)
MODE=sweep  the Figure 4 / DAN-76 data: per arm (WCM and unweighted masking over a lambda grid, gamma 0.25)
            fit + eval on held-out strongest. WCM at lambda 1e-3 reuses the main circuit when it exists.

Resumable at every step (one file per target for contexts and circuits; jsonl rows keyed by target / arm / held).
Each process takes targets[i::k] for SHARD=i/k.

  CUDA_VISIBLE_DEVICES=0 SHARD=0/8 MODE=main PYTHONPATH=src python experiments/062-h100-protocol-v1/driver.py
  env: MODE (main | sweep)  SHARD (0/1)  TARGETS (default targets_stage1.txt / targets_sweep.txt)
       OUT (default experiments/062-h100-protocol-v1/out)  LIMIT (first N of this shard, smoke tests)
       SWEEP_ARMS (comma list, default all)  SKIP_SPEC=1
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).parent
EXP = HERE.parent
for d in ("059-context-pool", "049-circuit-graph", "056-specificity"):
    sys.path.insert(0, str(EXP / d))
MODE = os.environ.get("MODE", "main")
SHARD_I, SHARD_K = (int(x) for x in os.environ.get("SHARD", "0/1").split("/"))
TARGETS = Path(os.environ.get("TARGETS", str(HERE / ("targets_stage1.txt" if MODE == "main" else "targets_sweep.txt"))))
OUT = Path(os.environ.get("OUT", str(HERE / "out")))
LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
SKIP_SPEC = os.environ.get("SKIP_SPEC") == "1"
GAMMA, LAM = 0.25, 1e-3
W_LAMS = (2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3)
U_LAMS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
SWEEP = [("W_%g" % l, True, l) for l in W_LAMS] + [("U_%g" % l, False, l) for l in U_LAMS]
WANT = [a for a in os.environ.get("SWEEP_ARMS", "").split(",") if a]
SWEEP = [a for a in SWEEP if not WANT or a[0] in WANT]
TAG = "shard%d" % SHARD_I


def log(s):
    print("[%s %s] %s" % (time.strftime("%H:%M:%S"), TAG, s), flush=True)


def jsonl_keys(path, fields):
    """Done keys from EVERY shard's file of this kind (eval.shard*.jsonl etc.), so a rerun with a different shard
    count or target list (stage 1 -> full) skips work any shard already finished."""
    out = set()
    for p in sorted(path.parent.glob(path.name.replace(TAG, "shard*"))):
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001  (a truncated last line after a kill)
                continue
            if "error" not in r:
                out.add(tuple(r.get(f) for f in fields))
    return out


class Runner:
    def __init__(self):
        os.environ.setdefault("TAG", "h100_unused")
        os.environ.setdefault("CTR_SOURCE", "close")
        import amp_eval_pass_v2 as V
        import protocol_harness as H
        self.V, self.H, self.P = V, H, H.P
        self.G = V.setup()
        M0 = self.G["M0"]
        # the eval / 056 injection replaces these per target; the context builder must always get the originals
        self._orig_sel = M0._neg_context_selector
        self._orig_pd = M0.build_probe_dataset
        self.ctx_dir = OUT / "ctx"; self.ctx_dir.mkdir(parents=True, exist_ok=True)
        self.dev = self.G["device"]

    # ---------------------------------------------------------------------------------------------- contexts
    def contexts(self, key):
        path = self.ctx_dir / ("%s.pt" % key)
        if path.exists():
            return torch.load(path, weights_only=False)[key]
        M0 = self.G["M0"]
        M0._neg_context_selector = self._orig_sel
        M0.build_probe_dataset = self._orig_pd
        self.P.seeds = lambda: [key]
        self.P.CTX = path
        self.P.build(self.G)
        return torch.load(path, weights_only=False)[key]

    def train_arm(self, rec):
        """32 + 16 (arm B); thin targets without a mid-band pool fall back to the 48 strongest (arm A)."""
        return ("B", False) if rec.get("mid") is not None else ("A", True)

    # ---------------------------------------------------------------------------------------------- fitting
    def method(self, gamma, lam, free_amp):
        from analysis.circuits.gradient_size_sweep_runner import _build_mode_method
        self.H.configure(gamma, lam, free_amp)
        return _build_mode_method("ablation_gradient", "mask", self.G["inference"], self.G["bank"], self.G["avg_acts"],
                                  self.G["M0"].probe_builder)

    def fit(self, M, rec, key, arm, path):
        if path.exists():
            return torch.load(path, weights_only=False), 0.0
        P, KINDS = self.P, self.G["KINDS"]
        tr = P.train_set(rec, arm)
        held = [P.pick(rec, "strong", rec["strong"]["held"])] + ([P.pick(rec, "mid", rec["mid"]["held"])] if rec["mid"] else [])
        pdset = P.probe(rec, tr, held, self.dev)
        M.build_probe_dataset = lambda comp, i, _p=pdset: _p
        M._floor_negatives = lambda probe_data, comp, i, logger: probe_data.neg_tokens
        l, k, i = self.H.parse(key)
        t0 = time.time()
        c = M.discover(l * len(KINDS) + KINDS.index(k), i)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(c, path)                                    # None = rejected; saved so resume skips it
        return c, time.time() - t0

    # ---------------------------------------------------------------------------------------------- scoring
    def score(self, c, rec, held):
        self.H.patch_eval_contexts(self.G, rec, held)
        return self.V.score_circuit(c, {}, skip_roles=True)    # empty live pool: free0_rand = empty circuit (not the
        #                                                        paper's random baseline, which is DAN-17)

    def spec(self, c, rec, arm_label):
        import specificity as S
        if not hasattr(self, "_taps"):
            self._taps = S.make_patchers(self.G)
        self.H.patch_eval_contexts(self.G, rec, "strong")
        return S.score(self.G, c, self.V, self._taps[0], self._taps[1], arm_label)


def n_nodes(c):
    return sum(1 for nd in c.nodes.values() if nd.metadata.get("role") != "seed")


def main_mode(R, targets):
    d = OUT / "main"
    ev_path, sp_path, st_path = d / ("eval.%s.jsonl" % TAG), d / ("spec.%s.jsonl" % TAG), d / ("status.%s.jsonl" % TAG)
    d.mkdir(parents=True, exist_ok=True)
    ev_done, sp_done = jsonl_keys(ev_path, ("seed", "held")), jsonl_keys(sp_path, ("seed",))
    st_done = jsonl_keys(st_path, ("seed",))
    M = R.method(GAMMA, LAM, True)
    fe, fs, ft = open(ev_path, "a"), open(sp_path, "a"), open(st_path, "a")
    for n_, key in enumerate(targets):
        if (key,) in st_done:
            continue
        st = dict(seed=key, shard=TAG)
        t = time.time()
        try:
            rec = R.contexts(key); st["t_ctx"] = round(time.time() - t, 1)
            if rec.get("strong") is None:
                st.update(skip="no_contrast" if rec.get("no_contrast") else "too_few_contexts", n_top=rec.get("n_top"))
                ft.write(json.dumps(st) + "\n"); ft.flush()
                continue
            arm, thin = R.train_arm(rec)
            st.update(arm=arm, thin=thin, n_top=rec["n_top"], n_mid=rec["n_mid"])
            c, t_fit = R.fit(M, rec, key, arm, d / "circuits" / ("%s.pt" % key))
            st["t_fit"] = round(t_fit, 1)
            if c is None:
                st["skip"] = "rejected"; ft.write(json.dumps(st) + "\n"); ft.flush(); continue
            st["n"] = n_nodes(c)
            t = time.time()
            for held in ("strong", "mid"):
                if rec.get(held) is None or (key, held) in ev_done:
                    continue
                row = R.score(c, rec, held)
                row.update(held=held, arm=arm, thin=thin, shard=TAG)
                fe.write(json.dumps(row) + "\n"); fe.flush()
            st["t_eval"] = round(time.time() - t, 1)
            t = time.time()
            if not SKIP_SPEC and (key,) not in sp_done:
                for r in R.spec(c, rec, arm):
                    fs.write(json.dumps(r) + "\n")
                fs.flush()
            st["t_spec"] = round(time.time() - t, 1)
            st["ok"] = True
        except Exception as e:  # noqa: BLE001
            st["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        ft.write(json.dumps(st) + "\n"); ft.flush()
        log("%4d/%d %-16s %s n=%s ctx %ss fit %ss eval %ss spec %ss" % (
            n_ + 1, len(targets), key, st.get("error") or st.get("skip") or "ok", st.get("n"), st.get("t_ctx"),
            st.get("t_fit"), st.get("t_eval"), st.get("t_spec")))


def sweep_mode(R, targets):
    d = OUT / "sweep"
    d.mkdir(parents=True, exist_ok=True)
    ev_path = d / ("eval.%s.jsonl" % TAG)
    ev_done = jsonl_keys(ev_path, ("seed", "arm"))
    fe = open(ev_path, "a")
    for name, weighted, lam in SWEEP:
        M = R.method(GAMMA, lam, weighted)
        log("arm %s (weighted=%s, lambda %g)" % (name, weighted, lam))
        for key in targets:
            if (key, name) in ev_done:
                continue
            try:
                rec = R.contexts(key)
                if rec.get("strong") is None:
                    continue
                arm, thin = R.train_arm(rec)
                main_c = OUT / "main" / "circuits" / ("%s.pt" % key)
                path = d / "circuits" / name / ("%s.pt" % key)
                if weighted and abs(lam - LAM) < 1e-12 and main_c.exists() and not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(torch.load(main_c, weights_only=False), path)        # identical config: reuse
                t = time.time()
                c, t_fit = R.fit(M, rec, key, arm, path)
                if c is None:
                    row = dict(seed=key, arm=name, skip="rejected")
                else:
                    row = R.score(c, rec, "strong")
                    row.update(held="strong", arm=name, weighted=weighted, lam=lam, train_arm=arm, thin=thin,
                               t_fit=round(t_fit, 1), t_total=round(time.time() - t, 1), shard=TAG)
            except Exception as e:  # noqa: BLE001
                row = dict(seed=key, arm=name, error="%s: %s" % (type(e).__name__, str(e)[:300]))
                traceback.print_exc()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            fe.write(json.dumps(row) + "\n"); fe.flush()
            log("%-8s %-16s %s n=%s" % (name, key, row.get("error") or row.get("skip") or "ok", row.get("n")))


def main():
    targets = [t for t in TARGETS.read_text().split() if t][SHARD_I::SHARD_K]
    if LIMIT:
        targets = targets[:LIMIT]
    log("MODE %s | %d targets | out %s" % (MODE, len(targets), OUT))
    R = Runner()
    (main_mode if MODE == "main" else sweep_mode)(R, targets)
    log("DONE")


if __name__ == "__main__":
    main()

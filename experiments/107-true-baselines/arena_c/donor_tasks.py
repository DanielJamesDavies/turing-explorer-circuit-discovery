"""Arena C (DAN-194): the donor-background (resample) test for TASK circuits, the behavioural mirror of
src/eval/donor_test.py (protocol v2 section 12.3, defined there for latent targets). A robustness diagnostic reported
beside the production pass rule; it does not replace it.

  bases      the held-out clean prompts of the circuit's task (SFC's 40 evaluation items under data mode "sfc")
  read       the task metric: logit diff (clean answer - patch answer) at the last position, as wcm_eval.score_task
             reads it (log_softmax of the anchor logits -> wcm_eval.task_metric "logit_diff")
  ablation   every latent outside the circuit, at every site of the circuit's scope, takes a donor prompt's natural code
             at the same position; members at alpha x live ("fit") or 1 x live ("a1"); the current stream's SAE error
             is kept. This is donor_test.ResamplePatcher at K = d_sae (no seed tap), written in its delta form
             x + decode(code - live code): with K = d_sae its Top-K budget (K - active members) is never binding, so
             every non-member takes the donor's value (0 where the donor is silent), exactly; the delta form makes the
             self-donor empty circuit an identity to float32 round-off
  pools      A = the training clean prompts (100); C = the held-out items' PATCH prompts (never a training donor: the
             training R donors are the training patch prompts); R [ours] = random windows of the local smoke corpus
             (data_public/pythia-70m, the data pass's; same tokenizer), truncated to the task's length T, minus any
             window equal to one of the task's prompts (clean or patch, training or held-out). Every donor has length T
  draws      DRAWS (8) paired draws, one donor per base per draw, from ONE seeded CPU generator: donor_test.
             draw_selections(layer 0, latent 0, ..., n_silent 0, seed = SEED + crc32(task) % 100000), pools in the
             order A, C, R; the empty circuit runs under the same donors
  recovery   per pool, donor_test.pooled: (mean over draws of the circuit's mean read - the same for the empty circuit)
             / (natural mean read - mean empty), None when undefined; empty_leak = mean empty / natural
  pass       resample_pass = the production pass on the current rule (wcm_eval.passes_task: Z, A, C in BAND, necessity
             >= NEC_MIN, and not vacuous) AND recovery A and C ("fit") each in BAND

Inputs: circuit JSONs under each --out folder (circuits/**.json, written by wcm_marks.run), their production rows
from that folder's eval.jsonl (matched on task / data mode / arm / lambda / steps / sites / means). Rows go to
<out>/donor/rows.jsonl, one per circuit, resumable (a scored circuit is skipped); STOP_EPOCH / STOP_AT respected (no
new task group starts after the deadline).

  HF_HUB_OFFLINE=1 ./.venv/bin/python experiments/107-true-baselines/arena_c/donor_tasks.py --out <dir> [<dir> ...]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wcm_marks as WM  # noqa: E402
import torch  # noqa: E402

from eval import donor_test as DT  # noqa: E402

VERSION = "donor-tasks/v1 (arena C; donor_test section 12.3 on the task read)"
SEED = DT.SEED
DRAWS = int(os.environ.get("DONOR_DRAWS", DT.DRAWS))
POOLS = DT.POOLS
VARIANTS = ("fit", "a1")
BASE_BATCH = int(os.environ.get("DONOR_BATCH", 10))     # bases per forward (donor codes are dense at k = d_sae)
R_POOL_N = 256
R_SEED = DT.POOL_SEED
CORPUS = WM.ROOT / "data_public" / "pythia-70m"


# ----------------------------------------------------------------------------------------------- pieces
def task_seed(task: str) -> int:
    return SEED + zlib.crc32(task.encode()) % 100000


def draws_for(task: str, pool_sizes: Mapping[str, int], n_bases: int, draws: int = DRAWS):
    """{pool: [selection [n_bases] per draw]}, short: donor_test.draw_selections with the task's seed."""
    return DT.draw_selections(0, 0, pool_sizes, n_bases, 0, draws=draws, seed=task_seed(task), pools=POOLS)


def corpus_windows(T: int, exclude: set, n: int = R_POOL_N, seed: int = R_SEED, corpus: Path = CORPUS):
    """(windows [n, T] long, info): n random corpus sequences (seeded), first T tokens, minus any equal to a task
    prompt. The shard format is 069's: [lead, t1..t64, -1, ...]; a sequence is t1..t64."""
    import numpy as np
    seqs = []
    for p in sorted(corpus.glob("shard_*.npy")):
        a = np.load(p)
        cut = np.nonzero(a == -1)[0]
        start = 0
        for c in cut:
            s = a[start + 1:c]                          # drop the lead token
            if len(s) >= T:
                seqs.append(s[:T])
            start = c + 1
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(seqs), generator=g).tolist()
    out, dropped = [], 0
    for j in order:
        w = tuple(int(x) for x in seqs[j])
        if w in exclude:
            dropped += 1
            continue
        out.append(w)
        if len(out) >= n:
            break
    return torch.tensor(out, dtype=torch.long), dict(n_corpus=len(seqs), n=len(out), dropped_equal_prompt=dropped,
                                                     seed=seed, corpus=str(corpus.relative_to(WM.ROOT)))


class DonorPatcher:
    """donor_test.ResamplePatcher at K = d_sae, delta form, no seed tap (see the module docstring)."""

    def __init__(self, bank, sites, keep_t, scales, donor):
        self.bank, self.sites, self.keep_t, self.scales, self.donor = bank, set(sites), keep_t, scales, donor

    def __call__(self, model):
        from model.hooks import multi_patch
        return multi_patch(model, self.transform)

    def transform(self, layer, kind, x):
        from sae.dense import sparse_topk_to_dense
        s = (layer, kind)
        if s not in self.sites:
            return x
        ta, ti = self.bank.encode(x, kind, layer)
        live = sparse_topk_to_dense(ta, ti, int(self.bank.d_sae), dtype=x.dtype)
        code = self.donor[s][:x.shape[0]].to(live.dtype).clone()
        kt = self.keep_t.get(s)
        if kt is not None:
            v = live[:, :, kt]
            sv = self.scales.get(s)
            code[:, :, kt] = v * sv.to(v.dtype) if sv is not None else v
        return x + self.bank.decode(code - live, kind, layer, add_bias=False).to(x.dtype)


@torch.no_grad()
def capture(inf, bank, tokens, sites) -> Dict:
    """{site: dense code [B, T, d_sae] float32} of `tokens` (natural run)."""
    from sae.dense import sparse_topk_to_dense
    k2i = {k: i for i, k in enumerate(bank.kinds)}
    want, out = set(sites), {}

    def hook(layer, acts):
        for kd in bank.kinds:
            if (layer, kd) in want:
                ta, ti = bank.encode(acts[k2i[kd]], kd, layer)
                out[(layer, kd)] = sparse_topk_to_dense(ta, ti, int(bank.d_sae), dtype=torch.float32)

    inf.forward(tokens, activations_callback=hook, return_activations=False, tokenize_final=False)
    return out


@torch.no_grad()
def read(inf, tokens, good, bad, patcher=None) -> torch.Tensor:
    """Task metric [B] (logit diff at the last position; every prompt has length T, no padding)."""
    from eval.wcm_eval import task_metric
    o = inf.forward(tokens, patcher=patcher, all_logits=True, return_activations=False, tokenize_final=False)
    lg = o[1] if isinstance(o, (tuple, list)) else o
    lp = torch.log_softmax(lg[:, -1].float(), dim=-1)
    return task_metric(lp, "logit_diff", good.to(lp.device), bad.to(lp.device)).cpu()


def keep_tensors(alphas: Mapping, device, unit: bool = False):
    kt = {s: torch.tensor(sorted(d), dtype=torch.long, device=device) for s, d in alphas.items() if d}
    sc = {} if unit else {s: torch.tensor([float(alphas[s][int(i)]) for i in kt[s].tolist()], device=device)
                          for s in kt}
    return kt, sc


# ----------------------------------------------------------------------------------------------- one task group
def task_context(task: str, mode: str, inf, bank, r_pool_n: int = R_POOL_N) -> Dict:
    """Bases, answers, the three pools and the natural read of one (task, data mode)."""
    T = WM.build_task(task, mode)
    dev = inf.device
    rows = T["rows"]
    tr, ho = T["split"]["train"], T["split"]["held"]
    L = len(rows[tr[0]]["tokens"])
    for i in tr + ho:
        assert len(rows[i]["tokens"]) == L and len(rows[i]["cf_tokens"]) == L, "every prompt must have length T=%d" % L
    t = lambda ids, f: torch.tensor([rows[i][f] for i in ids], dtype=torch.long, device=dev)  # noqa: E731
    bases = t(ho, "tokens")
    good = torch.tensor([rows[i]["good"] for i in ho])
    bad = torch.tensor([rows[i]["bad"] for i in ho])
    excl = {tuple(rows[i][f]) for i in tr + ho for f in ("tokens", "cf_tokens")}
    rp, rinfo = corpus_windows(L, excl, r_pool_n)
    pools = dict(A=t(tr, "tokens"), C=t(ho, "cf_tokens"), R=rp.to(dev))
    for nm, p in pools.items():
        assert int(p.shape[1]) == L, "donor pool %s has length %d, not T=%d" % (nm, int(p.shape[1]), L)
    natural = torch.cat([read(inf, bases[b:b + BASE_BATCH], good[b:b + BASE_BATCH], bad[b:b + BASE_BATCH])
                         for b in range(0, int(bases.shape[0]), BASE_BATCH)])
    sels, short = draws_for(task, {k: int(v.shape[0]) for k, v in pools.items()}, int(bases.shape[0]))
    return dict(task=task, mode=mode, T=L, bases=bases, good=good, bad=bad, pools=pools, natural=natural, sels=sels,
                short=short, r_pool=rinfo)


def score_group(inf, bank, ctx: Mapping, sites: Sequence, circuits: Sequence[Mapping], log=print) -> List[Dict]:
    """Donor reads of the empty circuit and of every circuit (variants fit / a1) under the SAME donors; one dict per
    circuit with recovery {pool: {variant: value}}, empty_leak, and the per-draw means."""
    dev = inf.device
    n = int(ctx["bases"].shape[0])
    kts = [(keep_tensors(c["alphas"], dev), keep_tensors(c["alphas"], dev, unit=True)) for c in circuits]
    e_means = {ds: [] for ds in POOLS}
    x_means = [{ds: {v: [] for v in VARIANTS} for ds in POOLS} for _ in circuits]
    for ds in POOLS:
        for sel in ctx["sels"][ds]:
            e_rows, x_rows = [], [{v: [] for v in VARIANTS} for _ in circuits]
            for b in range(0, n, BASE_BATCH):
                bt = ctx["bases"][b:b + BASE_BATCH]
                g, bd = ctx["good"][b:b + BASE_BATCH], ctx["bad"][b:b + BASE_BATCH]
                donor = capture(inf, bank, ctx["pools"][ds][sel[b:b + BASE_BATCH].to(dev)], sites)
                e_rows.append(read(inf, bt, g, bd, DonorPatcher(bank, sites, {}, {}, donor)))
                for j, ((kt, sc), (kt1, _)) in enumerate(kts):
                    x_rows[j]["fit"].append(read(inf, bt, g, bd, DonorPatcher(bank, sites, kt, sc, donor)))
                    x_rows[j]["a1"].append(read(inf, bt, g, bd, DonorPatcher(bank, sites, kt1, {}, donor)))
                del donor
            e_means[ds].append(float(torch.cat(e_rows).double().mean()))
            for j in range(len(circuits)):
                for v in VARIANTS:
                    x_means[j][ds][v].append(float(torch.cat(x_rows[j][v]).double().mean()))
    nat = float(ctx["natural"].double().mean())
    out = []
    for j, c in enumerate(circuits):
        rec = {ds: {v: DT.pooled(x_means[j][ds][v], e_means[ds], nat) for v in VARIANTS} for ds in POOLS}
        out.append(dict(recovery=rec, natural=nat, draws=DRAWS, n_bases=n,
                        empty_leak={ds: (sum(e_means[ds]) / len(e_means[ds]) / nat if nat > DT.UNDEFINED_EPS else None)
                                    for ds in POOLS},
                        empty_draw_means=e_means, circuit_draw_means=x_means[j]))
    return out


def resample_pass(prod: Optional[Mapping], recovery_fit: Mapping) -> bool:
    from eval.wcm_eval import passes_task
    if prod is None or prod.get("task_vacuous"):
        return False
    return bool(passes_task(prod) and DT.in_band(recovery_fit.get("A")) and DT.in_band(recovery_fit.get("C")))


# ----------------------------------------------------------------------------------------------- CLI
def _prod_rows(out: Path) -> Dict[Tuple, dict]:
    rows = {}
    p = out / "eval.jsonl"
    if p.exists():
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            rows[(r.get("sfc_task"), r.get("data_mode"), r.get("arm"), float(r.get("lam")), int(r.get("steps")),
                  r.get("sites"), r.get("means", "pooled"))] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", nargs="+", required=True)
    a = ap.parse_args()
    from eval.wcm_eval import passes_task
    C, _ = WM._import_081()
    started = _dt.datetime.now()
    inf = bank = None
    for od in map(Path, a.out):
        dd = od / "donor"
        dd.mkdir(parents=True, exist_ok=True)
        rows_p = dd / "rows.jsonl"
        done = set()
        if rows_p.exists():
            for line in open(rows_p):
                try:
                    done.add(json.loads(line)["circuit"])
                except Exception:  # noqa: BLE001
                    pass
        prod = _prod_rows(od)
        groups = defaultdict(list)
        for cp in sorted((od / "circuits").rglob("*.json")):
            rel = str(cp.relative_to(od))
            if rel in done:
                continue
            c = json.loads(cp.read_text())
            c["alphas"] = C.alphas_from_json(c["alphas"])
            c["_rel"] = rel
            groups[(c["task"], c["data_mode"], c.get("sites", "all"))].append(c)
        for (task, mode, sites), cs in sorted(groups.items()):
            if WM.stop_passed(os.environ.get("STOP_AT"), started):
                print("deadline passed: stopping", flush=True)
                return
            if inf is None:
                inf, bank = WM.load_model()
            ts = time.time()
            ctx = task_context(task, mode, inf, bank)
            res = score_group(inf, bank, ctx, WM.scope_sites(sites), cs)
            secs = round(time.time() - ts, 1)
            for c, r in zip(cs, res):
                key = (task, mode, c["arm"], float(c["lam"]), int(c["steps"]), sites, c.get("means", "pooled"))
                p = prod.get(key)
                row = dict(version=VERSION, circuit=c["_rel"], task=task, data_mode=mode, arm=c["arm"], lam=c["lam"],
                           steps=c["steps"], sites=sites, means=c.get("means", "pooled"), wcm=c.get("wcm"),
                           n_latents=sum(len(d) for d in c["alphas"].values()), T=ctx["T"], r_pool=ctx["r_pool"],
                           pool_short=ctx["short"], secs_group=secs, **r,
                           prod={k: (p or {}).get(k) for k in ("task_free0", "task_freeM_topk", "task_freeN_topk",
                                                                "task_nec", "task_vacuous", "task_full")},
                           prod_pass=None if p is None else bool(not p.get("task_vacuous") and passes_task(p)))
                row["resample_pass"] = resample_pass(p, {ds: r["recovery"][ds]["fit"] for ds in POOLS})
                with open(rows_p, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                rv = r["recovery"]
                print("%s | %d latents | rec fit A %s C %s R %s | a1 A %s C %s R %s | leak %s | resample_pass %s" % (
                    c["_rel"], row["n_latents"], *(_f(rv[d]["fit"]) for d in POOLS), *(_f(rv[d]["a1"]) for d in POOLS),
                    {d: _f(v) for d, v in row["empty_leak"].items()}, row["resample_pass"]), flush=True)


def _f(x):
    return "None" if x is None else "%.3f" % x


if __name__ == "__main__":
    main()

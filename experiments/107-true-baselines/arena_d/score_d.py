"""Arena D4: the scorer for D2 (Caples et al.'s masks and WCM circuits on Gemma-2-2B + Gemma Scope residual SAEs, gap 6).
Every circuit is scored both ways (107 rule 5). Experiment code: nothing under src/ or third_party/ is modified.

Circuits (one in-memory form for both kinds, `circuit` dicts):
  caples   their run folder: sae_mask_{i}.pt (SparseMask.state_dict(), {"mask": theta [T, d_sae]}, i = the i-th SAE of
           their list, layers ascending) and <lam>_run.json (checked when present: per hook the indices
           torch.where(theta > 0)[1], and total_density). A (position, latent) pair is a member when theta > 0 (their
           read-out). Per-position membership: keep [T, d_sae].
  wcm      wcm_gemma.run's circuit JSON (alphas {site: {latent: alpha}}): position-collapsed, keep [1, d_sae]
           (broadcast over positions), amplitude alpha per member.
  Sizes: n_latents = distinct (site, latent) with a member at some position >= 1 (BOS is never edited, so a BOS pair has
  no effect); pairs_nonbos = (site, position >= 1, latent) pairs (a collapsed circuit: n x (T - 1), wcm_gemma.sizes);
  pairs_all = every theta > 0 pair incl. BOS (= their total_density; collapsed: n x T); n_bos_pairs.

OURS (repo .venv, `ours`): our tests on the deduplicated held-out rc_test subset (wcm_gemma.build_task data "caples",
  400 records), per substrate:
    native   SAE error live (x + decode(c_hat) - decode(c)).
    caples   their substrate (decode(c_hat) + eps_bar[pos], BOS stream kept), eps_bar and the C fill from
             caples_means.py's file (source "theirs" = their functions, running-mean bug kept; "exact" optional row).
  Frames (score_task's field names and normalisations, task_*): full; e0 / eM_topk / eN_topk = the empty circuit at the
  Z (zero) / A (our per-position mean of the training clean prompts) / C fill (native: our per-position corrupted mean;
  caples: their file's code mean); free0 / freeM_topk / freeN_topk = members at alpha x their re-encoded live value,
  every other (site, position, latent) at the fill; ratios (m(C) - m(e)) / (m(full) - m(e)). The bank's k = d_sae, so
  the "_topk" (sparsity-preserving) fill is the plain mean fill (asserted). Necessity (DAN-7 form, (m(full) - m(nec)) /
  (m(full) - m(e0))), two columns (Daniel 2026-11-05, as arena A):
    task_nec           (a) members set to 0, everything else live, the error live (score_task's reading);
    task_nec_errclean  (b) members set to 0, the SAE error at EVERY dictionary site held at its value in the clean run
                       of the same prompt (site output = decode(c_hat_live) + err_clean). In Caples mode the error is
                       eps_bar by definition of the substrate in the clean and the edited run alike, so (b) = (a) and is
                       not recomputed (`nec_errclean_note`). Residual family only (the PLT site's error lives at the MLP
                       output; refused there).
  The per-position path is our own patcher (PPPatcher); for a position-collapsed circuit it reproduces score_task
  (checked in test_score_d.py), and a collapsed circuit broadcast to [T, d_sae] scores identically.
  `their_protocol` (Caples mode only, ours, diagnostic): their eval re-stated in our HF stack: binary membership (alpha
  dropped), the C fill, on their eval batches [0, -1..-9] of rc_train; must equal THEIRS' ld (cross-stack check).

THEIRS (caples107, `theirs`): their helpers/utils.py, unmodified: the model and SAEs loaded as caples_means.py does
  (their selection; fixture for tests), sae.mean_ablation / sae.mean_error = caples_means.py's saved output of their
  get_sae_means / get_sae_error_means ([1, T, d], as their functions leave them; --recompute-means calls them again),
  each circuit as utils.SAEMasks(hook_points, masks) (Caples: (theta > 0).float() [T, d]; WCM: members broadcast over
  positions, alpha dropped since SAEMasks is binary), LD = utils.logit_diff_fn on run_sae_hook_fn(model, saes, batch,
  circuit_mask=..., mean_mask=True, use_mean_error=True), averaged over their eval batches batch[-i], i = 0..9.
  Reported: ld, f_model = ld / ld(true model) (their JSON's `faithfulness`), f_spliced = ld / ld(SAE model, mean error,
  no mask), and the SFC-normalised sfc_* = (ld - ld_empty) / (ld_ref - ld_empty), m(empty) = every latent at its mean
  (SAEMasks of zeros) on the same batches.

CLI (repo root; CPU / GPU as the environment says):
  ./.venv/bin/python experiments/107-true-baselines/arena_d/score_d.py all --caples-means out/caples_means/rc_gap6.pt \
      --caples-dirs <run copy>/masks/sva/rc_train_saegap6_mean_error/logit_diff_0.0224_run --wcm-circuits <circuit.json>
      [--modes native,caples] [--caples-source theirs|exact] [--out DIR]       (ours, then theirs in caples107, merged)
  ~/venvs/caples107/bin/python -I score_d.py theirs --masks DIR/masks.pt --caples-means PT --out DIR/theirs.json \
      [--fixture FX] [--device cuda] [--source exact] [--recompute-means]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CAPLES_SRC = ROOT / "third_party" / "ScalableSAECircuits"
CAPLES_PY = Path.home() / "venvs" / "caples107" / "bin" / "python"
OUT_DEFAULT = HERE / "out" / "score_d"
MASKS_FORMAT = "arena-d-score-masks-v1"
EVAL_BS = 16
N_EVAL_BATCHES = 10                       # their eval_logit_diff(10, ...): batch[-i], i = 0..9


# ======================================================================================================= circuits
def _hook(layer: int) -> str:
    return "blocks.%d.hook_resid_post" % layer


def caples_circuit(run_dir, layers, check_json: bool = True) -> Dict:
    """Their run folder -> circuit (per-position keep = theta > 0). layers = their SAE list's layers, in order."""
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob("sae_mask_*.pt"))
    if len(files) != len(layers):
        raise ValueError("%s: %d sae_mask files, %d SAE layers %s" % (run_dir, len(files), len(layers), list(layers)))
    keep, theta = {}, {}
    for i, l in enumerate(layers):
        sd = torch.load(run_dir / ("sae_mask_%d.pt" % i), map_location="cpu", weights_only=True)
        th = sd["mask"].float()
        if th.dim() != 2:
            raise ValueError("%s sae_mask_%d: theta shape %s, need [T, d_sae] (--per_token_mask)"
                             % (run_dir, i, tuple(th.shape)))
        theta[int(l)] = th
        keep[int(l)] = th > 0
    T = int(next(iter(keep.values())).shape[0])
    meta = dict(dir=str(run_dir), json_checked=False)
    js = sorted(run_dir.glob("*_run.json"))
    if check_json and js:
        d = json.loads(js[0].read_text())
        for l in layers:
            if _hook(l) in d and d[_hook(l)] != torch.where(theta[int(l)] > 0)[1].tolist():
                raise ValueError("%s: theta > 0 indices at %s differ from %s" % (run_dir, _hook(l), js[0].name))
        tot = sum(int(k.sum()) for k in keep.values())
        if "total_density" in d and int(d["total_density"]) != tot:
            raise ValueError("%s: total_density %s != %d" % (run_dir, d["total_density"], tot))
        meta.update(json_checked=True, json=js[0].name, their_faithfulness=d.get("faithfulness"),
                    their_logit_diff=d.get("logit_diff"), their_total_density=d.get("total_density"))
    return dict(name="caples:" + run_dir.name, source="caples", T=T, layers=[int(l) for l in layers], keep=keep,
                scale={l: torch.ones_like(k, dtype=torch.float32) for l, k in keep.items()}, meta=meta)


def wcm_circuit(path, d_sae: int, name: Optional[str] = None) -> Dict:
    """wcm_gemma.run's circuit JSON -> circuit (position-collapsed: keep / scale [1, d_sae])."""
    path = Path(path)
    rec = json.loads(path.read_text())
    keep, scale = {}, {}
    for k, v in rec["alphas"].items():
        l, kind = k.split("/")
        if kind != "resid":
            raise ValueError("%s: site kind %r (arena D4 scores the residual family)" % (path, kind))
        kp = torch.zeros(1, d_sae, dtype=torch.bool)
        sc = torch.ones(1, d_sae)
        for i, a in v.items():
            kp[0, int(i)] = True
            sc[0, int(i)] = float(a)
        keep[int(l)], scale[int(l)] = kp, sc
    meta = {k: rec.get(k) for k in ("task", "data_mode", "lam", "steps", "means", "substrate", "c_fill", "family")}
    meta["path"] = str(path)
    return dict(name=name or ("wcm:" + path.stem), source="wcm", T=None, layers=sorted(keep), keep=keep, scale=scale,
                meta=meta)


def keep_all_circuit(layers, T: int, d_sae: int) -> Dict:
    keep = {int(l): torch.ones(T, d_sae, dtype=torch.bool) for l in layers}
    return dict(name="keep_all", source="control", T=T, layers=list(keep), keep=keep,
                scale={l: torch.ones(T, d_sae) for l in keep}, meta={})


def empty_circuit() -> Dict:
    return dict(name="empty", source="control", T=None, layers=[], keep={}, scale={}, meta={})


def circuit_sizes(c: Dict, T: int) -> Dict:
    n_lat, pairs_nb, pairs_all, bos = 0, 0, 0, 0
    per_site = {}
    for l, k in sorted(c["keep"].items()):
        kk = k.expand(T, -1) if k.shape[0] == 1 else k
        if int(kk.shape[0]) != T:
            raise ValueError("%s: mask has %d positions, prompts %d" % (c["name"], kk.shape[0], T))
        n = int(kk[1:].any(0).sum())
        per_site["resid_%d" % l] = n
        n_lat += n
        pairs_nb += int(kk[1:].sum())
        pairs_all += int(kk.sum())
        bos += int(kk[0].sum())
    return dict(n_latents=n_lat, pairs_nonbos=pairs_nb, pairs_all=pairs_all, n_bos_pairs=bos,
                latents_per_site=per_site)


def write_masks(circuits: List[Dict], T: int, path) -> Path:
    """The binary masks THEIRS applies through SAEMasks: {name: {layer: bool [T, d_sae]}} (collapsed: broadcast)."""
    out = {}
    for c in circuits:
        out[c["name"]] = {int(l): (k.expand(T, -1) if k.shape[0] == 1 else k).clone() for l, k in c["keep"].items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(format=MASKS_FORMAT, T=int(T), circuits=out), path)
    return path


# ======================================================================================================= OURS
class PPPatcher:
    """One frame of our tests with per-position membership. keep / scale {site: [T or 1, d_sae]} (absent site: no
    member). frame "circuit": c_hat = where(keep, scale * c, fill) (fill None = zero, [d] or [T, d]); "remove": c_hat =
    where(keep, 0, c) (necessity). err {site: [B, T, d]} given: the site outputs decode(c_hat) + err (the error held at
    that value); else x + (decode(c_hat) - decode(c)) (the engine's form; an all-member circuit is the identity
    exactly). Sites outside `sites` are untouched; under "remove" without err, a site with no member too."""

    def __init__(self, bank, sites, keep, scale, frame: str = "circuit", fill=None, err=None):
        if frame not in ("circuit", "remove"):
            raise ValueError(frame)
        self.bank, self.sites, self.frame, self.fill, self.err = bank, set(sites), frame, fill, err
        self.keep, self.scale = keep, scale
        self.edited = set()

    def __call__(self, model):
        from model.hooks import multi_patch
        return multi_patch(model, self.transform)

    def transform(self, layer: int, kind: str, x: torch.Tensor) -> torch.Tensor:
        from sae.dense import sparse_topk_to_dense
        s = (layer, kind)
        if s not in self.sites:
            return x
        kp = self.keep.get(s)
        if self.frame == "remove" and kp is None and self.err is None:
            return x
        T = int(x.shape[1])
        if kp is not None:
            if int(kp.shape[0]) not in (1, T):
                raise ValueError("site %s: mask has %d positions, the prompts %d" % (s, kp.shape[0], T))
            kp = kp.to(x.device)
        ta, ti = self.bank.encode(x, kind, layer)
        c = sparse_topk_to_dense(ta, ti, self.bank.d_sae, dtype=x.dtype)
        if self.frame == "remove":
            ch = c if kp is None else c.masked_fill(kp, 0.0)
        else:
            f = self.fill[s].to(device=x.device, dtype=x.dtype) if self.fill is not None else None
            if f is not None and f.dim() == 2 and int(f.shape[0]) != T:
                raise ValueError("site %s: fill has %d positions, the prompts %d" % (s, f.shape[0], T))
            if f is None:
                f = torch.zeros((), device=x.device, dtype=x.dtype)
            if kp is None:
                ch = f.expand_as(c)
            else:
                ch = torch.where(kp, self.scale[s].to(device=x.device, dtype=x.dtype) * c, f)
        self.edited.add(s)
        if self.err is not None:
            return self.bank.decode(ch, kind, layer).to(x.dtype) + self.err[s]
        return x + (self.bank.decode(ch, kind, layer) - self.bank.decode(c, kind, layer)).to(x.dtype)


def _site_dict(c: Dict, kind: str = "resid"):
    return ({(l, kind): k for l, k in c["keep"].items()}, {(l, kind): v for l, v in c["scale"].items()})


def clean_errors(eng, tokens) -> Dict:
    """{site: [B, T, d]} the SAE error x - decode(c) at every dictionary site in the clean (unedited) run."""
    from sae.dense import sparse_topk_to_dense
    k2i = {k: i for i, k in enumerate(eng.kinds)}
    want = set(eng.sites)
    err = {}

    def hook(layer, acts):
        for kd in eng.kinds:
            s = (layer, kd)
            if s in want:
                x = acts[k2i[kd]]
                ta, ti = eng.bank.encode(x, kd, layer)
                c = sparse_topk_to_dense(ta, ti, eng.bank.d_sae, dtype=x.dtype)
                err[s] = (x - eng.bank.decode(c, kd, layer).to(x.dtype)).detach()

    with torch.no_grad():
        eng.inf.forward(tokens, activations_callback=hook, return_activations=False, tokenize_final=False)
    return err


def use_mode(eng) -> None:
    """Put the shared site model in this engine's substrate (engines of both modes share one inference)."""
    import wcm_gemma as W
    if isinstance(eng.inf.model, W.GemmaResSiteModel):
        W.apply_caples(eng.inf, eng.bank, eng.cm if eng.substrate == "caples" else None)


def score_ours(eng, B: Dict, c: Dict, errclean: bool = True, batch_size: int = EVAL_BS) -> Dict:
    """Our tests for one circuit on B's held-out rows in eng's substrate (see the module docstring)."""
    from eval import wcm_eval as WE
    import wcm_gemma as W
    use_mode(eng)
    ts = time.time()
    bank = eng.bank
    if int(bank.k) != int(bank.d_sae):
        raise ValueError("score_ours assumes k = d_sae (sparse fill = plain mean fill); bank k %s" % bank.k)
    kind_of = W.SPECS[eng.family]["kind"]
    sites = set(eng.sites)
    keep, scale = _site_dict(c, kind_of)
    bad_sites = set(keep) - sites
    if bad_sites:
        raise ValueError("%s: members at sites %s outside the bank's dictionary sites" % (c["name"], sorted(bad_sites)))
    n = B["n_train"]
    toks, anc = B["tokens"][n:], B["anchors"][n:]
    kind = B["metric"]["task"]
    g_ho = WE._token_sets(B["good"].cpu())[n:]
    b_ho = WE._token_sets(B["bad"].cpu())[n:]
    M = eng.means(B)

    def rd(make):
        lp, _, _ = WE.read_rows(eng.inf, make, toks, anc, None, batch_size)
        v = WE.task_metric(lp, kind, g_ho, b_ho)
        return float(v.mean()), float((v > 0).float().mean())

    def co(kp, fill):
        return rd(lambda a, b: PPPatcher(bank, sites, kp, scale, "circuit", fill))

    val, acc = {}, {}
    for f, mk in (("full", None),
                  ("e0", lambda a, b: PPPatcher(bank, sites, {}, {}, "circuit", None)),
                  ("eM_topk", lambda a, b: PPPatcher(bank, sites, {}, {}, "circuit", M["A"])),
                  ("eN_topk", lambda a, b: PPPatcher(bank, sites, {}, {}, "circuit", M["C"]))):
        val[f], acc[f] = rd(mk)
    for f, e, fill in (("free0", "e0", None), ("freeM_topk", "eM_topk", M["A"]), ("freeN_topk", "eN_topk", M["C"])):
        val[f], acc[f] = co(keep, fill) if keep else (val[e], acc[e])
    val["nec"], acc["nec"] = (rd(lambda a, b: PPPatcher(bank, sites, keep, scale, "remove")) if keep
                              else (val["full"], acc["full"]))
    note = None
    if eng.substrate == "caples":
        val["nec_errclean"], acc["nec_errclean"] = val["nec"], acc["nec"]
        note = "caples substrate: the site error is eps_bar in the clean and the edited run, so (b) = (a)"
    elif errclean and kind_of == "resid":
        def mk(a, b):
            return PPPatcher(bank, sites, keep, scale, "remove", err=clean_errors(eng, toks[a:b]))
        val["nec_errclean"], acc["nec_errclean"] = rd(mk)
    else:
        val["nec_errclean"] = acc["nec_errclean"] = None
        note = "not computed (%s)" % ("transcoder family" if kind_of != "resid" else "errclean off")
    full, e0 = val["full"], val["e0"]
    row = dict(circuit=c["name"], source=c["source"], substrate=eng.substrate if eng.substrate == "native" else
               "caples_" + eng.cm["source"], means=eng.means_mode, c_fill=eng.c_fill, n_ho=int(toks.shape[0]),
               task_kind=kind, task_full=full, task_e0=e0, task_eM_topk=val["eM_topk"], task_eN_topk=val["eN_topk"],
               task_den=full - e0)
    for f, ek in (("free0", "e0"), ("freeM_topk", "eM_topk"), ("freeN_topk", "eN_topk")):
        row["task_%s_raw" % f] = val[f]
        row["task_%s" % f] = WE.ratio(val[f], val[ek], full)
    for f in ("nec", "nec_errclean"):
        row["task_%s_raw" % f] = val[f]
        row["task_%s" % f] = None if val[f] is None else WE.necessity(val[f], full, e0)
    row["nec_errclean_note"] = note
    for f, a in acc.items():
        row["task_acc_%s" % f] = a
    row["task_pass"] = WE.passes_task(row, "task")
    row["secs"] = round(time.time() - ts, 1)
    return row


def their_batches(task: str, tok, n: int, batch_size: int, length: int) -> List:
    """Their eval batches batch[-i], i = 0..9, of <task>_train under their filter (interp_bottlenecks.py:180-228 and
    do_training_run's eval): [(tokens [bs, T], clean answer ids, patch answer ids)]."""
    import wcm_gemma as W
    recs = W._caples_filter(CAPLES_SRC / "data" / "sva" / ("%s_train.json" % task), tok, length, n)
    nb = len(recs) // batch_size
    out = []
    for i in range(N_EVAL_BATCHES):
        b = recs[((-i) % nb) * batch_size:((-i) % nb + 1) * batch_size]
        out.append((torch.tensor([tok(e["clean_prefix"]).input_ids for e in b]),
                    torch.tensor([W._answer_id(tok, e["clean_answer"]) for e in b]),
                    torch.tensor([W._answer_id(tok, e["patch_answer"]) for e in b])))
    return out


@torch.no_grad()
def their_protocol_ours(eng, B: Dict, c: Dict, batches) -> Dict:
    """[ours, diagnostic] their evaluation re-stated in our HF stack (Caples substrate): binary membership, alpha
    dropped, every non-member at the C fill (their file's code mean), LD = mean over batches of the batch-mean
    logit difference; refs: the spliced model (open gate), the true model, the empty mask."""
    import wcm_gemma as W
    if eng.substrate != "caples":
        raise ValueError("their protocol is defined on the Caples substrate only")
    kind_of = W.SPECS[eng.family]["kind"]
    sites = set(eng.sites)
    keep, _ = _site_dict(c, kind_of)
    ones = {s: torch.ones_like(k, dtype=torch.float32) for s, k in keep.items()}
    M = eng.means(B)
    dev = eng.dev

    def ld(make, caples_on=True):
        W.apply_caples(eng.inf, eng.bank, eng.cm if caples_on else None)
        tot = []
        for x, cl, pl in batches:
            x = x.to(dev)
            _, lg, _ = eng.inf.forward(x, patcher=None if make is None else make(), tokenize_final=False,
                                       return_activations=False, all_logits=True, grad_enabled=False)
            r = torch.arange(x.shape[0], device=lg.device)
            tot.append(float((lg[r, -1, cl.to(lg.device)] - lg[r, -1, pl.to(lg.device)]).float().mean()))
        W.apply_caples(eng.inf, eng.bank, eng.cm)
        return sum(tot) / len(tot), tot

    ld_c, per = ld(lambda: PPPatcher(eng.bank, sites, keep, ones, "circuit", M["C"]))
    ld_e, _ = ld(lambda: PPPatcher(eng.bank, sites, {}, {}, "circuit", M["C"]))
    ld_s, _ = ld(None)
    ld_m, _ = ld(None, caples_on=False)
    return dict(ld=ld_c, ld_per_batch=per, ld_empty=ld_e, ld_spliced=ld_s, ld_model=ld_m, **_ratios(ld_c, ld_e, ld_s,
                                                                                                         ld_m))


def _ratios(ld, ld_e, ld_s, ld_m) -> Dict:
    def r(a, b):
        return a / b if abs(b) > 1e-12 else None

    return dict(f_model=r(ld, ld_m), f_spliced=r(ld, ld_s), sfc_model=r(ld - ld_e, ld_m - ld_e),
                sfc_spliced=r(ld - ld_e, ld_s - ld_e))


def run_ours(engines: Dict, task: Dict, circuits: List[Dict], out: Path, errclean: bool = True,
             their_eval: Optional[List] = None, log=print) -> List[Dict]:
    """Score every circuit under every engine (mode); resumable on (circuit, substrate, means) in ours.jsonl."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "ours.jsonl"
    done = set()
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line)
                done.add((r["circuit"], r["substrate"], r["means"]))
            except Exception:  # noqa: BLE001  (a truncated last line after a kill)
                pass
    T = int(task["prompt_len_train"][0])
    rows = []
    for mode, eng in engines.items():
        B = eng.batch(task)
        use_mode(eng)
        eng.means(B)
        sub = eng.substrate if eng.substrate == "native" else "caples_" + eng.cm["source"]
        for c in circuits:
            if (c["name"], sub, eng.means_mode) in done:
                log("skip (scored) %s %s" % (c["name"], sub))
                continue
            row = score_ours(eng, B, c, errclean=errclean)
            row.update(circuit_sizes(c, T), prompt_len=T, task=task["name"], split=task["split_meta"]["rule"],
                       circuit_meta=c["meta"])
            if their_eval is not None and eng.substrate == "caples":
                row["their_protocol"] = their_protocol_ours(eng, B, c, their_eval)
            with open(path, "a") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
            rows.append(row)
            log("%s %s: %d latents / %d pairs | Z %s A %s C %s nec %s nec_errclean %s" % (
                c["name"], sub, row["n_latents"], row["pairs_nonbos"], _f(row["task_free0"]),
                _f(row["task_freeM_topk"]), _f(row["task_freeN_topk"]), _f(row["task_nec"]),
                _f(row["task_nec_errclean"])))
    return rows


def _f(x):
    return "None" if x is None else "%.3f" % x


def merge(ours_rows: List[Dict], theirs: Optional[Dict], out: Path) -> List[Dict]:
    """One line per circuit: sizes, ours by substrate, theirs."""
    by = {}
    for r in ours_rows:
        m = by.setdefault(r["circuit"], dict(circuit=r["circuit"], source=r["source"], ours={},
                                             **{k: r[k] for k in ("n_latents", "pairs_nonbos", "pairs_all",
                                                                  "n_bos_pairs", "latents_per_site")}))
        m["ours"][r["substrate"] + "/" + r["means"]] = {k: v for k, v in r.items() if k.startswith("task_") or k in (
            "their_protocol", "nec_errclean_note", "c_fill", "n_ho")}
    if theirs is not None:
        for name, t in theirs["circuits"].items():
            if name in by:
                by[name]["theirs"] = dict(t, refs=theirs["refs"], label=theirs["meta"]["label"])
    rows = list(by.values())
    with open(Path(out) / "scores.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    return rows


def ours_main(a) -> int:
    sys.path.insert(0, str(HERE))
    import wcm_gemma as W
    W.register()
    family = a.family
    modes = [m for m in a.modes.split(",") if m]
    inf, bank = W.load_model(family, a.device, a.dtype)
    engines = {m: W.make_engine(family, m, a.means if m == "native" else "per_position", a.caples_means,
                                a.caples_source, shared=(inf, bank)) for m in modes}
    tok = W.tokenizer(W._F.FAMILIES[family].tokenizer_id)
    task = W.build_task(a.task, "caples", tok=tok, family=family, n_held=a.n_held)
    T = int(task["prompt_len_train"][0])
    layers = [l for l, _ in bank.real_sites]
    circuits = [caples_circuit(d, layers) for d in a.caples_dirs.split(",") if d]
    circuits += [wcm_circuit(p, bank.d_sae) for p in a.wcm_circuits.split(",") if p]
    if not a.no_controls:
        circuits += [keep_all_circuit(layers, T, bank.d_sae), empty_circuit()]
    names = [c["name"] for c in circuits]
    if len(set(names)) != len(names):
        raise ValueError("duplicate circuit names %s" % names)
    out = Path(a.out)
    ev = their_batches(a.task, tok, W.CAPLES_N, W.CAPLES_BS, T) if "caples" in modes else None
    rows = run_ours(engines, task, circuits, out, errclean=not a.no_errclean, their_eval=ev)
    write_masks(circuits, T, out / "masks.pt")
    if a.cmd == "all" and not a.no_theirs:
        cmd = [str(CAPLES_PY), "-I", str(Path(__file__).resolve()), "theirs", "--masks", str(out / "masks.pt"),
               "--caples-means", a.caples_means, "--source", a.caples_source, "--out", str(out / "theirs.json"),
               "--device", a.device or ("cuda" if torch.cuda.is_available() else "cpu")]
        env = dict(os.environ, WANDB_MODE="disabled")
        subprocess.run(cmd, check=True, env=env)
    allrows = [json.loads(l) for l in open(out / "ours.jsonl")]
    th = json.loads((out / "theirs.json").read_text()) if (out / "theirs.json").exists() else None
    merge(allrows, th, out)
    return 0


# ======================================================================================================= THEIRS
def theirs_main(a) -> int:
    """caples107: their utils, unmodified (see the module docstring)."""
    os.environ.setdefault("WANDB_MODE", "disabled")
    sys.dont_write_bytecode = True                      # importing their helpers must not write into third_party/
    sys.path.insert(0, a.caples_src)
    sys.path.insert(0, str(HERE))
    import helpers.utils as utils                       # theirs, unmodified
    import caples_means as CM                           # their model / SAE / data loading, re-stated (D3)
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[a.dtype]
    t0 = time.time()
    model, saes, ids, toks = CM.load_fixture(a, torch) if a.fixture else CM.load_real(a, torch, dtype)
    clean, corr, clean_lab, corr_lab = toks
    nb, bs, T = int(clean.shape[0]), int(clean.shape[1]), int(clean.shape[2])
    dev = a.device
    d = torch.load(a.caples_means, map_location="cpu", weights_only=False)
    if d.get("format") != CM.FORMAT:
        raise ValueError("%s: not a %s file" % (a.caples_means, CM.FORMAT))
    mt = d["mean_tokens"]
    mb = int(mt.shape[0]) // bs
    if not torch.equal(corr[:mb].reshape(-1, T).cpu().long(), mt.long()):
        raise ValueError("%s was made on other prompts than these patch batches 0..%d" % (a.caples_means, mb - 1))
    if a.recompute_means:                               # their functions again (pod default: the saved output)
        corr_d = corr.to(dev)
        saes = utils.get_sae_means(model, saes, corr_d, total_batches=mb, batch_size=bs, device=dev)
        saes = utils.get_sae_error_means(model, saes, corr_d, total_batches=mb, batch_size=bs, device=dev)
        means_src = "recomputed (their functions; running-mean bug kept)"
    else:
        src = d[a.source]
        for s in saes:
            L = int(s.cfg.hook_layer)
            s.mean_ablation = src[L]["code"].float().reshape(1, T, -1).to(dev)
            s.mean_error = src[L]["err"].float().reshape(1, T, -1).to(dev)
        means_src = "%s (caples_means.py file)" % a.source
    mk = torch.load(a.masks, map_location="cpu", weights_only=False)
    if mk.get("format") != MASKS_FORMAT or int(mk["T"]) != T:
        raise ValueError("%s: not a %s file for T = %d" % (a.masks, MASKS_FORMAT, T))
    hooks = [s.cfg.hook_name for s in saes]
    layers = [int(s.cfg.hook_layer) for s in saes]
    idx = [(-i) % nb for i in range(N_EVAL_BATCHES)]    # do_training_run's eval: batch[-i], i = 0..9
    amp = torch.autocast(device_type="cuda" if str(dev).startswith("cuda") else "cpu", dtype=dtype) \
        if dtype != torch.float32 else contextlib.nullcontext()

    def lds(fn):
        out = []
        with torch.no_grad(), amp:
            for i in idx:
                out.append(float(utils.logit_diff_fn(fn(clean[i].to(dev)), clean_lab[i].to(dev),
                                                     corr_lab[i].to(dev))))
        return sum(out) / len(out), out

    def masks_of(circ):
        extra = set(int(k) for k in circ) - set(layers)
        if extra:
            raise ValueError("mask layers %s are not SAE layers %s" % (sorted(extra), layers))
        return utils.SAEMasks(hook_points=hooks, masks=[
            (circ[L].float() if L in circ else torch.zeros(T, saes[j].cfg.d_sae)).to(dev)
            for j, L in enumerate(layers)])

    def masked(M):
        return lambda t: utils.run_sae_hook_fn(model, saes, t, circuit_mask=M, mean_mask=True, use_mean_error=True)[0]

    ld_m, per_m = lds(lambda t: model(t))
    ld_s, per_s = lds(lambda t: utils.run_sae_hook_fn(model, saes, t, use_mean_error=True)[0])
    ld_e, per_e = lds(masked(masks_of({})))
    res = {}
    for name, circ in mk["circuits"].items():
        circ = {int(k): v for k, v in circ.items()}
        M = masks_of(circ)
        ld, per = lds(masked(M))
        res[name] = dict(ld=ld, ld_per_batch=per, n_pairs=int(M.get_num_latents()),
                         **_ratios(ld, ld_e, ld_s, ld_m))
    label = ("TEST FIXTURE [tiny random Gemma-2]" if a.fixture else
             "SMOKE [ours: %s, %s]" % (a.hf_model, a.dtype) if (a.hf_model or a.dtype != "float32") else
             "theirs: %s fp32" % a.model)
    meta = dict(label=label, sae_ids=ids, layers=layers, hook_names=hooks, eval_batches=idx, means=means_src,
                means_file=str(a.caples_means), means_label=d.get("meta", {}).get("label"), n_batches=nb, seq_len=T,
                use_mean_error=True, mean_mask=True, t_total_s=round(time.time() - t0, 1), torch=torch.__version__)
    out = dict(meta=meta, refs=dict(ld_model=ld_m, ld_spliced=ld_s, ld_empty=ld_e, per_batch=dict(
        model=per_m, spliced=per_s, empty=per_e)), circuits=res)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(dict(out=a.out, label=label, refs={k: v for k, v in out["refs"].items() if k != "per_batch"},
                          circuits={n: dict(ld=r["ld"], f_model=r["f_model"], sfc_model=r["sfc_model"])
                                    for n, r in res.items()}), indent=1))
    return 0


# ======================================================================================================= CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", choices=("ours", "theirs", "all"))
    # ours / all
    p.add_argument("--family", default="gemma-2-2b-gsres-gap6")
    p.add_argument("--task", default="rc")
    p.add_argument("--modes", default="native,caples")
    p.add_argument("--means", default="per_position", help="native mode's A / C means (caples: per_position only)")
    p.add_argument("--caples-dirs", default="")
    p.add_argument("--wcm-circuits", default="")
    p.add_argument("--n-held", type=int, default=400)
    p.add_argument("--no-controls", action="store_true", help="skip the keep-all and empty control circuits")
    p.add_argument("--no-errclean", action="store_true")
    p.add_argument("--no-theirs", action="store_true")
    p.add_argument("--caples-means", default=None)
    p.add_argument("--caples-source", default="theirs", choices=("theirs", "exact"))
    p.add_argument("--out", default=str(OUT_DEFAULT))
    # theirs (caples_means.py's loader arguments)
    p.add_argument("--masks", default=None)
    p.add_argument("--source", default="theirs", choices=("theirs", "exact"))
    p.add_argument("--recompute-means", action="store_true")
    p.add_argument("--caples-src", default=str(CAPLES_SRC))
    p.add_argument("--example-length", type=int, default=7)
    p.add_argument("--N", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--sae-gap", type=int, default=6)
    p.add_argument("--l0-target", type=int, default=100)
    p.add_argument("--model", default="google/gemma-2-2b")
    p.add_argument("--hf-model", default=None)
    p.add_argument("--hf-cache", default=None)
    p.add_argument("--fixture", default=None)
    p.add_argument("--dtype", default=None, help="ours: bank / model dtype (default float32); theirs: float32 | "
                                                 "bfloat16")
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    if a.cmd == "theirs":
        a.dtype = a.dtype or "float32"
        a.device = a.device or "cpu"
        a.task = a.task if "/" in a.task else "sva/%s_train" % a.task
        a.mean_batches = 40                             # caples_means.load_real reads no other of its args
        if not a.masks or not a.caples_means:
            p.error("theirs needs --masks and --caples-means")
    elif "caples" in a.modes and not a.caples_means:
        p.error("--modes caples needs --caples-means")
    return a


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.cmd == "theirs":
        return theirs_main(a)
    return ours_main(a)


if __name__ == "__main__":
    sys.exit(main())

"""Arena A (2026-11-05 night): why is necessity (phi_sup_blind_tk) low on gpt2-small-tcz? Forward passes only.

For every fitted circuit in DIRS, on its held-out strongest contexts (eval_probe(rec, "strong"), explicit split: the
scorer's slice and anchors), the target's post-activation read (relu(w.x + b) = the post-"Top-K" read at k = d_sae, the
scorer's _tk read) under per-site edits at every real upstream site (the scorer's upstream_sites; NullSAE slots skipped).
At a site the stream is split into  members + non-members + bias + error  and each part is set independently:
  members      live | zero | mean (the scorer's A means: collect_site_anchors on the 48 train rows) | alpha (alpha x live)
  non-members  live | zero | mean | clean (the clean run's values at that position, i.e. frozen)
  error        live | zero | mean (per-site mean over train-row positions, clean run) | clean (frozen)
  z sites: the stream is the c_proj input and error = x - decode(code), the Kissane SAE error.
  tc sites: the stream written is the MLP OUTPUT, error = mlp(x_in) - decode(code) (the transcoder error), so an error
           edit replaces the MLP's own residual; the hook returns y with mlp(x_in) + (y - x_in) = the new output.
Conditions (read = mean post-activation at the anchors over the held-out rows; the scorer's a_pos / e0 / raw reads):
  clean                    nothing edited (= a_pos)
  rm_<E> / full_<E> / e0_<E> / co_<E>   for error mode E in live, zero, mean: members zero / live / zero / alpha;
                           non-members live / live / zero / zero.  rm_live = the scorer's necessity read (b),
                           e0_live = its e0, co_live = its free0 circuit-only read. nec_E = (full_E - rm_E) / (full_E - e0_E),
                           Z_E = (co_E - e0_E) / (full_E - e0_E): necessity and Z when every upstream error is E
                           (c) = E zero, (d) = E mean.
  rm_nmmean   (e)          members zero, non-members mean, error live;  eM_dense: everything mean (scorer's A empty)
  rm_memmean               members at their mean, rest live (survival under mean fill vs rm_live = zero fill)
  rm_zmem / rm_tcmem       only the z (z + z2) / only the tc members zeroed, rest live
  f_zfrozen   (f)          members zero; at z sites non-members + error frozen at clean; tc sites live
  f_tcfrozen               the same with the tc sites frozen and the z sites live
  g_frozen                 members zero, every site's non-members + error frozen at clean: the members' summed
                           direct (non-recomputed) effect; rm_live - g_frozen = what live recomputation re-supplies
  train: clean_tr, rm_tr, e0_tr on the 48 train rows (necessity where the fit was trained)
Per site: members, L0 (active latents per position, held-out rows, clean), members active at the anchors, and the
members' share of the site's clean decoded norm at the anchors.

  DIRS="label=<OUT with main/circuits>:<records dir>,..." OUT=<results dir> STORES=<datapass> HF_HUB_OFFLINE=1 \
      PYTHONPATH=src ./.venv/bin/python experiments/107-true-baselines/arena_a/diag_necessity_tcz.py
"""
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STORES = Path(os.environ.get("STORES", str(HERE / "out" / "pilot" / "datapass")))
OUT = Path(os.environ.get("OUT", str(HERE / "out" / "panel" / "diag_nec")))
os.environ.update(STORES=str(STORES), OUT=str(OUT), MODE="main", HF_HUB_OFFLINE="1")
os.environ.setdefault("DATA", str(ROOT / "data_public" / "gpt2-small"))
os.environ.setdefault("WCM_VERSION", "v4")
sys.path.insert(0, str(HERE))
import driver_tcz  # noqa: E402  (registers the family, imports 062's driver, float32 without TF32)
import torch  # noqa: E402

import wcm_tcz as WT  # noqa: E402

driver = driver_tcz.driver
A = HERE / "out"
DEFAULT_DIRS = ("pilot_l1e-3=%s:%s,pilot_l4e-3=%s:%s,panel_l1e-3=%s:%s" % (
    A / "pilot" / "wcm_l1e-3", A / "pilot" / "rec" / "ctx", A / "pilot" / "wcm_l4e-3", A / "pilot" / "rec" / "ctx",
    A / "panel" / "wcm_l1e-3", A / "panel" / "rec" / "ctx"))
BATCH = int(os.environ.get("DIAG_BATCH", 8))


def mlp_manual(block_mlp, x):
    """GPT-2 MLP output without triggering the module's forward hook (eval mode: dropout is identity)."""
    return block_mlp.c_proj(block_mlp.act(block_mlp.c_fc(x.contiguous())))


class Diag:
    """One forward's per-site edit spec (see the module docstring)."""

    def __init__(self, ctx, mem="live", nm="live", err="live", freeze=(), only_kinds=None, capture=False,
                 err_acc=None, freeze_nm=(), freeze_err=()):
        self.c, self.mem, self.nm, self.err = ctx, mem, nm, err
        # kinds whose non-members + error (freeze), non-members only (freeze_nm), error only (freeze_err) are frozen
        # at clean; members by mem
        self.freeze = set(freeze) | (set(freeze_nm) & set(freeze_err))
        self.freeze_nm, self.freeze_err = set(freeze_nm), set(freeze_err)
        self.only_kinds = only_kinds            # member edit only at sites of these kinds (others: members live)
        self.capture, self.err_acc = capture, err_acc
        self.read = None

    def __call__(self, model):
        from model.hooks import multi_patch
        return multi_patch(model, self.transform)

    def transform(self, l, k, x):
        c = self.c
        if (l, k) == c["site"]:
            self.read = torch.relu(x @ c["w"].to(x.dtype) + c["b"].to(x.dtype))
            return x
        if (l, k) not in c["ups"]:
            return x
        if self.capture:
            c["xclean"][(l, k)] = x.detach().clone()
        bank = c["bank"]
        lat, _ = bank.encode(x, k, l)
        lat = lat.to(x.dtype)
        recon = bank.decode(lat, k, l)
        mlp_out = mlp_manual(c["mlps"][l], x) if k == "tc" else None
        err = (mlp_out if k == "tc" else x) - recon
        if self.err_acc is not None:
            s = self.err_acc.setdefault((l, k), [torch.zeros(x.shape[-1], device=x.device, dtype=torch.float64), 0])
            s[0] += err.double().sum((0, 1))
            s[1] += err.shape[0] * err.shape[1]
        if self.capture or (self.mem == "live" and self.nm == "live" and self.err == "live" and not self.freeze
                            and not self.freeze_nm and not self.freeze_err):
            return x
        kk = "z" if k == "z2" else k
        frozen = kk in self.freeze
        fz_nm, fz_err = frozen or kk in self.freeze_nm, frozen or kk in self.freeze_err
        if fz_nm or fz_err or self.nm == "clean" or self.err == "clean":
            xc = c["xclean"][(l, k)]
            lat_c, _ = bank.encode(xc, k, l)
            lat_c = lat_c.to(x.dtype)
            err_c = (mlp_manual(c["mlps"][l], xc) if k == "tc" else xc) - bank.decode(lat_c, k, l)
        nm = "clean" if fz_nm else self.nm
        em = "clean" if fz_err else self.err
        if nm == "live":
            new = lat.clone()
        elif nm == "zero":
            new = torch.zeros_like(lat)
        elif nm == "mean":
            new = c["means"][(l, k)].to(device=x.device, dtype=x.dtype).view(1, 1, -1).expand_as(lat).clone()
        else:
            new = lat_c.clone()
        m = c["members"].get((l, k))
        edit_members = m is not None and (self.only_kinds is None or k in self.only_kinds or
                                          (k == "z2" and "z" in self.only_kinds))
        if m is not None:
            if not edit_members:
                new[..., m] = lat[..., m]
            elif self.mem == "live":
                new[..., m] = lat[..., m]
            elif self.mem == "zero":
                new[..., m] = 0.0
            elif self.mem == "mean":
                new[..., m] = c["means"][(l, k)].to(device=x.device, dtype=x.dtype)[m]
            elif self.mem == "alpha":
                new[..., m] = lat[..., m] * c["alpha"][(l, k)].to(x.dtype)
            else:
                raise ValueError(self.mem)
        if em == "live":
            e = err
        elif em == "zero":
            e = torch.zeros_like(err)
        elif em == "mean":
            e = c["err_means"][(l, k)].to(device=x.device, dtype=x.dtype).view(1, 1, -1)
        else:
            e = err_c
        out = bank.decode(new, k, l) + e
        if k == "tc":                            # the hook adds (y - x_in) to mlp(x_in): y = out - mlp(x_in) + x_in
            return out - mlp_out + x
        return out


def run(inf, ctx, toks, anchors, specs):
    """{name: per-row reads [N]} for spec factories {name: lambda: Diag}; the clean capture runs first per batch."""
    reads = {n: [] for n in specs}
    inf.disable_compile()
    try:
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            for s0 in range(0, int(toks.shape[0]), BATCH):
                tk = toks[s0:s0 + BATCH]
                anc = anchors[s0:s0 + BATCH]
                ctx["xclean"] = {}
                for name, mk in specs.items():
                    p = mk()
                    inf.forward(tk, patcher=p, grad_enabled=False, return_activations=False, tokenize_final=False)
                    rr = torch.arange(tk.shape[0], device=p.read.device)
                    reads[name].append(p.read[rr, anc.to(p.read.device)].float().cpu())
                    if name == "clean" and "stats" in ctx:
                        site_stats(ctx, anc)
    finally:
        inf.enable_compile()
    ctx["xclean"] = {}
    return {n: torch.cat(v) for n, v in reads.items()}


def site_stats(ctx, anc):
    """accumulate per-site L0 (all positions), members active at the anchors, members' share of the decoded norm at
    the anchors, from the batch's clean capture"""
    bank = ctx["bank"]
    for (l, k), xc in ctx["xclean"].items():
        lat, _ = bank.encode(xc, k, l)
        st = ctx["stats"].setdefault("%d.%s" % (l, k), dict(l0_sum=0.0, n_pos=0, mem_active_anchor=0.0,
                                                           l0_anchor=0.0, mem_norm_share=0.0, n_anchor=0))
        act = lat > 0
        tgt = (mlp_manual(ctx["mlps"][l], xc) if k == "tc" else xc).float()   # what the dictionary reconstructs
        res_ = tgt - bank.decode(lat.to(xc.dtype), k, l).float()
        st["fvu_num"] = st.get("fvu_num", 0.0) + float((res_ ** 2).sum())
        st["fvu_den"] = st.get("fvu_den", 0.0) + float(((tgt - tgt.mean((0, 1), keepdim=True)) ** 2).sum())
        st["l0_sum"] += float(act.sum())
        st["n_pos"] += int(act.shape[0] * act.shape[1])
        rr = torch.arange(lat.shape[0], device=lat.device)
        la = lat[rr, anc.to(lat.device)]                                    # [B, d_sae]
        st["l0_anchor"] += float((la > 0).sum())
        m = ctx["members"].get((l, k))
        st.setdefault("mem_active_sum", 0.0)
        if m is not None:
            st["mem_active_sum"] += float(act[..., m].sum())
            st["mem_active_anchor"] += float((la[:, m] > 0).sum())
            mm = torch.zeros_like(la)
            mm[:, m] = la[:, m]
            num = bank.decode(mm, k, l, add_bias=False).norm(dim=-1)
            den = bank.decode(la, k, l, add_bias=False).norm(dim=-1).clamp(min=1e-9)
            st["mem_norm_share"] += float((num / den).sum())
        st["n_anchor"] += int(la.shape[0])


def specs_for(ctx):
    D = lambda **kw: (lambda: Diag(ctx, **kw))  # noqa: E731
    sp = dict(clean=D(capture=True))
    for e in ("live", "zero", "mean"):
        sp["full_" + e] = D(err=e)
        sp["rm_" + e] = D(mem="zero", err=e)
        sp["e0_" + e] = D(mem="zero", nm="zero", err=e)
        sp["co_" + e] = D(mem="alpha", nm="zero", err=e)
    sp.update(rm_nmmean=D(mem="zero", nm="mean"), eM_dense=D(mem="mean", nm="mean"),
              coM_dense=D(mem="alpha", nm="mean"), rm_memmean=D(mem="mean"),
              rm_zmem=D(mem="zero", only_kinds={"z"}), rm_tcmem=D(mem="zero", only_kinds={"tc"}),
              f_zfrozen=D(mem="zero", freeze={"z"}), f_tcfrozen=D(mem="zero", freeze={"tc"}),
              g_frozen=D(mem="zero", freeze={"z", "tc"}),
              f_tc_nm=D(mem="zero", freeze_nm={"tc"}), f_tc_err=D(mem="zero", freeze_err={"tc"}),
              f_z_nm=D(mem="zero", freeze_nm={"z"}), f_z_err=D(mem="zero", freeze_err={"z"}),
              g_nm=D(mem="zero", freeze_nm={"z", "tc"}), g_err=D(mem="zero", freeze_err={"z", "tc"}))
    del sp["full_live"]                    # = clean
    return sp


def frac_(a, b, c):
    return (a - b) / (c - b) if abs(c - b) > 1e-9 else None


def main():
    from circuit.protocol_contexts import eval_probe
    from eval.ablation_faithfulness import upstream_sites
    from eval.floors import collect_site_anchors
    from eval.wcm_eval import circuit_alphas, circuit_key, eval_split
    R = driver.Runner()
    inf, bank, dev = R.G["inference"], R.G["bank"], R.G["device"]
    mlps = [b.mlp for b in inf.model.hf.transformer.h]
    OUT.mkdir(parents=True, exist_ok=True)
    # v2 (2026-11-05 02:45: split freezes + per-site FVU): its own file; results.jsonl = the first pass (02:31)
    out_path = OUT / os.environ.get("RESULTS", "results_v2.jsonl")
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                r = json.loads(line)
                done.add((r["label"], r["key"]))
            except ValueError:
                continue
    fh = open(out_path, "a")
    real = set(WT.real_sites())
    # EXTRA_DIRS (default: the pilot + panel fits) are appended to DIRS, so the night chain's last diag step (DIRS =
    # the arms) also writes the v2 rows of the base circuits; EXTRA_DIRS= (empty) turns it off
    items = os.environ.get("DIRS", DEFAULT_DIRS).split(",")
    items += [x for x in os.environ.get("EXTRA_DIRS", DEFAULT_DIRS + ",panel_l1e-3_rerun=%s:%s" % (
        A / "panel" / "wcm_l1e-3_rerun", A / "panel" / "rec" / "ctx")).split(",") if x and x not in items]
    for item in items:
        label, rest = item.split("=", 1)
        cdir, rdir = rest.rsplit(":", 1)
        cdir, rdir = Path(cdir), Path(rdir)
        evals = {}
        for p in sorted((cdir / "main").glob("eval.shard*.jsonl")):
            for line in open(p):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("held") == "strong":
                    evals[r["seed"]] = r
        for cp in sorted((cdir / "main" / "circuits").glob("*.pt")):
            t0 = time.time()
            c = torch.load(cp, weights_only=False)
            if c is None:
                print("%s %s: rejected circuit" % (label, cp.stem), flush=True)
                continue
            key = circuit_key(c)
            if (label, key) in done or (os.environ.get("KEYS") and key not in os.environ["KEYS"].split(",")):
                continue
            l, k, i = int(key.split(".")[0]), key.split(".")[1], int(key.split(".")[2])
            rec = torch.load(rdir / ("%s.pt" % key), weights_only=False)[key]
            pd_ = eval_probe(rec, "strong", dev)
            pt, pa = pd_.pos_tokens[:64], pd_.pos_argmax[:64]
            n_tr = eval_split(int(pt.shape[0]), pd_.metadata, "explicit")
            alphas = circuit_alphas(c)
            ups = {s for s in upstream_sites(bank, l, k) if s in real and not getattr(bank.saes[s[1]][s[0]], "null", False)}
            members = {s: torch.tensor(sorted(d), device=dev, dtype=torch.long) for s, d in alphas.items() if d}
            alpha = {s: torch.tensor([alphas[s][int(j)] for j in members[s].tolist()], device=dev) for s in members}
            w, b = bank.preact_vector(k, l, i)
            means, _ = collect_site_anchors(inf, bank, pt[:n_tr], ups, pa[:n_tr], pin_position_specific=False)
            ctx = dict(site=(l, k), w=w, b=b, ups=ups, bank=bank, mlps=mlps, members=members, alpha=alpha,
                       means=means, xclean={})
            acc = {}
            inf.disable_compile()
            try:
                with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
                    for s0 in range(0, n_tr, BATCH):
                        inf.forward(pt[s0:min(s0 + BATCH, n_tr)], patcher=Diag(ctx, err_acc=acc), grad_enabled=False,
                                    return_activations=False, tokenize_final=False)
            finally:
                inf.enable_compile()
            ctx["err_means"] = {s: (v[0] / v[1]).float() for s, v in acc.items()}
            ctx["stats"] = {}
            ho = run(inf, ctx, pt[n_tr:], pa[n_tr:], specs_for(ctx))
            stats = ctx.pop("stats")
            D = lambda **kw: (lambda: Diag(ctx, **kw))  # noqa: E731
            tr = run(inf, ctx, pt[:n_tr], pa[:n_tr], dict(clean=D(capture=True), rm=D(mem="zero"),
                                                         e0=D(mem="zero", nm="zero")))
            m = {n: float(v.mean()) for n, v in ho.items()}
            m["full_live"] = m["clean"]
            res = dict(label=label, key=key, layer=l, kind=k, n=sum(len(d) for d in alphas.values()),
                       n_ho=int(pt.shape[0] - n_tr), n_up=len(ups), reads=m)
            res["nec"] = {e: frac_(m["rm_" + e], m["full_" + e], m["e0_" + e]) for e in ("live", "zero", "mean")}
            res["Z"] = {e: frac_(m["co_" + e], m["e0_" + e], m["full_" + e]) for e in ("live", "zero", "mean")}
            a0, e0 = m["clean"], m["e0_live"]
            res["nec_like"] = {n: frac_(m[n], a0, e0) for n in ("rm_live", "rm_nmmean", "rm_memmean", "rm_zmem",
                                                                 "rm_tcmem", "f_zfrozen", "f_tcfrozen", "g_frozen",
                                                                 "f_tc_nm", "f_tc_err", "f_z_nm", "f_z_err", "g_nm",
                                                                 "g_err", "eM_dense")}
            res["survive"] = {n: (m[n] / a0 if a0 > 0 else None) for n in (
                "rm_live", "rm_memmean", "rm_nmmean", "eM_dense", "e0_live", "f_zfrozen", "f_tcfrozen", "g_frozen",
                "rm_zero", "e0_zero", "full_zero", "rm_mean", "full_mean")}
            mt = {n: float(v.mean()) for n, v in tr.items()}
            res["train"] = dict(mt, nec=frac_(mt["rm"], mt["clean"], mt["e0"]))
            res["nec_A_dense"] = frac_(m["rm_memmean"], a0, m["eM_dense"])
            res["sites"] = {s: dict(l0=round(v["l0_sum"] / max(v["n_pos"], 1), 2),
                                    l0_anchor=round(v["l0_anchor"] / max(v["n_anchor"], 1), 2),
                                    n_mem=len(alphas.get((int(s.split(".")[0]), s.split(".")[1]), {})),
                                    mem_active=round(v["mem_active_sum"] / max(v["n_pos"], 1), 2),
                                    fvu=round(v["fvu_num"] / max(v["fvu_den"], 1e-12), 3),
                                    mem_active_anchor=round(v["mem_active_anchor"] / max(v["n_anchor"], 1), 2),
                                    mem_norm_share=round(v["mem_norm_share"] / max(v["n_anchor"], 1), 3))
                            for s, v in sorted(stats.items(), key=lambda kv: (int(kv[0].split(".")[0]),
                                                                              WT.KINDS.index(kv[0].split(".")[1])))}
            ev = evals.get(key) or {}
            res["scorer"] = {f: ev.get(f) for f in ("a_pos_tk", "e0_tk", "phi_sup_blind_raw_tk", "phi_sup_blind_tk",
                                                    "free0_tk", "freeM_topk_tk", "freeN_topk_tk", "free0_raw_tk",
                                                    "eM_dense_tk", "freeM_dense_tk")}
            res["check"] = dict(
                a_pos=None if ev.get("a_pos_tk") is None else m["clean"] - ev["a_pos_tk"],
                e0=None if ev.get("e0_tk") is None else m["e0_live"] - ev["e0_tk"],
                rm=None if ev.get("phi_sup_blind_raw_tk") is None else m["rm_live"] - ev["phi_sup_blind_raw_tk"],
                co=None if ev.get("free0_raw_tk") is None else m["co_live"] - ev["free0_raw_tk"])
            res["secs"] = round(time.time() - t0, 1)
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            fmt = lambda v: "-" if v is None else "%.2f" % v  # noqa: E731
            print("%-12s %-12s n %5d | a %.3f rm %.3f e0 %.3f | nec live %s zero %s mean %s | Z live %s zero %s | "
                  "nmmean %s g %s fz %s ftc %s | tr nec %s | chk rm %.1e | %.0fs" % (
                      label, key, res["n"], a0, m["rm_live"], e0, *(fmt(res["nec"][e]) for e in ("live", "zero", "mean")),
                      fmt(res["Z"]["live"]), fmt(res["Z"]["zero"]), fmt(res["nec_like"]["rm_nmmean"]),
                      fmt(res["nec_like"]["g_frozen"]), fmt(res["nec_like"]["f_zfrozen"]),
                      fmt(res["nec_like"]["f_tcfrozen"]), fmt(res["train"]["nec"]),
                      res["check"]["rm"] if res["check"]["rm"] is not None else float("nan"), res["secs"]), flush=True)
            del ctx
            torch.cuda.empty_cache()
    print("diag_necessity: done", flush=True)


if __name__ == "__main__":
    main()

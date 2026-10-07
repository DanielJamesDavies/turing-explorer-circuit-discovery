"""Prediction circuits CLI (DAN-134): which fired circuits drive TuringLLM's next-token prediction?

  PYTHONPATH=src python experiments/064-prediction-circuits/circuits_cli.py            # interactive REPL
  PYTHONPATH=src python experiments/064-prediction-circuits/circuits_cli.py run "The capital of France is"
  PYTHONPATH=src python experiments/064-prediction-circuits/circuits_cli.py target 10.resid.32101
  PYTHONPATH=src python experiments/064-prediction-circuits/circuits_cli.py top-predictors --layer 12 --pass-only

Commands (the same in the REPL and as one-shot arguments):
  run "<prompt>" [--gen N] [--sample] [--top K] [--sort dla|score] [--from I]
        per position: token, the model's top-5 next tokens, and the fired circuits ranked by DLA to the top-1
  pos <index> [--top K] [--latents K]
        one position of the last run: all fired circuits (DLA to top-1 and to the actual next token, explorer
        score, member support) and the top active latents of ANY kind by DLA (circuit targets marked *)
  target <key>   research key 5.mlp.2277 (0-based) or label L6.mlp.2278 / "L6 · MLP · 2278" (1-based)
        top promoted / suppressed tokens, logit_ctx next tokens, circuit pass + metrics, top-context snippets
  top-predictors [--layer L] [--kind K] [--pass-only] [--tier pred|focused|writer|all] [--sort boost|z1|ctx_z] [-n N]
        the most prediction-like targets (layer is 1-based, as in the labels)
Labels are 1-based ("L6 · MLP · 2278"); the 0-based research key is shown alongside.

DLA = first-order effect on log p(token) of the target's write (activation x unit decoder direction) through the
exact final-RMSNorm Jacobian at that position and the unembedding, centred by the model's own distribution.
It is the DIRECT path only (see README: it tracks ablations with r ~ 0.7 in the last four layers, and
underestimates early-layer effects).
Needs results/ from target_effects.py and summarise.py (for top-predictors / tiers).
"""
from __future__ import annotations

import argparse
import cmd
import shlex
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import common as C

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


class State:
    """Lazily loaded resources; model + SAEs only when `run` needs them."""

    def __init__(self):
        self._model = self._bank = self._tok = self._circ = self._mi = self._lc = self._toks = None
        self._D = self._eff = None
        self.last = None

    def _t(self, what, fn):
        t0 = time.time()
        print(f"  loading {what} ...", end="", flush=True)
        out = fn()
        print(f" {time.time() - t0:.0f}s", flush=True)
        return out

    @property
    def tok(self):
        if self._tok is None:
            self._tok = C.load_tokenizer()
            self.dec = C.Dec(self._tok)
        return self._tok

    @property
    def model(self):
        if self._model is None:
            self._model = self._t("model", C.load_model)
            self.W_U, self.g, self.eps = C.unembed(self._model)
        return self._model

    @property
    def bank(self):
        if self._bank is None:
            self._bank = self._t("36 SAEs", C.load_bank)
        return self._bank

    @property
    def circ(self):
        if self._circ is None:
            self.con = C.connect()
            self._circ = self._t("circuits", lambda: C.load_circuits(self.con))
            self.tt = C.TargetTable(self._circ)
        return self._circ

    @property
    def mi(self):
        if self._mi is None:
            circ = self.circ  # opens self.con
            self._mi = self._t("members (7.5M rows)", lambda: C.MemberIndex(self.con, circ))
        return self._mi

    @property
    def D(self):
        if self._D is None:
            self._D = self._t("target decoder directions",
                              lambda: C.decoder_dirs(self.bank, self.circ.seed_gid.to_numpy()))
        return self._D

    @property
    def eff(self):
        """Per-target effects + tiers + top promoted/suppressed tokens (results/ of target_effects/summarise)."""
        if self._eff is None:
            e = pd.read_parquet(C.RESULTS / "target_effects.parquet")
            tiers = C.RESULTS / "target_tiers.parquet"
            if tiers.exists():
                e = e.merge(pd.read_parquet(tiers), on="gid")
            tk = np.load(C.RESULTS / "target_topk.npz")
            self.topk = {k: tk[k] for k in tk.files}
            self.topk_row = {int(gg): i for i, gg in enumerate(tk["gid"])}
            self._eff = e.set_index("gid", drop=False)
        return self._eff

    @property
    def lc(self):
        if self._lc is None:
            self._lc = self._t("logit_ctx", C.load_logit_ctx)
        return self._lc

    @property
    def toks(self):
        if self._toks is None:
            self._toks = np.load(C.BUNDLE / "tokens" / "tokens.npy", mmap_mode="r")
        return self._toks

    # ------------------------------------------------------------------ helpers
    def promotes(self, gid, n=4, suppress=False):
        self.eff
        r = self.topk_row.get(int(gid))
        if r is None:
            return ""
        ids = self.topk["bot_ids" if suppress else "top_ids"][r][:n]
        return " ".join(repr(self.dec(int(x))) for x in ids)

    def direct_logits(self, gid):
        """Centred direct logit vector (per unit activation, typical norm scale) for any latent."""
        self.model
        d = C.decoder_dirs(self.bank, [gid])[0]
        L = (d * self.g) @ self.W_U[:C.N_REAL_VOCAB].T / C.load_rms_typ()
        return L - L.mean()


S = State()


# ---------------------------------------------------------------------- run / pos
def do_run(a):
    tok, model = S.tok, S.model
    S.bank, S.circ, S.mi, S.D, S.eff
    ids = [C.BOS] + tok.encode(a.prompt)
    n_prompt = len(ids)
    if a.gen:
        rng = torch.Generator().manual_seed(a.seed)
        x = torch.tensor([ids])
        with torch.no_grad():
            for _ in range(a.gen):
                lg, _ = model(x)
                p = F.softmax(lg[0, -1].float(), -1)
                if a.sample:
                    tp, ti = p.topk(5)
                    nxt = int(ti[torch.multinomial(tp, 1, generator=rng)])
                else:
                    nxt = int(p.argmax())
                x = torch.cat([x, torch.tensor([[nxt]])], 1)
        ids = x[0].tolist()
    if len(ids) > C.SAE_WINDOW:
        print(f"  note: {len(ids)} tokens; the SAEs / bundle statistics cover positions 0..{C.SAE_WINDOW - 1}")
    logits, acts = C.forward_capture(model, ids)
    vals, idx = C.encode_all(S.bank, acts)
    probs = F.softmax(logits, -1)
    xf = acts[11, 2]
    S.last = {"ids": ids, "n_prompt": n_prompt, "logits": logits, "probs": probs, "vals": vals, "idx": idx,
              "xf": xf, "pos": {}}
    print(f"\n{S.dec.text(ids[1:])!r}\n")
    for t in range(a.from_, len(ids)):
        info = analyse_pos(t)
        if info is None:
            continue
        print_pos_header(t, info)
        fired = info["fired"].sort_values("dla_top1" if a.sort == "dla" else "score", ascending=False)
        for _, r in fired.head(a.top).iterrows():
            print_fired(r)
        if len(fired) == 0:
            print("      (no circuit fired)")
    print("\n  `pos <i>` drills into one position.")


def analyse_pos(t):
    L = S.last
    if t in L["pos"]:
        return L["pos"][t]
    ids = L["ids"]
    if ids[t] == C.BOS:
        return None
    p = L["probs"][t]
    top5p, top5 = p.topk(5)
    top1 = int(top5[0])
    nxt = ids[t + 1] if t + 1 < len(ids) else -1
    gids, avals = C.active_at(L["vals"], L["idx"], t)
    rows, tg, ta, sup, score = C.fired_circuits(S.tt, S.mi, gids, avals)
    dirs = S.D[torch.as_tensor(rows)] if len(rows) else torch.zeros(0, 1024)
    xf = L["xf"][t]
    d1 = C.logprob_dla(xf, p, S.W_U, S.g, S.eps, top1, dirs, ta).numpy() if len(rows) else np.zeros(0)
    dn = (C.logprob_dla(xf, p, S.W_U, S.g, S.eps, nxt, dirs, ta).numpy() if len(rows) and nxt >= 0
          else np.full(len(rows), np.nan))
    f = pd.DataFrame({"row": rows, "gid": tg, "act": ta, "support": sup, "score": score, "dla_top1": d1,
                      "dla_next": dn})
    f["pass"] = S.circ["pass"].to_numpy()[rows] if len(rows) else []
    info = {"top5": top5.tolist(), "top5p": top5p.tolist(), "top1": top1, "next": nxt, "fired": f,
            "gids": gids, "acts": avals}
    L["pos"][t] = info
    return info


def print_pos_header(t, info):
    L = S.last
    tag = "" if t < L["n_prompt"] else " (gen)"
    nx = "  ".join(f"{S.dec(i)!r} {pp:.2f}" for i, pp in zip(info["top5"], info["top5p"]))
    print(f"[{t:>3}] {S.dec(L['ids'][t])!r}{tag}  ->  {nx}   fired {len(info['fired'])}")


def print_fired(r, show_next=False):
    extra = f" {r.dla_next:+7.3f}" if show_next else ""
    print(f"      {r.dla_top1:+7.3f}{extra} {r.act:6.2f} {r.score:6.3f} {'pass' if r['pass'] else '  - '}  "
          f"{C.full_label(int(r.gid)):<38} {S.promotes(r.gid)}")


def do_pos(a):
    if S.last is None:
        print("  no run yet: use `run \"<prompt>\"` first")
        return
    t = a.index
    if not (0 <= t < len(S.last["ids"])):
        print(f"  position out of range 0..{len(S.last['ids']) - 1}")
        return
    info = analyse_pos(t)
    if info is None:
        print("  BOS position: no circuits reported")
        return
    ids = S.last["ids"]
    print(f"\ncontext: {S.dec.text(ids[max(1, t - 12):t])!r} [[{S.dec(ids[t])}]]")
    print_pos_header(t, info)
    if info["next"] >= 0:
        lp = torch.log(S.last["probs"][t, info["next"]])
        print(f"      actual next token {S.dec(info['next'])!r}: log p = {float(lp):.2f}")
    print(f"\n  fired circuits by DLA to top-1 {S.dec(info['top1'])!r}")
    print("      DLA-top1 DLA-next    act  score  pass  target                                 promotes")
    for _, r in info["fired"].sort_values("dla_top1", ascending=False).head(a.top).iterrows():
        print_fired(r, show_next=True)
    print("\n  same circuits by explorer score (what the explorer shows first)")
    for _, r in info["fired"].sort_values("score", ascending=False).head(min(a.top, 8)).iterrows():
        print_fired(r, show_next=True)
    # all active latents, not only circuit targets
    L = S.last
    p = L["probs"][t]
    gids, acts = info["gids"], info["acts"]
    dirs = C.decoder_dirs(S.bank, gids)
    dla = C.logprob_dla(L["xf"][t], p, S.W_U, S.g, S.eps, info["top1"], dirs, acts).numpy()
    o = np.argsort(-dla)[:a.latents]
    print(f"\n  top active latents of ANY kind by DLA to top-1 (* = has a circuit; {len(gids)} active)")
    for j in o:
        mark = "*" if S.tt.is_target[gids[j]] else " "
        print(f"      {dla[j]:+7.3f} {acts[j]:6.2f} {mark} {C.full_label(int(gids[j]))}")


# ---------------------------------------------------------------------- target
def do_target(a):
    try:
        gid = C.parse_target(a.key)
    except ValueError as e:
        print(f"  {e}")
        return
    S.tok, S.circ, S.eff
    print(f"\n{C.full_label(gid)}")
    row = S.tt.row_of.get(gid)
    if gid in S.eff.index:
        e = S.eff.loc[gid]
        print(f"  top promoted : {S.promotes(gid, 10)}")
        vals = S.topk["top_vals"][S.topk_row[gid]][:10]
        print(f"                 {' '.join(f'{v:.3f}' for v in vals)}  (centred logit per unit activation)")
        print(f"  top suppressed: {S.promotes(gid, 10, suppress=True)}")
        tiers = [k for k in ("writer", "pred_like", "focused") if k in e and bool(e[k])]
        print(f"  output-ness  : boost@peak {e['boost_peak']:.2f} nats (peak {e['peak']:.1f}), "
              f"top-1 z {e['z1']:.1f}, kurtosis {e['kurt']:.2f}, unembed gain {e['gain']:.2f}, "
              f"logit_ctx agreement z {e['ctx_z']:.2f}"
              f"  tiers: {', '.join(tiers) or 'none'}")
    else:
        L = S.direct_logits(gid)
        tv, ti = L.topk(10)
        bv, bi = (-L).topk(10)
        print("  (not a circuit target: direct effect computed live)")
        print(f"  top promoted : {' '.join(repr(S.dec(int(x))) for x in ti)}")
        print(f"  top suppressed: {' '.join(repr(S.dec(int(x))) for x in bi)}")
    lt, lp, lcnt = S.lc
    nx = C.distinct_next_tokens(lt, lp, gid, 12)
    comp, lat = divmod(gid, C.N_LAT)
    print(f"  logit_ctx next (max p, last-position firings n={int(lcnt[comp, lat])}): "
          + " ".join(f"{S.dec(t)!r} {p:.2f}" for t, p in nx))
    if row is None:
        print("  no circuit for this latent in the 062 run")
        return
    c = S.circ.iloc[row]
    print(f"  circuit {c.key}: pass {bool(c['pass'])}, nodes {c.n_nodes}, Z {c.free0_tk:.3f} A {c.freeM_topk_tk:.3f} "
          f"C {c.freeN_topk_tk:.3f}, phi_sup {c.phi_sup_blind_tk:.3f}, near-threshold {bool(c.near_threshold)}, "
          f"amplifier {bool(c.amp_any)}, target peak {c.peak:.2f}")
    rows = S.con.execute("SELECT rank, seq_id, peak, arg FROM target_ctx WHERE gid = ? AND pool = 'strong' "
                         "ORDER BY rank LIMIT ?", (gid, a.ctx)).fetchall()
    print(f"  top contexts (strong pool, peak token in [[ ]], then the next 3 tokens):")
    seen = set()
    for rank, sid, peak, arg in rows:
        if not sid:
            continue
        ids = [int(x) for x in S.toks[sid - 1]]
        w = (S.dec.text(ids[max(0, arg - 12):arg]) + " [[" + S.dec.text(ids[arg:arg + 1]) + "]]"
             + S.dec.text(ids[arg + 1:arg + 4]))
        if w in seen:
            continue
        seen.add(w)
        print(f"    {peak:6.2f}  {w}")


# ---------------------------------------------------------------------- top-predictors
def do_top(a):
    S.tok
    e = S.eff
    if a.layer is not None:
        e = e[e.layer == a.layer - 1]
    if a.kind:
        e = e[e.kind == a.kind]
    if a.pass_only:
        e = e[e["pass"]]
    tier = {"pred": "pred_like", "focused": "focused", "writer": "writer"}.get(a.tier)
    if tier:
        if tier not in e:
            print("  run summarise.py first (tiers)")
            return
        e = e[e[tier]]
    key = {"boost": "boost_peak", "z1": "z1", "ctx_z": "ctx_z"}[a.sort]
    e = e.sort_values(key, ascending=False)
    print(f"\n{len(e)} targets match; top {a.n} by {key}")
    print("  boost@pk    z1  ctx_z  pass  target                                 promotes")
    for _, r in e.head(a.n).iterrows():
        print(f"  {r.boost_peak:8.2f} {r.z1:5.1f} {r.ctx_z:6.2f}  {'pass' if r['pass'] else '  - '}  "
              f"{C.full_label(int(r.gid)):<38} {S.promotes(r.gid, 5)}")


# ---------------------------------------------------------------------- parsing
def parser():
    p = argparse.ArgumentParser(prog="circuits_cli", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd")
    r = sub.add_parser("run", help="run a prompt; fired circuits ranked by DLA to the top prediction")
    r.add_argument("prompt")
    r.add_argument("--gen", type=int, default=0, help="generate N tokens first (greedy unless --sample)")
    r.add_argument("--sample", action="store_true", help="sample from the top-5 (seeded) instead of greedy")
    r.add_argument("--seed", type=int, default=12)
    r.add_argument("--top", type=int, default=5, help="fired circuits shown per position")
    r.add_argument("--sort", choices=("dla", "score"), default="dla")
    r.add_argument("--from", dest="from_", type=int, default=0, help="first position to print")
    q = sub.add_parser("pos", help="drill into one position of the last run")
    q.add_argument("index", type=int)
    q.add_argument("--top", type=int, default=15)
    q.add_argument("--latents", type=int, default=10)
    t = sub.add_parser("target", help="one target: output tokens, logit_ctx, circuit metrics, contexts")
    t.add_argument("key")
    t.add_argument("--ctx", type=int, default=8)
    tp = sub.add_parser("top-predictors", help="most prediction-like targets")
    tp.add_argument("--layer", type=int, help="1-based layer")
    tp.add_argument("--kind", choices=C.KINDS)
    tp.add_argument("--pass-only", action="store_true")
    tp.add_argument("--tier", choices=("pred", "focused", "writer", "all"), default="pred")
    tp.add_argument("--sort", choices=("boost", "z1", "ctx_z"), default="boost")
    tp.add_argument("-n", type=int, default=25)
    return p


HANDLERS = {"run": do_run, "pos": do_pos, "target": do_target, "top-predictors": do_top}


def dispatch(argv):
    try:
        a = parser().parse_args(argv)
    except SystemExit:
        return
    if a.cmd is None:
        parser().print_help()
        return
    HANDLERS[a.cmd](a)


class Repl(cmd.Cmd):
    intro = ("Prediction circuits CLI. Commands: run \"<prompt>\" [--gen N], pos <i>, target <key>, "
             "top-predictors [...], help, quit.  `<command> -h` for options.")
    prompt = "circuits> "

    def default(self, line):
        try:
            argv = shlex.split(line, posix=True)
        except ValueError as e:
            print(f"  {e}")
            return
        if argv and argv[0] in ("quit", "exit", "q"):
            return True
        try:
            dispatch(argv)
        except Exception as e:  # keep the loaded state on errors
            print(f"  error: {type(e).__name__}: {e}")

    def do_help(self, arg):
        print(__doc__)

    def do_quit(self, arg):
        return True

    do_exit = do_quit

    def do_EOF(self, arg):
        print()
        return True

    def emptyline(self):
        pass


if __name__ == "__main__":
    if len(sys.argv) > 1:
        dispatch(sys.argv[1:])
    else:
        Repl().cmdloop()

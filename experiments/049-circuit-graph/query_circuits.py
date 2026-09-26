"""QUERY -> WHICH CIRCUITS LIT UP. Run TuringLLM on a text, encode every
(layer, kind) site with its SAE, and score every production circuit:

  seed fired     the circuit's seed latent is active anywhere (max act,
                 peak position, the token there)
  frac_any       fraction of the circuit's member nodes active ANYWHERE in
                 the text
  frac_before    fraction active at or before the seed's peak position —
                 the causally relevant one (members are upstream of the seed)
  frac_amp       amplitude-weighted version of frac_before (the tri-amp
                 gains say which members the circuit leans on)

Also the family tally (results_full/circuit_families_nohub.csv) so you
can see which organs a query recruits.

  PYTHONPATH=src python experiments/049-circuit-graph/query_circuits.py -q "The Eiffel Tower is in"
  PYTHONPATH=src python experiments/049-circuit-graph/query_circuits.py            # interactive: type queries, blank line to quit
Options: --top N (25), --min-frac F (0) filter on frac_before, --csv PATH dump all seed-fired rows,
         --families to print the family tally, TABLES/RESULTS env to point at a different set.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hardware import detect_devices, is_fast_memory, should_compile
from model.hooks import multi_patch
from model.inference import Inference
from model.tokenizer import Tokenizer
from sae.bank import SAEBank

HERE = Path(__file__).parent
T = Path(os.environ.get("TABLES", str(HERE / "tables_full")))
RES = Path(os.environ.get("RESULTS", str(HERE / "results_full")))

ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", action="append", default=[], help="query text (repeatable)")
ap.add_argument("--top", type=int, default=25)
ap.add_argument("--min-frac", type=float, default=0.0)
ap.add_argument("--csv", default=None)
ap.add_argument("--families", action="store_true")
ap.add_argument("--no-bos", action="store_true", help="do not prepend <s>")
ap.add_argument("--sort", choices=["act", "frac", "amp", "fired"], default="act",
                help="rank by seed activation, member-fire fraction, amplitude-weighted fraction, or number of members fired")
ap.add_argument("--skip-pos", type=int, default=-1,
                help="hide seeds whose peak is at position <= this (e.g. 1 hides the <s>/first-word scaffold circuits)")
ap.add_argument("--predict", type=int, default=0,
                help="show the model's top next tokens and the K circuits active AT THE LAST POSITION with their direct-logit output signatures")
ap.add_argument("--generate", type=int, default=0, help="with --predict: greedy-generate this many tokens and print them")
ap.add_argument("--inspect", type=int, default=0, help="detail the top K circuits: members by layer, fired where/how strongly, amplitude, silent members")
ap.add_argument("--inspect-members", type=int, default=30, help="max fired members listed per inspected circuit")
args = ap.parse_args()

torch.set_float32_matmul_precision("high")
devices = detect_devices(); device = devices[0]
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
tok = Tokenizer()
KINDS = list(bank.kinds)

print("loading tables ...", flush=True)
C = pd.read_parquet(T / "circuits.parquet")
M = pd.read_parquet(T / "members.parquet")[["cid", "layer", "kind", "index", "amplitude"]]
M["kind"] = M["kind"].astype(str)
fam = None
fp = RES / "circuit_families_nohub.csv"
if fp.exists():
    fam = pd.read_csv(fp).set_index("cid")["family"]
print("  %d circuits, %d member rows%s" % (len(C), len(M), "" if fam is None else ", families loaded"), flush=True)
_W_U = None


def output_signature(layer, kind, index, k=5):
    """DIRECT LOGIT ATTRIBUTION of the latent: its decoder direction (attn,
    mlp and resid all add into the residual stream) through the final
    RMSNorm scale and the unembedding; top-k promoted and top suppressed."""
    global _W_U
    if _W_U is None:
        _W_U = (inference.model.lm_head.weight.detach().float() * inference.model.transformer.norm_f.scale.detach().float()[None, :])
    sae = bank.saes[kind][layer]
    d = sae.decoder.weight[:, index].detach().float().to(_W_U.device)
    lg = _W_U @ d
    top = torch.topk(lg, k); bot = torch.topk(-lg, 2)
    return ("+ " + " ".join(repr(tok.decode([int(i)])) for i in top.indices)
            + "   - " + " ".join(repr(tok.decode([int(i)])) for i in bot.indices))


def generate(ids, n):
    """Greedy continuation of n tokens (no patcher)."""
    out = list(ids)
    inference.disable_compile()
    try:
        with torch.no_grad():
            for _ in range(n):
                _, lg, _ = inference.forward(torch.tensor([out], device=device), grad_enabled=False, return_activations=False, tokenize_final=False)
                lg = lg[0, -1] if lg.dim() == 3 else lg[0]
                out.append(int(torch.argmax(lg)))
    finally:
        inference.enable_compile()
    return out[len(ids):]


class Capture:
    """Encode at every site; keep the sparse (pos, latent, act) triples."""

    def __init__(self):
        self.rows = []

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        ta, ti = bank.encode(x, kind, layer_idx)          # [1, T, K]
        ta = ta[0].float().cpu().numpy(); ti = ti[0].cpu().numpy()
        pos = np.repeat(np.arange(ta.shape[0]), ta.shape[1])
        keep = ta.reshape(-1) > 0
        self.rows.append(pd.DataFrame({"layer": layer_idx, "kind": kind, "index": ti.reshape(-1)[keep].astype(np.int64),
                                       "pos": pos[keep], "act": ta.reshape(-1)[keep]}))
        return x


def run_query(text):
    ids = tok.encode(text)
    bos = tok.get_bos_token()
    if not args.no_bos and (not ids or ids[0] != bos):
        ids = [bos] + ids
    toks = torch.tensor([ids], device=device)
    cap = Capture()
    inference.disable_compile()
    try:
        with torch.no_grad():
            _, logits, _ = inference.forward(toks, patcher=cap, grad_enabled=False, return_activations=False, tokenize_final=False)
    finally:
        inference.enable_compile()
    act = pd.concat(cap.rows, ignore_index=True)
    words = [tok.decode([t]) for t in ids]
    last = len(ids) - 1
    if args.predict:
        lg = logits[0, -1] if logits.dim() == 3 else logits[0]
        pr = torch.softmax(lg.float(), dim=-1)
        top = torch.topk(pr, 10)
        print("\nQUERY: %r" % text)
        print("MODEL PREDICTS NEXT: " + "  ".join("%s %.2f" % (repr(tok.decode([int(i)])), float(p)) for p, i in zip(top.values, top.indices)))
        if args.generate:
            cont = generate(ids, args.generate)
            print("GREEDY CONTINUATION: %r" % tok.decode(cont))
    # per-latent summary: max act, position of max, first active position
    g = act.groupby(["layer", "kind", "index"])
    lat = pd.DataFrame({"max_act": g["act"].max(), "first_pos": g["pos"].min(),
                        "peak_pos": g.apply(lambda d: int(d.loc[d["act"].idxmax(), "pos"]))}).reset_index()
    # seeds
    S = C[["cid", "seed_layer", "seed_kind", "seed_index", "n_members"]].rename(
        columns={"seed_layer": "layer", "seed_kind": "kind", "seed_index": "index"})
    S["kind"] = S["kind"].astype(str)
    S = S.merge(lat, on=["layer", "kind", "index"], how="inner")
    if S.empty:
        print("  no circuit seed fired on this query"); return None
    # members of the fired circuits: fired anywhere / at-or-before the seed peak
    Mm = M[M["cid"].isin(S["cid"])].merge(lat[["layer", "kind", "index", "first_pos"]], on=["layer", "kind", "index"], how="left")
    Mm = Mm.merge(S[["cid", "peak_pos"]], on="cid")
    Mm["any"] = Mm["first_pos"].notna()
    Mm["before"] = Mm["any"] & (Mm["first_pos"] <= Mm["peak_pos"])
    Mm["amp"] = Mm["amplitude"].fillna(1.0).clip(lower=0)
    agg = Mm.groupby("cid").apply(lambda d: pd.Series({
        "frac_any": d["any"].mean(), "frac_before": d["before"].mean(),
        "frac_amp": float((d["amp"] * d["before"]).sum() / max(d["amp"].sum(), 1e-9)),
        "n_fired": int(d["before"].sum())}))
    S = S.merge(agg, left_on="cid", right_index=True)
    S["seed"] = S["layer"].astype(str) + "." + S["kind"] + "." + S["index"].astype(str)
    S["peak_tok"] = [repr(words[p]) if p < len(words) else "?" for p in S["peak_pos"]]
    if fam is not None:
        S["family"] = S["cid"].map(fam)
    S = S[S["frac_before"] >= args.min_frac]
    if args.skip_pos >= 0:
        S = S[S["peak_pos"] > args.skip_pos]
    key = {"act": "max_act", "frac": "frac_before", "amp": "frac_amp", "fired": "n_fired"}[args.sort]
    S = S.sort_values([key, "max_act"], ascending=False)
    print("\nQUERY: %r  (%d tokens)" % (text, len(ids)))
    print("tokens: " + " ".join("%d:%s" % (i, repr(w)) for i, w in enumerate(words)))
    print("circuits whose seed fired: %d of %d | by seed layer %s" % (len(S), len(C), S["layer"].value_counts().sort_index().to_dict()))
    print("member-fire fraction (at/before seed peak): median %.2f | p90 %.2f | circuits with >= 0.5: %d | amplitude-weighted median %.2f"
          % (S["frac_before"].median(), S["frac_before"].quantile(0.9), int((S["frac_before"] >= 0.5).sum()), S["frac_amp"].median()))
    print("\n  %-16s %7s %5s %-14s %5s %6s %7s %7s %6s %s" % ("seed", "act", "pos", "token", "n", "fired", "f_bef", "f_amp", "f_any", "family"))
    for _, r in S.head(args.top).iterrows():
        print("  %-16s %7.2f %5d %-14s %5d %6d %7.2f %7.2f %6.2f %s"
              % (r["seed"], r["max_act"], r["peak_pos"], r["peak_tok"][:14], r["n_members"], r["n_fired"], r["frac_before"], r["frac_amp"], r["frac_any"],
                 ("" if fam is None else str(r.get("family", "")))))
    if args.predict:
        # circuits active AT the last position (the prediction position), ranked by activation there
        al = act[act["pos"] == last][["layer", "kind", "index", "act"]].rename(columns={"act": "act_last"})
        L = C[["cid", "seed_layer", "seed_kind", "seed_index", "n_members"]].rename(
            columns={"seed_layer": "layer", "seed_kind": "kind", "seed_index": "index"})
        L["kind"] = L["kind"].astype(str)
        L = L.merge(al, on=["layer", "kind", "index"], how="inner")
        # member-fire fraction at/before the last position for these circuits
        Ml = M[M["cid"].isin(L["cid"])].merge(lat[["layer", "kind", "index", "first_pos"]], on=["layer", "kind", "index"], how="left")
        fr = Ml.assign(b=Ml["first_pos"].notna()).groupby("cid")["b"].mean()
        L["frac"] = L["cid"].map(fr)
        if fam is not None:
            L["family"] = L["cid"].map(fam)
        L = L.sort_values("act_last", ascending=False)
        print("\nCIRCUITS ACTIVE AT THE PREDICTION POSITION (%d:%s): %d | by layer %s"
              % (last, repr(words[last]), len(L), L["layer"].value_counts().sort_index().to_dict()))
        print("  %-16s %7s %5s %5s %6s  %s" % ("seed", "act", "n", "fired", "family", "direct-logit signature: + promoted  - suppressed"))
        for _, r in L.head(args.predict).iterrows():
            print("  %-16s %7.2f %5d %5.2f %6s  %s" % ("%d.%s.%d" % (r["layer"], r["kind"], r["index"]), r["act_last"], r["n_members"], r["frac"],
                                                   ("" if fam is None else str(r.get("family", ""))),
                                                   output_signature(int(r["layer"]), r["kind"], int(r["index"]))))
    if args.inspect:
        # per-latent peak act/pos over the whole text for member detail
        lat_full = lat.set_index(["layer", "kind", "index"])
        for _, r in S.head(args.inspect).iterrows():
            cid = int(r["cid"]); d = Mm[Mm["cid"] == cid].copy()
            d = d.merge(lat[["layer", "kind", "index", "max_act", "peak_pos"]].rename(columns={"peak_pos": "m_pos"}),
                        on=["layer", "kind", "index"], how="left")
            d["site"] = "L" + d["layer"].astype(str) + "." + d["kind"]
            fired = d[d["before"]].copy(); silent = d[~d["before"]]
            fired["score"] = fired["max_act"] * fired["amp"]
            print("\n  ===== %s | seed peak %.2f at pos %d %s | %d members: %d fired at/before peak (%.0f%%), %d silent | amp-weighted %.2f | family %s"
                  % (r["seed"], r["max_act"], r["peak_pos"], r["peak_tok"], len(d), len(fired), 100 * r["frac_before"], len(silent), r["frac_amp"],
                     r.get("family", "-")))
            bl = d.groupby("layer").agg(n=("before", "size"), fired=("before", "sum"))
            print("  fired by layer: " + " ".join("L%d %d/%d" % (l, x.fired, x.n) for l, x in bl.iterrows()))
            bk = d.groupby("kind", observed=True).agg(n=("before", "size"), fired=("before", "sum"))
            print("  fired by kind : " + " ".join("%s %d/%d" % (k, x.fired, x.n) for k, x in bk.iterrows()))
            # where do fired members peak? position histogram with tokens
            pc = fired.groupby("m_pos").size().sort_index()
            print("  fired members' peak positions: " + " ".join("%d:%s=%d" % (p, repr(words[int(p)]) if int(p) < len(words) else "?", n) for p, n in pc.items()))
            print("  %-14s %7s %5s %-12s %6s %7s   (top fired members by act x amplitude)" % ("member", "act", "pos", "token", "amp", "attrib"))
            for _, m in fired.sort_values("score", ascending=False).head(args.inspect_members).iterrows():
                print("  %-14s %7.2f %5d %-12s %6.2f" % ("%s.%d" % (m["site"], m["index"]), m["max_act"], m["m_pos"],
                                                       repr(words[int(m["m_pos"])])[:12] if int(m["m_pos"]) < len(words) else "?", m["amp"]))
            if len(silent):
                top_silent = silent.sort_values("amp", ascending=False).head(10)
                print("  silent members with the largest fitted amplitude: " + ", ".join("%s.%d (amp %.2f)" % (m["site"], m["index"], m["amp"]) for _, m in top_silent.iterrows()))
                late = silent[silent["any"]]
                if len(late):
                    print("  silent-before-peak but active LATER in the text: %d" % len(late))
    if args.families and fam is not None:
        t = S.groupby("family").agg(n=("cid", "size"), f_bef=("frac_before", "median"), act=("max_act", "max")).sort_values("n", ascending=False)
        print("\n  families recruited (circuits with seed fired | median member-fire | max seed act):")
        for f, r in t.head(15).iterrows():
            print("    family %4s | %4d circuits | f_bef %.2f | max act %.2f" % (f, r["n"], r["f_bef"], r["act"]))
    if args.csv:
        S.to_csv(args.csv, index=False); print("  ->", args.csv)
    return S


if args.query:
    for q in args.query:
        run_query(q)
else:
    print("interactive: type a query (blank line to quit)")
    while True:
        try:
            q = input("\nquery> ").strip()
        except EOFError:
            break
        if not q:
            break
        run_query(q)

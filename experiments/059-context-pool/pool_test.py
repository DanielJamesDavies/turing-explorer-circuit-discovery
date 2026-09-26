"""WHICH ACTIVATING CONTEXTS SHOULD WCM TRAIN ON? The 64 strongest vs strongest + mid-band, and more data.

Protocol v1 takes "the 64 strongest" contexts (058: mid-band contexts entered for only 3.3% of targets). This asks
whether circuits trained only on peak firing also explain ordinary firing, and whether mid-band contexts or simply
more training data help. Four training arms, all on the primary config (close C, rank-keep 3e-3, gamma 0.25,
lambda 1e-3, ablation values from the training contexts) and the stratified split (src/circuit/context_split.py):

  D   32 strongest                     (the 48 strongest-train minus a stratified third; ranks 1-2 kept)
  A   48 strongest                     (the protocol-v1 training set)
  B   32 strongest (= D) + 16 mid-band (the same budget as A; D -> A adds 16 strongest, D -> B adds 16 mid)
  C   48 strongest + 48 mid-band = 96  (everything that is not held out)

Every arm is scored on BOTH held-out sets, disjoint from every arm's training data:
  strong-held   16 of the 64 strongest (stratified, rotating offsets)
  mid-held      16 of the mid-band reservoir (stratified)
with ONE evaluation reference for all arms: the A ablation value is the mean over the 48 strongest-train contexts
and C is the close contrast set (stratified 48 / 16), so the cells differ only in the circuit.

Contexts are built once (stage build) and cached, so fitting and scoring see identical sequences. Injection: the
discovery method's build_probe_dataset / _floor_negatives and the eval pass's probe builder / selector are
replaced per target; the engine and the eval code are unchanged (they slice the reordered list as [:n_train]).

  PYTHONPATH=src python experiments/059-context-pool/pool_test.py
  env: STAGE (build | fit | eval | summary)  ARM (D | A | B | C, for fit / eval)  ONLY (comma targets, smoke)
"""
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent / "049-circuit-graph"
E055 = HERE.parent / "055-close-contrast"
STAGE = os.environ.get("STAGE", "build")
ARM = os.environ.get("ARM", "A")
ONLY = [s for s in os.environ.get("ONLY", "").split(",") if s]
CTX = HERE / "data" / "contexts.pt"
POOL_ARMS = ("D", "A", "B", "C")                 # the random-latent pool is built from these (fixed across runs)
ARMS = ("D", "A", "B", "C", "E", "B2", "C800", "Dm", "Em", "Cm")
# matched-contrast arms (Daniel, 2026-09-24): as D / E / C, but the C ablation value averages over as many contrast
# training contexts as the arm has activating training contexts (32 / 64 / 96; A and B are already matched at 48).
# Production equivalent: config.discovery.contrast_context_count = "match".
MATCHED = {"Dm": "D", "Em": "E", "Cm": "C"}
# equal-weight arms (Daniel, 2026-09-25): the protocol's contexts (B, and its resample B2) trained with
# gamma_C = gamma_A = 1 and lambda = 2e-3 instead of 0.25 / 1e-3 (055's eq2 setting). Compared against B / B2.
WEIGHTED = {"Bw": "B", "B2w": "B2"}
ARMS = ARMS + tuple(WEIGHTED)
# follow-ups (2026-09-24): E = 48 strongest + 16 mid; B2 = B's composition from a different stratified sample (the
# noise floor for B vs A); C800 = C at 800 steps, so each context is seen as often as in A / B (the budget control)
DESC = {"D": "32 strongest", "A": "48 strongest", "B": "32 strongest + 16 mid", "C": "48 strongest + 48 mid",
        "E": "48 strongest + 16 mid", "B2": "32 + 16, resampled", "C800": "48 + 48, 800 steps",
        "Dm": "32 strongest, 32 contrast", "Em": "48 + 16 mid, 64 contrast", "Cm": "48 + 48 mid, 96 contrast",
        "Bw": "32 + 16, gamma 1, lambda 2e-3", "B2w": "32 + 16 resampled, gamma 1, lambda 2e-3"}
HELD = ("strong", "mid")
BAND = (0.8, 1.25)
HEAD = ["free0_tk", "freeM_topk_tk", "freeN_topk_tk"]


def seeds():
    s = [x for x in (E055 / "seeds.txt").read_text().split() if x]
    return [x for x in s if x in ONLY] if ONLY else s


def parse(s):
    l, k, i = s.split(".")
    return int(l), k, int(i)


# ----------------------------------------------------------------------------------------------- build
def build(G):
    from circuit.context_split import stratified_split, stratified_subset, stratified_order
    from pipeline.component_index import component_idx as comp_of
    from sae.dense import target_latent_activations
    from store.context import mid_ctx, top_ctx

    inference, bank, M0, KINDS, NK = (G[k] for k in ("inference", "bank", "M0", "KINDS", "NK"))
    pb = M0.probe_builder
    sel = M0._neg_context_selector()

    def load(ids):
        """Tokens for `ids` (same loader path as ProbeDatasetBuilder._load_all_ids) plus the corpus sequence ids actually
        loaded, row-aligned with the tokens (the loader skips ids it cannot locate)."""
        batches = list(pb.loader.get_batches_by_ids(ids, max_length=65))
        t = torch.cat([tk for _, tk in batches], dim=0)
        got = [int(x) for b, _ in batches for x in b.tolist()]
        pos = t[:, :64]
        tgt = t[:, 1:65]
        if tgt.shape[1] < 64:
            tgt = torch.cat([tgt, torch.zeros(tgt.shape[0], 64 - tgt.shape[1], dtype=tgt.dtype, device=tgt.device)], 1)
        return pos, tgt, got

    def peak(l, k, i, tokens):
        vals, args = [], []

        def hook(layer_idx, activations):
            if layer_idx == l:
                ta, ti = bank.encode(activations[KINDS.index(k)], k, layer_idx)
                a = target_latent_activations(ta, ti, i).float()
                vals.append(a.max(-1).values.cpu()); args.append(a.argmax(-1).cpu())
        inference.disable_compile()
        try:
            with torch.no_grad():
                for s0 in range(0, int(tokens.shape[0]), 16):
                    inference.forward(tokens[s0:s0 + 16], activations_callback=hook, return_activations=False,
                                      tokenize_final=False)
        finally:
            inference.enable_compile()
        return torch.cat(vals), torch.cat(args)

    out = {}
    for key in seeds():
        l, k, i = parse(key); comp = comp_of(l, KINDS.index(k), NK)
        top_ids, seen = [], set()
        for sid in top_ctx.ctx_seq_idx[comp, i].tolist():
            if int(sid) > 0 and int(sid) not in seen:
                top_ids.append(int(sid)); seen.add(int(sid))
        mid_ids = []
        for sid in mid_ctx.ctx_seq_idx[comp, i].tolist():
            if int(sid) > 0 and int(sid) not in seen:
                mid_ids.append(int(sid)); seen.add(int(sid))
        rec = dict(n_top=len(top_ids), n_mid=len(mid_ids))
        for name, ids in (("strong", top_ids[:64]), ("mid", mid_ids[:64])):
            if len(ids) < 8:
                rec[name] = None
                continue
            pos, tgt, got = load(ids)
            v, a = peak(l, k, i, pos)
            tr, ho = stratified_split(v)
            rec[name] = dict(pos=pos.cpu(), tgt=tgt.cpu(), arg=a, peak=v, train=tr, held=ho, ids=got)
        s = rec["strong"]
        if s is None:                                   # too few stored contexts: record and move on (thin target)
            out[key] = rec
            print("built %-15s TOO FEW top contexts (%d)" % (key, rec["n_top"]), flush=True)
            continue
        # D: a stratified two thirds of strongest-train (ranks 1-2 kept); B's mid: 16 spread over mid-train
        d_keep, _ = stratified_split(s["peak"][s["train"]], holdout_frac=1 / 3, keep_top=2)
        rec["D_train"] = [s["train"][j] for j in d_keep]
        if rec["mid"] is not None:
            m = rec["mid"]
            rec["B_mid"] = [m["train"][j] for j in stratified_subset(m["peak"][m["train"]], 16)]
        # contrast: close selector, ranked by similarity -> stratified order (48 train first, 16 held last)
        cs = sel.select(comp, i, "close", max_sequences=64, batch_size=16, exact=False, non_activation_threshold=0.0,
                        filter_batch_size=32, load_window_size=256)
        if cs is None or cs.tokens.shape[0] < 4:        # no verified-silent contrast contexts: unusable target
            rec["strong"] = None; rec["no_contrast"] = True; out[key] = rec
            print("built %-15s NO contrast contexts" % key, flush=True)
            continue
        nt = cs.tokens[:64].cpu()
        order = stratified_order(-torch.arange(nt.shape[0], dtype=torch.float64))
        rec["neg"] = nt[order]
        nids = [int(x) for x in list(cs.sequence_ids)[:nt.shape[0]]]
        rec["neg_ids"] = [nids[j] for j in order] if len(nids) == nt.shape[0] else None   # row-aligned with rec["neg"]
        out[key] = rec
        print("built %-15s top %d mid %d | strong peak %.1f-%.1f | mid peak %s | neg %d"
              % (key, rec["n_top"], rec["n_mid"], float(s["peak"].min()), float(s["peak"].max()),
                 "-" if rec["mid"] is None else "%.1f-%.1f" % (float(rec["mid"]["peak"].min()), float(rec["mid"]["peak"].max())),
                 int(nt.shape[0])), flush=True)
    CTX.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, CTX)


# ----------------------------------------------------------------------------------------------- sets
def pick(rec, name, idx):
    d = rec[name]
    return d["pos"][idx], d["tgt"][idx], d["arg"][idx]


def cat(parts):
    return tuple(torch.cat([p[j] for p in parts], 0) for j in range(3))


def build_neg128(G):
    """Extend each target's contrast pool to 128 close contexts. The first 64 must be exactly the cached 64 (the
    selector ranks deterministically), so the held-out contrast set is unchanged; ranks 65-128 become the extra
    training pool for the matched arms."""
    from pipeline.component_index import component_idx as comp_of
    ctx = torch.load(CTX, weights_only=False)
    KINDS, NK = G["KINDS"], G["NK"]
    sel = G["M0"]._neg_context_selector()
    for key, rec in ctx.items():
        l, k, i = parse(key); comp = comp_of(l, KINDS.index(k), NK)
        cs = sel.select(comp, i, "close", max_sequences=128, batch_size=16, exact=False, non_activation_threshold=0.0,
                        filter_batch_size=32, load_window_size=256)
        nt = cs.tokens[:128].cpu()
        old = {tuple(r.tolist()) for r in rec["neg"]}
        head = {tuple(r.tolist()) for r in nt[:rec["neg"].shape[0]]}
        rec["neg_prefix_ok"] = head == old
        rec["neg_extra"] = nt[rec["neg"].shape[0]:]                   # ranks 65-128, most similar first
        print("neg128 %-15s got %d | prefix identical %s | extra %d" % (key, nt.shape[0], rec["neg_prefix_ok"],
                                                                      rec["neg_extra"].shape[0]), flush=True)
    torch.save(ctx, CTX)


def matched_neg(rec, n_act):
    """Contrast list whose TRAINING part (under the engine's split rule) has n_act contexts: a similarity-stratified
    subset of the 48 contrast-train contexts when n_act <= 48, else all 48 plus a stratified (n_act - 48) of ranks
    65-128; padded with the 16 held-out contrast contexts (then leftover extras) as the engine's held-out part."""
    from circuit.context_split import stratified_subset
    n_tr0 = rec["neg"].shape[0] - int(round(rec["neg"].shape[0] * 0.25))
    train0, held = rec["neg"][:n_tr0], rec["neg"][n_tr0:]
    extra = rec.get("neg_extra")
    if n_act <= n_tr0:
        idx = stratified_subset(-torch.arange(n_tr0, dtype=torch.float64), n_act)
        train, left = train0[idx], train0[[j for j in range(n_tr0) if j not in set(idx)]]
    else:
        k = min(n_act - n_tr0, int(extra.shape[0]))
        idx = stratified_subset(-torch.arange(extra.shape[0], dtype=torch.float64), k)
        train = torch.cat([train0, extra[idx]], 0)
        left = extra[[j for j in range(extra.shape[0]) if j not in set(idx)]]
    n_tr = int(train.shape[0])
    n = n_tr
    while n - int(round(n * 0.25)) < n_tr:
        n += 1
    pad = torch.cat([held, left], 0)[:n - n_tr]
    return torch.cat([train, pad], 0)


def train_set(rec, arm):
    from circuit.context_split import stratified_split, stratified_subset
    arm = MATCHED.get(arm, WEIGHTED.get(arm, arm))
    s = rec["strong"]
    if arm == "D":
        return cat([pick(rec, "strong", rec["D_train"])])
    if arm == "A":
        return cat([pick(rec, "strong", s["train"])])
    if rec["mid"] is None:
        return None
    m = rec["mid"]
    if arm == "B":
        return cat([pick(rec, "strong", rec["D_train"]), pick(rec, "mid", rec["B_mid"])])
    if arm in ("C", "C800"):
        return cat([pick(rec, "strong", s["train"]), pick(rec, "mid", m["train"])])
    if arm == "E":
        return cat([pick(rec, "strong", s["train"]), pick(rec, "mid", rec["B_mid"])])
    if arm == "B2":                                  # same shape as B, different stratified sample of each pool
        keep, _ = stratified_split(s["peak"][s["train"]], holdout_frac=1 / 3, keep_top=2, offset=3)
        mids = stratified_subset(m["peak"][m["train"]], 16, phase=0.17)
        return cat([pick(rec, "strong", [s["train"][j] for j in keep]), pick(rec, "mid", [m["train"][j] for j in mids])])
    raise ValueError(arm)


def probe(rec, train, held_parts, dev):
    """A ProbeDataset whose first n_train = len(train) contexts are the training set: pad with held-out contexts to
    the length n where n - round(n / 4) == n_train (the engine's split rule), so every consumer's slice is right."""
    from circuit.probe_dataset import ProbeDataset
    n_tr = int(train[0].shape[0])
    n = n_tr
    while n - int(round(n * 0.25)) < n_tr:
        n += 1
    while n - int(round(n * 0.25)) > n_tr:          # not reachable for these sizes; guard only
        n -= 1
    filler = cat(held_parts) if held_parts else None
    need = n - n_tr
    parts = [train] + ([tuple(x[:need] for x in filler)] if need and filler is not None else [])
    pos, tgt, arg = cat(parts)
    return ProbeDataset(pos_tokens=pos.to(dev), target_tokens=tgt.to(dev), neg_tokens=rec["neg"].to(dev),
                        pos_argmax=arg.to(dev), metadata=dict(n_train=n_tr, n=int(pos.shape[0])))


# ----------------------------------------------------------------------------------------------- fit
def fit(G):
    from analysis.circuits.gradient_size_sweep_runner import _build_mode_method
    from config import config
    ctx = torch.load(CTX, weights_only=False)
    disc = config.discovery
    lm = disc.learned_mask
    lm.mask_floor_source = "triple"; lm.free_amplitude = True; lm.l1_lambda = 1e-3; lm.deep_site_threshold = 99
    lm.dual_floor_weight = 0.25; lm.triple_floor_weight = 0.25; lm.offtarget_weight = 0.0
    lm.rank_weight = 3e-3; lm.rank_mode = "keep"; lm.floors_train_only = True
    if ARM == "C800":
        lm.steps = 800                                                   # budget control: ~33 passes per context
    if ARM in WEIGHTED:                                                  # equal ablation-term weights, node price doubled
        lm.dual_floor_weight = 1.0; lm.triple_floor_weight = 1.0; lm.l1_lambda = 2e-3
    disc.floor_negctx_mode = "close"; disc.eval_batch_size = 64
    disc.probe_sequence_count = 128; disc.eval_sequence_count = 128      # room for arm C's 96 + 32
    M = _build_mode_method("ablation_gradient", "mask", G["inference"], G["bank"], G["avg_acts"], G["M0"].probe_builder)
    dev = G["device"]
    KINDS = G["KINDS"]
    out_dir = HERE / ("data_%s" % ARM); out_dir.mkdir(exist_ok=True)
    path = out_dir / "discovered_circuits.shard0.pt"
    found = torch.load(path, weights_only=False) if path.exists() else {}
    for key in seeds():
        if key in found or key not in ctx:
            continue
        rec = ctx[key]
        tr = train_set(rec, ARM)
        if tr is None:
            print("  %-15s skipped (no mid-band pool)" % key, flush=True); continue
        held = [pick(rec, "strong", rec["strong"]["held"])] + ([pick(rec, "mid", rec["mid"]["held"])] if rec["mid"] else [])
        pdset = probe(rec, tr, held, dev)
        if ARM in MATCHED:
            if not rec.get("neg_prefix_ok", False):
                print("  %-15s skipped (contrast prefix mismatch)" % key, flush=True); continue
            pdset.neg_tokens = matched_neg(rec, int(tr[0].shape[0])).to(dev)
        M.build_probe_dataset = lambda comp, i, _p=pdset: _p
        M._floor_negatives = lambda probe_data, comp, i, logger: probe_data.neg_tokens
        l, k, i = parse(key); comp = l * len(KINDS) + KINDS.index(k)
        ts = time.time()
        try:
            c = M.discover(comp, i)
        except Exception as e:  # noqa: BLE001
            print("  %-15s ERROR %s: %s" % (key, type(e).__name__, str(e)[:200]), flush=True); continue
        if c is None:
            print("  %-15s rejected" % key, flush=True); continue
        found[key] = c
        n = sum(1 for nd in c.nodes.values() if nd.metadata.get("role") != "seed")
        print("  %-15s arm %s  train %3d  %4d nodes  %.0fs" % (key, ARM, pdset.metadata["n_train"], n, time.time() - ts),
              flush=True)
        torch.save(found, path)


# ----------------------------------------------------------------------------------------------- eval
class FixedSelector:
    def __init__(self, tokens):
        self.tokens = tokens

    def select(self, *a, **k):
        from utils.neg_context_selector import NegContextSelection
        return NegContextSelection(tokens=self.tokens, sequence_ids=[], mode="close", metadata={})


def evaluate(G, V):
    ctx = torch.load(CTX, weights_only=False)
    dev = G["device"]
    M0 = G["M0"]
    # one random-latent pool for every arm (members of all four arms' circuits), as the eval pass builds it
    pool = defaultdict(set); circuits = {}
    for arm in ARMS:
        p = HERE / ("data_%s" % arm) / "discovered_circuits.shard0.pt"
        if not p.exists():
            continue
        cs = torch.load(p, weights_only=False, map_location="cpu")
        circuits[arm] = cs
        if arm not in POOL_ARMS:
            continue
        for c in cs.values():
            for n in c.nodes.values():
                if n.metadata.get("role") != "seed":
                    f = n.metadata["feature_id"]; pool[(f.layer, f.kind)].add(f.index)
    pool = {s: np.array(sorted(v)) for s, v in pool.items()}
    out = HERE / "results" / ("eval_%s.jsonl" % ARM); out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["seed"], r["held"]) for r in map(json.loads, open(out)) if "error" not in r}
    with open(out, "a") as fh:
        for key in seeds():
            c = circuits.get(ARM, {}).get(key)
            if c is None:
                continue
            rec = ctx[key]
            for h in HELD:
                if (key, h) in done or rec[h] is None:
                    continue
                strong_tr = pick(rec, "strong", rec["strong"]["train"])
                pd_ = probe(rec, strong_tr, [pick(rec, h, rec[h]["held"])], dev)
                M0.build_probe_dataset = lambda comp, i, _p=pd_: _p
                M0._neg_context_selector = lambda _t=rec["neg"].to(dev): FixedSelector(_t)
                try:
                    row = V.score_circuit(c, pool, skip_roles=True)
                    row.update(arm=ARM, held=h, n_train_fit=int(train_set(rec, ARM)[0].shape[0]))
                except Exception as e:  # noqa: BLE001
                    row = dict(seed=key, held=h, arm=ARM, error="%s: %s" % (type(e).__name__, str(e)[:300]))
                fh.write(json.dumps(row) + "\n"); fh.flush()
                print("  %-15s arm %s held %-6s n %s  free0 %s  fM %s  fN %s" % (
                    key, ARM, h, row.get("n"), *["%.2f" % row[f] if isinstance(row.get(f), float) else "-" for f in HEAD]),
                    flush=True)


# ----------------------------------------------------------------------------------------------- summary
def summary():
    import pandas as pd
    rows = []
    for arm in ARMS:
        p = HERE / "results" / ("eval_%s.jsonl" % arm)
        if p.exists():
            rows += [r for r in map(json.loads, open(p)) if "error" not in r and "skip" not in r]
    df = pd.DataFrame(rows)
    lo, hi = BAND
    lines = ["# 059 activating-context pool test (%d rows)\n" % len(df),
             "Primary config, stratified split. All arms scored on the same two held-out sets with one evaluation "
             "reference (A ablation value from the 48 strongest-train contexts; close contrast set). Activation read. "
             "Pass = all three faithfulness scores in [%.2f, %.2f] (illustrative; the pass rule is DAN-8).\n" % BAND]
    common = set.intersection(*[set(df[df.arm == a].seed) for a in ARMS if (df.arm == a).any()])
    lines.append("Targets scored by every arm: %d (arms B/C need a mid-band pool).\n" % len(common))
    d = df[df.seed.isin(common)]
    for h in HELD:
        x = d[d.held == h]
        lines.append("## Held-out: %s\n" % ("strongest (16 of the 64 strongest)" if h == "strong" else "mid-band (16 of the reservoir)"))
        lines.append("| arm | trains on | nodes (median) | free0 | freeM_topk | freeN_topk | worst-of-3 dev | necessity | "
                     "sufficiency to induce | pass | closer to 1 than A | closer to 1 than B |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        dev_of = {arm: (x[x.arm == arm].set_index("seed")[HEAD] - 1).abs() for arm in ARMS if (x.arm == arm).any()}
        for arm in ARMS:
            a = x[x.arm == arm]
            if a.empty:
                continue
            inb = ((a[HEAD] >= lo) & (a[HEAD] <= hi)).all(axis=1)
            med = lambda f: ("%.3f" % a[f].median()) if f in a and a[f].notna().any() else "-"

            def closer(ref):
                if ref not in dev_of or ref == arm:
                    return "-"
                d = (dev_of[arm].mean(axis=1) - dev_of[ref].mean(axis=1)).dropna()
                return "%d/%d (%+.3f)" % (int((d < 0).sum()), len(d), d.median())
            lines.append("| %s | %s | %d | %s | %s | %s | %.2f | %s | %s | %d/%d | %s | %s |" % (
                arm, DESC[arm], int(a.n.median()), med("free0_tk"), med("freeM_topk_tk"), med("freeN_topk_tk"),
                dev_of[arm].max(axis=1).median(), med("phi_sup_blind_tk"), med("phi_cf_alpha_blind_tk"),
                int(inb.sum()), len(a), closer("A"), closer("B")))
        lines.append("")
    # node overlap between arms
    lines.append("## Node overlap between arms (median Jaccard over common targets)\n")
    mem = {}
    for arm in ARMS:
        p = HERE / ("data_%s" % arm) / "discovered_circuits.shard0.pt"
        if p.exists():
            cs = torch.load(p, weights_only=False, map_location="cpu")
            for c in cs.values():
                seed = next(n for n in c.nodes.values() if n.metadata.get("role") == "seed").metadata["feature_id"]
                key = "%d.%s.%d" % (seed.layer, seed.kind, seed.index)
                mem[(arm, key)] = {(n.metadata["feature_id"].layer, n.metadata["feature_id"].kind, n.metadata["feature_id"].index)
                                   for n in c.nodes.values() if n.metadata.get("role") != "seed"}
    for a, b in (("D", "A"), ("D", "B"), ("A", "B"), ("A", "C"), ("B", "C"), ("B", "B2"), ("A", "E"), ("B", "E"),
                 ("C", "C800"), ("B", "Bw"), ("B2", "B2w"), ("Bw", "B2w")):
        js = [len(mem[(a, s)] & mem[(b, s)]) / max(1, len(mem[(a, s)] | mem[(b, s)])) for s in common
              if (a, s) in mem and (b, s) in mem]
        if js:
            lines.append("- %s vs %s: %.2f (n=%d)" % (a, b, float(np.median(js)), len(js)))
    txt = "\n".join(lines)
    print(txt)
    (HERE / "results" / "summary.md").write_text(txt, encoding="utf-8")


def main():
    if STAGE == "summary":
        summary(); return
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("TAG", "pool_unused")
    os.environ.setdefault("CTR_SOURCE", "close")
    import amp_eval_pass_v2 as V
    G = V.setup()
    {"build": lambda: build(G), "build_neg128": lambda: build_neg128(G), "fit": lambda: fit(G),
     "eval": lambda: evaluate(G, V)}[STAGE]()


if __name__ == "__main__":
    main()

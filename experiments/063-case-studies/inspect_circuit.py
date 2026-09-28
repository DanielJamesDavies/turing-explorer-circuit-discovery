"""CIRCUIT BROWSER: a compact, readable report per protocol-v1 circuit, for finding case studies.

Per circuit (one markdown file, results/reports/<target>.md):
  - the target: scores under the DAN-8 rule (pass, Z / A / C, necessity, sufficiency to induce, size, amplifier and
    near-threshold flags), its peak-token histogram and consistency, windows from its top contexts with the peak
    token marked, a contrast-context snippet, and its first-order logit effect;
  - the top members by CONTRIBUTION at the target's anchor, alpha x mean activation at the anchor x decoder norm
    (how much each writes into the stream where the target fires), with: engagement on the target's activating
    contexts vs its contrast contexts, its own peak tokens and consistency (a high share on one token marks a
    string detector), windows from its own top contexts, and (for resid/mlp) its logit effect;
  - the circuit's composition by site kind and layer.
Token-driven reads only, no auto-interp labels. The logit effect is decoder x final-norm gain x unembedding: a
"what would it say" read, not a causal claim.

  PYTHONPATH=src python experiments/063-case-studies/inspect_circuit.py
  env: OUT (062 output dir with main/ and ctx/; default the stage-1 download)
       KEYS (comma list of targets)  or  SELECT=1 (all passing, non-amplifier, non-near-threshold circuits)
                                     or  SELECT=deep (passing, non-near-threshold, layer >= 6, amplifiers included)
                                     or  SELECT=deepconcept (deep + peak-token consistency <= MAX_CONS, default 0.7,
                                         read from the context cache before any model pass)
       N_MEMBERS (12)  N_CTX (6 own contexts per member)  N_WIN (target windows, 6)  LIMIT (first N keys)
       FORCE=1 (regenerate reports that already exist; otherwise resumable)
       RESULTS (default experiments/063-case-studies/results)
"""
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent
EXP = HERE.parent
for d in ("049-circuit-graph", "062-h100-protocol-v1"):
    sys.path.insert(0, str(EXP / d))
OUT = Path(os.environ.get("OUT", str(EXP / "062-h100-protocol-v1" / "out_stage1" / "out")))
RESULTS = Path(os.environ.get("RESULTS", str(HERE / "results")))
N_MEMBERS = int(os.environ.get("N_MEMBERS", 12))
N_CTX = int(os.environ.get("N_CTX", 6))
N_WIN = int(os.environ.get("N_WIN", 6))
LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
K = 128
HEAD = ["free0_tk", "freeM_topk_tk", "freeN_topk_tk"]


def parse(key):
    l, k, i = key.split(".")
    return int(l), k, int(i)


def scores_table():
    """Per-target scores and flags from the run: eval rows (held-out strongest) + merge.py's targets.csv."""
    import pass_rule as P
    os.environ["OUT"] = str(OUT)
    P.OUT = OUT
    d = P.load()
    vac = d.get("vacuous_tk", pd.Series(False, index=d.index)).fillna(False).astype(bool)
    d["passes"] = P.band(d, P.FAITH["tk"], .8, 1.5) & (d[P.NEC] >= .9) & ~vac
    d["worst_dev"] = (d[HEAD].astype(float) - 1).abs().max(axis=1)
    return d


def select(d, mode="1"):
    """SELECT=1: passing, not amplifier-flagged, not near-threshold (the first case-study screen).
    SELECT=deep: passing, not near-threshold, layer >= 6, amplifier-flagged INCLUDED (a concept target whose circuit
    also lifts related concepts can carry the flag; the flag is printed in each report)."""
    if mode in ("deep", "deepconcept"):
        good = d[d.passes & ~d.near & (d.layer >= 6)]
        if mode == "deepconcept":
            # SELECT=deepconcept: also drop string detectors BEFORE any model pass. The context cache stores each
            # target's activating tokens and anchors, so the peak-token consistency (share of its strongest contexts
            # peaking on its most common token) costs one file read per target. Keep consistency <= MAX_CONS.
            from collections import Counter
            cap = float(os.environ.get("MAX_CONS", 0.7))
            keep = []
            for key in good.index:
                rec = torch.load(OUT / "ctx" / ("%s.pt" % key), weights_only=False)[key]
                pos, arg = rec["strong"]["pos"], rec["strong"]["arg"].long()
                toks = Counter(int(pos[b, int(arg[b])]) for b in range(pos.shape[0]))
                if toks.most_common(1)[0][1] / pos.shape[0] <= cap:
                    keep.append(key)
            print("deepconcept: %d deep passing circuits, %d with peak-token consistency <= %.0f%%"
                  % (len(good), len(keep), 100 * cap), flush=True)
            good = good.loc[keep]
    else:
        good = d[d.passes & (d.amp_any != True) & ~d.near]                # noqa: E712
    # deep targets first (fewer, and the more interesting ones), then by kind
    return good.sort_values(["layer", "kind"], ascending=[False, True]).index.tolist()


class Browser:
    def __init__(self):
        os.environ.setdefault("TAG", "inspect_unused")
        import amp_eval_pass_v2 as V
        from model.tokenizer import Tokenizer
        from store.context import top_ctx
        self.G = G = V.setup()
        self.inference, self.bank, self.M0, self.KINDS = G["inference"], G["bank"], G["M0"], G["KINDS"]
        self.NK, self.D = G["NK"], G["D"]
        self.top_ctx = top_ctx
        self.loader = self.M0.probe_builder.loader
        self.tok = Tokenizer()
        model = self.inference.model
        self.g_norm = model.transformer.norm_f.scale.detach().float()
        self.W_U = model.lm_head.weight.detach().float()

    # ------------------------------------------------------------------------------------------ helpers
    def dec(self, ids):
        return self.tok.decode([int(t) for t in ids]).replace("\n", "\\n")

    def window(self, ids, p, before=12, after=3):
        return (self.dec(ids[max(0, p - before):p]) + " [[" + self.dec(ids[p:p + 1]) + "]]"
                + self.dec(ids[p + 1:p + 1 + after]))

    @staticmethod
    def distinct(windows, n):
        """The first n windows with distinct text (the corpus repeats some passages under different sequence ids)."""
        out = []
        for w in windows:
            if w not in out:
                out.append(w)
            if len(out) >= n:
                break
        return out

    def logits(self, l, k, i, n=6):
        wd = self.bank.saes[k][l].decoder.weight.detach()[:, i].float().to(self.W_U.device)
        lg = (wd * self.g_norm.to(wd.device)) @ self.W_U.T
        return [self.dec([t]) for t in lg.topk(n).indices.tolist()], [self.dec([t]) for t in (-lg).topk(n).indices.tolist()]

    def dec_norm(self, l, k, i):
        return float(self.bank.saes[k][l].decoder.weight.detach()[:, i].float().norm())

    def acts(self, nodes, tokens):
        """Dense activations [B, T, len(nodes)] of the given (layer, kind, index) latents on tokens [B, T]."""
        from sae.dense import sparse_topk_to_dense
        want = {}
        for j, (l, k, i) in enumerate(nodes):
            want.setdefault((l, k), []).append((j, i))
        k2i = {k: n for n, k in enumerate(self.KINDS)}
        out = torch.zeros(tokens.shape[0], tokens.shape[1], len(nodes))
        rows = [0]

        def hook(layer_idx, activations):
            for kd in self.KINDS:
                s = (layer_idx, kd)
                if s not in want:
                    continue
                ta, ti = self.bank.encode(activations[k2i[kd]], kd, layer_idx)
                cols = [j for j, _ in want[s]]
                idx = torch.tensor([i for _, i in want[s]], device=ta.device)
                dense = sparse_topk_to_dense(ta, ti, self.D, dtype=torch.float32)[..., idx].cpu()
                out[rows[0]:rows[0] + dense.shape[0], :dense.shape[1], cols] = dense

        self.inference.disable_compile()
        try:
            with torch.no_grad():
                for s0 in range(0, int(tokens.shape[0]), 16):
                    rows[0] = s0
                    self.inference.forward(tokens[s0:s0 + 16].to(self.G["device"]), activations_callback=hook,
                                           return_activations=False, tokenize_final=False)
        finally:
            self.inference.enable_compile()
        return out

    def own_contexts(self, l, k, i):
        """Tokens [n, 64] of the latent's strongest stored contexts (store order = strongest first)."""
        from pipeline.component_index import component_idx as comp_of
        comp = comp_of(l, self.KINDS.index(k), self.NK)
        ids, seen = [], set()
        for sid in self.top_ctx.ctx_seq_idx[comp, i].tolist():
            if int(sid) > 0 and int(sid) not in seen:
                ids.append(int(sid)); seen.add(int(sid))
            if len(ids) >= N_CTX:
                break
        if not ids:
            return None
        batches = list(self.loader.get_batches_by_ids(ids, max_length=64))
        return torch.cat([t for _, t in batches], 0)

    # ------------------------------------------------------------------------------------------ report
    def report(self, key, row):
        tl, tk, ti = parse(key)
        c = torch.load(OUT / "main" / "circuits" / ("%s.pt" % key), weights_only=False)
        rec = torch.load(OUT / "ctx" / ("%s.pt" % key), weights_only=False)[key]
        members = []
        for nd in c.nodes.values():
            md = nd.metadata
            if md.get("role") == "seed":
                continue
            f = md["feature_id"]
            members.append((int(f.layer), str(f.kind), int(f.index), float(md.get("amplitude", 1.0))))
        nodes = [(l, k, i) for l, k, i, _ in members]
        pos, arg = rec["strong"]["pos"], rec["strong"]["arg"].long()
        neg = rec["neg"]
        A_pos = self.acts(nodes + [(tl, tk, ti)], pos)                 # [B, T, n + 1]
        A_neg = self.acts(nodes, neg)
        rr = torch.arange(pos.shape[0])
        at_anchor = A_pos[rr, arg.clamp(0, pos.shape[1] - 1), :]      # [B, n + 1]
        contrib = []
        for j, (l, k, i, a) in enumerate(members):
            m_anchor = float(at_anchor[:, j].mean())
            contrib.append(dict(node="%d.%s.%d" % (l, k, i), l=l, k=k, i=i, alpha=a,
                                anchor=m_anchor, fire=float((at_anchor[:, j] > 0).float().mean()),
                                pos_max=float(A_pos[:, :, j].max(1).values.mean()),
                                neg_max=float(A_neg[:, :, j].max(1).values.mean()),
                                contrib=abs(a) * m_anchor * self.dec_norm(l, k, i)))
        cdf = pd.DataFrame(contrib).sort_values("contrib", ascending=False)
        total = cdf.contrib.sum()

        lines = ["# %s" % key, ""]
        flags = []
        if row is not None:
            # amp_any is a pandas/numpy bool (or NaN): test its truth value, never identity with the Python True
            flags = ["PASS" if row.passes else "FAIL", "amplifier" if (pd.notna(row.amp_any) and bool(row.amp_any))
                     else "not amplifier",
                     "near-threshold" if row.near else "clean rank %s" % ("%.0f" % row.rank_clean
                                                                           if pd.notna(row.rank_clean) else "?")]
            lines.append("**%s** | %d members | Z %.2f A %.2f C %.2f | worst |dev| %.2f | necessity %.2f | "
                         "induce %.2f" % (" · ".join(flags), len(members), row.free0_tk, row.freeM_topk_tk,
                                          row.freeN_topk_tk, row.worst_dev, row.phi_sup_blind_tk,
                                          row.phi_cf_alpha_blind_tk))
        comp = cdf.groupby("k").size().to_dict()
        lay = cdf.groupby("l").size()
        lines.append("composition: %s | layers %s" % (comp, ", ".join("L%d:%d" % (l, n) for l, n in lay.items())))
        # target
        peaks = Counter(self.dec([pos[b, int(arg[b])]]) for b in range(pos.shape[0]))
        top = peaks.most_common(6)
        up, down = self.logits(tl, tk, ti)
        lines += ["", "## Target", "peak tokens: %s | consistency %.0f%%" % (
            ", ".join("%r×%d" % pc for pc in top), 100 * top[0][1] / pos.shape[0])]
        lines += ["- …%s" % w for w in self.distinct(
            (self.window(pos[b].tolist(), int(arg[b])) for b in range(pos.shape[0])), N_WIN)]
        lines.append("contrast (target silent): …%s…" % self.dec(neg[0, 20:44].tolist()))
        lines.append("logit effect: + %s | − %s" % (up, down))
        # members
        lines += ["", "## Top %d members by contribution at the target's anchor (share of the circuit's total)" % N_MEMBERS]
        top_m = cdf.head(N_MEMBERS)
        own_tok = {r.node: self.own_contexts(r.l, r.k, r.i) for r in top_m.itertuples()}
        for r in top_m.itertuples():
            t = own_tok[r.node]
            head = "### %s  α %.2f | share %.1f%% | fires at anchor %.0f%% | seq max: on target ctx %.2f vs contrast %.2f" % (
                r.node, r.alpha, 100 * r.contrib / total if total else 0, 100 * r.fire, r.pos_max, r.neg_max)
            lines.append(head)
            if t is None:
                lines.append("(no stored contexts)")
                continue
            a = self.acts([(r.l, r.k, r.i)], t)[:, :, 0]
            pk = a.argmax(1)
            toks = Counter(self.dec([t[b, int(pk[b])]]) for b in range(t.shape[0]))
            tt = toks.most_common(4)
            lines.append("own peaks: %s | consistency %.0f%%%s" % (
                ", ".join("%r×%d" % pc for pc in tt), 100 * tt[0][1] / t.shape[0],
                " (LEXICAL)" if tt[0][1] / t.shape[0] >= 0.8 else ""))
            lines += ["- …%s" % w for w in self.distinct(
                (self.window(t[b].tolist(), int(pk[b])) for b in range(t.shape[0])), 2)]
            if r.k != "attn":
                u, _ = self.logits(r.l, r.k, r.i, n=5)
                lines.append("logits +: %s" % u)
        rest = cdf.iloc[N_MEMBERS:]
        lines.append("\nremaining %d members carry %.0f%% of the contribution" % (
            len(rest), 100 * rest.contrib.sum() / total if total else 0))
        return "\n".join(lines) + "\n", cdf


def main():
    d = scores_table()
    keys = ([k for k in os.environ.get("KEYS", "").split(",") if k]
            or (select(d, os.environ["SELECT"]) if os.environ.get("SELECT") else []))
    if not keys:
        raise SystemExit("give KEYS=... or SELECT=1")
    keys = keys[:LIMIT] if LIMIT else keys
    (RESULTS / "reports").mkdir(parents=True, exist_ok=True)
    (RESULTS / "members").mkdir(parents=True, exist_ok=True)
    B = Browser()
    t0 = time.time()
    for n, key in enumerate(keys):
        path = RESULTS / "reports" / ("%s.md" % key)
        if path.exists() and not os.environ.get("FORCE"):         # resumable; FORCE=1 regenerates
            continue
        try:
            txt, cdf = B.report(key, d.loc[key] if key in d.index else None)
            path.write_text(txt, encoding="utf-8")
            cdf.to_csv(RESULTS / "members" / ("%s.csv" % key), index=False)
        except Exception as e:  # noqa: BLE001
            print("  %s FAILED: %s: %s" % (key, type(e).__name__, e), flush=True)
            continue
        print("%4d/%d %-16s %.0fs elapsed" % (n + 1, len(keys), key, time.time() - t0), flush=True)
    (RESULTS / "index.csv").write_text(d.loc[[k for k in keys if k in d.index],
                                             ["layer", "kind", "n", "passes", "amp_any", "near", "worst_dev"] + HEAD]
                                       .to_csv(), encoding="utf-8")


if __name__ == "__main__":
    main()

"""MEMBER REPORTS on the production (v4) circuits: 063's inspect_circuit.Browser, pointed at the production OUT and
the fresh (deduplicated) records.

Per target (out/reports/<key>.md + out/members/<key>.csv):
  - scores (held-out strongest and mid-band rows, merge.py pass rule), size, composition;
  - the target's peak tokens on its strongest contexts and windows around its anchors (fresh records);
  - every member's contribution at the target's anchor (|alpha| x mean activation at the anchor x decoder norm, as
    063) with engagement on activating vs contrast contexts;
  - the top N_MEMBERS members with their OWN strongest stored contexts (outputs/top_ctx), own peak-token consistency
    (>= 80% on one token = LEXICAL) and logit effect;
  - if the 062 full run has a circuit for the same target: old membership / old rank of each member.

  PYTHONPATH=src python -X utf8 experiments/110-case-studies-v4/report.py
  env: KEYS (comma list)  N_MEMBERS (40)  N_CTX (6)  N_WIN (10)  FORCE=1
"""
import os
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib110 as L  # noqa: E402

L.use_production()
N_MEMBERS = int(os.environ.get("N_MEMBERS", 40))
os.environ.setdefault("N_CTX", "6")
N_WIN = int(os.environ.get("N_WIN", 10))


def make_browser(G):
    """inspect_circuit.Browser on an already set-up scorer G (no second model load)."""
    import inspect_circuit as IC
    from model.tokenizer import Tokenizer
    from store.context import top_ctx
    B = object.__new__(IC.Browser)
    B.G = G
    B.inference, B.bank, B.M0, B.KINDS = G["inference"], G["bank"], G["M0"], G["KINDS"]
    B.NK, B.D = G["NK"], G["D"]
    B.top_ctx = top_ctx
    B.loader = B.M0.probe_builder.loader
    B.tok = Tokenizer()
    model = B.inference.model
    B.g_norm = model.transformer.norm_f.scale.detach().float()
    B.W_U = model.lm_head.weight.detach().float()
    return B


def member_table(B, key, c, rec):
    tl, tk, ti = L.parse(key)
    mem = L.members(c)
    nodes = [(l, k, i) for l, k, i, _ in mem]
    pos, arg = rec["strong"]["pos"], rec["strong"]["arg"].long()
    A_pos = B.acts(nodes + [(tl, tk, ti)], pos)
    A_neg = B.acts(nodes, rec["neg"])
    rr = torch.arange(pos.shape[0])
    at = A_pos[rr, arg.clamp(0, pos.shape[1] - 1), :]
    rows = []
    for j, (l, k, i, a) in enumerate(mem):
        m = float(at[:, j].mean())
        rows.append(dict(node="%d.%s.%d" % (l, k, i), l=l, k=k, i=i, alpha=a, anchor=m,
                         fire=float((at[:, j] > 0).float().mean()),
                         pos_max=float(A_pos[:, :, j].max(1).values.mean()),
                         neg_max=float(A_neg[:, :, j].max(1).values.mean()),
                         contrib=abs(a) * m * B.dec_norm(l, k, i)))
    df = pd.DataFrame(rows).sort_values("contrib", ascending=False).reset_index(drop=True)
    df["share"] = df.contrib / df.contrib.clip(lower=0).sum()
    return df


def report(B, key, row, mid):
    tl, tk, ti = L.parse(key)
    c = L.load_circuit(key)
    rec = L.record(key)
    cdf = member_table(B, key, c, rec)
    old = None
    if (L.OLD_OUT / "main" / "circuits" / ("%s.pt" % key)).exists():
        oc = L.load_circuit(key, old=True)
        old = {"%d.%s.%d" % (l, k, i): a for l, k, i, a in L.members(oc)}
        cdf["in_old"] = cdf.node.isin(old)
    pos, arg = rec["strong"]["pos"], rec["strong"]["arg"].long()
    f = lambda v: "—" if v is None or pd.isna(v) else "%.2f" % v
    lines = ["# %s (production v4 circuit, fresh records)" % key, ""]
    if row is not None:
        lines.append("**%s** | %d members | held-out strongest Z %s A %s C %s | worst |dev| %s | necessity %s | "
                     "induce %s" % ("PASS" if row.passes else ("VACUOUS" if row.vacuous else "FAIL"), len(cdf),
                                    f(row.free0_tk), f(row.freeM_topk_tk), f(row.freeN_topk_tk), f(row.worst_dev),
                                    f(row.phi_sup_blind_tk), f(row.phi_cf_alpha_blind_tk)))
    if mid is not None:
        lines.append("mid-band: %s | Z %s A %s C %s | necessity %s" % (
            "PASS" if mid.passes else ("VACUOUS" if mid.vacuous else "FAIL"), f(mid.free0_tk), f(mid.freeM_topk_tk),
            f(mid.freeN_topk_tk), f(mid.phi_sup_blind_tk)))
    if old is not None:
        a, b = set(cdf.node), set(old)
        lines.append("062 full-run circuit: %d members | overlap %d | Jaccard %.3f | share of v4 contribution on "
                     "shared members %.0f%%" % (len(b), len(a & b), len(a & b) / len(a | b),
                                               100 * cdf[cdf.in_old].share.clip(lower=0).sum()))
    lines.append("composition: %s | layers %s" % (cdf.groupby("k").size().to_dict(),
                                                   ", ".join("L%d:%d" % (l, n) for l, n in cdf.groupby("l").size().items())))
    peaks = Counter(B.dec([pos[b, int(arg[b])]]) for b in range(pos.shape[0]))
    top = peaks.most_common(6)
    up, down = B.logits(tl, tk, ti)
    lines += ["", "## Target", "peak tokens: %s | consistency %.0f%%" % (
        ", ".join("%r×%d" % pc for pc in top), 100 * top[0][1] / pos.shape[0])]
    lines += ["- …%s" % w for w in B.distinct((B.window(pos[b].tolist(), int(arg[b]), before=24, after=4)
                                               for b in range(pos.shape[0])), N_WIN)]
    lines.append("contrast (target silent): …%s…" % B.dec(rec["neg"][0, 20:50].tolist()))
    lines.append("logit effect: + %s | − %s" % (up, down))
    lines += ["", "## Top %d members by contribution at the target's anchor" % N_MEMBERS]
    oldrank = {}
    if old is not None and (Path(L.EXP) / "063-case-studies" / "results_case" / "members" / ("%s.csv" % key)).exists():
        om = pd.read_csv(Path(L.EXP) / "063-case-studies" / "results_case" / "members" / ("%s.csv" % key))
        oldrank = {n: r for r, n in enumerate(om.node)}
    for r in cdf.head(N_MEMBERS).itertuples():
        t = B.own_contexts(r.l, r.k, r.i)
        tag = ""
        if old is not None:
            tag = " | old: %s" % (("rank %d" % (oldrank[r.node] + 1)) if r.node in oldrank else
                                  ("member" if r.node in old else "NEW"))
        lines.append("### %s  α %.2f | share %.1f%% | fires at anchor %.0f%% | seq max target %.2f vs contrast %.2f%s"
                     % (r.node, r.alpha, 100 * r.share, 100 * r.fire, r.pos_max, r.neg_max, tag))
        if t is None:
            lines.append("(no stored contexts)")
            continue
        a = B.acts([(r.l, r.k, r.i)], t)[:, :, 0]
        pk = a.argmax(1)
        tt = Counter(B.dec([t[b, int(pk[b])]]) for b in range(t.shape[0])).most_common(4)
        lines.append("own peaks: %s | consistency %.0f%%%s" % (
            ", ".join("%r×%d" % pc for pc in tt), 100 * tt[0][1] / t.shape[0],
            " (LEXICAL)" if tt[0][1] / t.shape[0] >= 0.8 else ""))
        lines += ["- …%s" % w for w in B.distinct((B.window(t[b].tolist(), int(pk[b]), before=16, after=3)
                                                   for b in range(t.shape[0])), 3)]
        if r.k != "attn":
            lines.append("logits +: %s" % B.logits(r.l, r.k, r.i, n=6)[0])
    rest = cdf.iloc[N_MEMBERS:]
    lines.append("\nremaining %d members carry %.0f%% of the contribution" % (len(rest), 100 * rest.share.sum()))
    return "\n".join(lines) + "\n", cdf


def main():
    keys = [k for k in os.environ.get("KEYS", "9.resid.20419").split(",") if k]
    (L.RESULTS / "reports").mkdir(parents=True, exist_ok=True)
    (L.RESULTS / "members").mkdir(parents=True, exist_ok=True)
    d, dm = L.scores("strong"), L.scores("mid")
    import validate as VAL
    Vd = VAL.Validator()
    B = make_browser(Vd.G)
    t0 = time.time()
    for n, key in enumerate(keys):
        path = L.RESULTS / "reports" / ("%s.md" % key)
        if path.exists() and not os.environ.get("FORCE"):
            continue
        try:
            txt, cdf = report(B, key, d.loc[key] if key in d.index else None, dm.loc[key] if key in dm.index else None)
        except Exception as e:  # noqa: BLE001
            print("  %s FAILED: %s: %s" % (key, type(e).__name__, e), flush=True)
            continue
        path.write_text(txt, encoding="utf-8")
        cdf.to_csv(L.RESULTS / "members" / ("%s.csv" % key), index=False)
        print("%4d/%d %-16s %.0fs" % (n + 1, len(keys), key, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

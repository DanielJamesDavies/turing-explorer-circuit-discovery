"""GREATER-THAN MECHANISM in the production circuits.

Prompts (047 gt_clusters.pt): "The {noun} lasted from the year {c}{YY} to
the year {c}" -> next digit must be > YY's tens digit. Published mechanism
(Hanna et al. 2023, GPT-2): early MLPs at the YY token compute the tens
digit, late attention heads at the final position attend to YY, late
MLPs boost digits > tens and suppress digits <= tens.

For the task-selective circuits (task_circuit_sets.json['gt']):
  1. WHERE each seed fires: mean activation per token ROLE (century
     digits, tens digit, units digit, "year", the final "19", other) and
     the fraction of prompts where it PEAKS at each role.
  2. TENS DEPENDENCE: mean seed activation (at its role position) for
     each tens digit 0-9 -> detectors (one digit) vs graded (monotone in
     tens) vs flat.
  3. OUTPUT: direct logit attribution of the seed over the ten digit
     tokens -> which digits it promotes/suppresses, and whether that is
     "above the tens" for the prompts it fires on.
  4. MEMBERS: which members fire, at which roles, with amplitudes; how
     many are digit-family latents.
  5. CAUSAL: metric drop from ablating each circuit alone (hubs excluded)
     and from ablating the whole set ONLY at the first-year positions vs
     ONLY at the final positions vs everywhere.

  PYTHONPATH=src python experiments/049-circuit-graph/gt_mechanism.py
"""
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hardware import detect_devices, is_fast_memory, should_compile
from model.hooks import multi_patch
from model.inference import Inference
from model.tokenizer import Tokenizer
from sae.bank import SAEBank
from sae.dense import sparse_topk_to_dense

HERE = Path(__file__).parent
T = Path(os.environ.get("TABLES", str(HERE / "tables_full")))
R = Path(os.environ.get("OUT", str(HERE / "results_full")))
D047 = HERE.parent / "047-known-circuits"
EVAL_BS = 16
torch.set_float32_matmul_precision("high")
devices = detect_devices(); device = devices[0]
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
tok = Tokenizer(); KINDS = list(bank.kinds); D = bank.d_sae
inference.disable_compile()

C = pd.read_parquet(T / "circuits.parquet")
M = pd.read_parquet(T / "members.parquet"); M["kind"] = M["kind"].astype(str)
seed_of = {int(r.cid): (int(r.seed_layer), str(r.seed_kind), int(r.seed_index)) for r in C.itertuples()}
skey = {c: "%d.%s.%d" % s for c, s in seed_of.items()}
cid_of = {v: k for k, v in skey.items()}
members = defaultdict(lambda: defaultdict(set)); amps = defaultdict(lambda: defaultdict(dict))
for cid, l, k, i, a in zip(M["cid"].values, M["layer"].values, M["kind"].values, M["index"].values, M["amplitude"].values):
    members[int(cid)][(int(l), k)].add(int(i)); amps[int(cid)][(int(l), k)][int(i)] = float(a) if a == a else 1.0
fam = pd.read_csv(R / "circuit_families_nohub.csv").set_index("cid")["family"]
fo = pd.read_csv(R / "fanout_latents.csv")
HUBS = defaultdict(set)
for _, r in fo[fo["fanout"] >= 0.05 * len(C)].iterrows():
    HUBS[(int(r["layer"]), str(r["kind"]))].add(int(r["index"]))
DIGIT_FAM = 24
digit_latents = set()
for c in C.loc[C["cid"].map(fam) == DIGIT_FAM, "cid"]:
    for s, v in members[int(c)].items():
        digit_latents |= {(s, i) for i in v}

# ---- data ----------------------------------------------------------------------
d = torch.load(D047 / "gt_clusters.pt", weights_only=False)
digit_ids = [int(x) for x in d["digit_ids"]]
rows = []
for i, w in enumerate(d["windows"]):
    a = int(d["anchors"][i]); ids = [int(x) for x in w[:a + 1]]
    tens = int(d["tens"][i]); target = int(d["targets"][i])
    words = [tok.decode([t]) for t in ids]
    dpos = [p for p, t in enumerate(ids) if t in digit_ids]
    if len(dpos) < 6:
        continue
    roles = ["other"] * len(ids)
    roles[dpos[0]] = "c1"; roles[dpos[1]] = "c2"; roles[dpos[2]] = "tens"; roles[dpos[3]] = "units"
    roles[dpos[-2]] = "f1"; roles[dpos[-1]] = "f2"
    for p, wd in enumerate(words):
        if wd.strip() == "year":
            roles[p] = "year1" if p < dpos[2] else "year2"
    rows.append(dict(ids=ids, roles=roles, tens=tens, target=target, contrast=digit_ids[tens], prompt=d["prompts"][i]))
print("gt prompts: %d | tens digit histogram %s" % (len(rows), dict(sorted(Counter(r["tens"] for r in rows).items()))))
ROLES = ["c1", "c2", "tens", "units", "year2", "f1", "f2", "other"]

sel = json.load(open(R / "task_circuit_sets.json"))["gt"]["circuits"]
cids = [cid_of[s] for s in sel]
print("selective circuits: %s" % ", ".join("%s(f%s)" % (s, fam.get(cid_of[s], "-")) for s in sel))


def batches(rs):
    for s in range(0, len(rs), EVAL_BS):
        b = rs[s:s + EVAL_BS]
        L = max(len(r["ids"]) for r in b)
        tk = torch.tensor([r["ids"] + [0] * (L - len(r["ids"])) for r in b], dtype=torch.long, device=device)
        yield b, tk


class CaptureAll:
    """sparse (row, pos, site, idx, act) for every site."""

    def __init__(self, off):
        self.off = off; self.recs = []

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        ta, ti = bank.encode(x, kind, layer_idx)
        a = ta.float().cpu().numpy(); i = ti.cpu().numpy()
        B, Tn, Kk = a.shape
        rr = np.repeat(np.arange(B), Tn * Kk); pp = np.tile(np.repeat(np.arange(Tn), Kk), B)
        keep = a.reshape(-1) > 0
        self.recs.append(pd.DataFrame({"row": rr[keep] + self.off, "pos": pp[keep], "layer": layer_idx, "kind": kind,
                                       "index": i.reshape(-1)[keep].astype(np.int64), "act": a.reshape(-1)[keep]}))
        return x


recs = []; off = 0
with torch.no_grad():
    for b, tk in batches(rows):
        cap = CaptureAll(off); inference.forward(tk, patcher=cap, grad_enabled=False, return_activations=False, tokenize_final=False)
        recs.extend(cap.recs); off += len(b)
A = pd.concat(recs, ignore_index=True)
role_of = {(ri, p): ro for ri, r in enumerate(rows) for p, ro in enumerate(r["roles"])}
A["role"] = [role_of.get((ri, p), "pad") for ri, p in zip(A["row"], A["pos"])]
A = A[A["role"] != "pad"]
tens_of = np.array([r["tens"] for r in rows])
print("captured %d active (row,pos,latent) triples" % len(A))

# ---- DLA over digits ------------------------------------------------------------------
W_U = inference.model.lm_head.weight.detach().float() * inference.model.transformer.norm_f.scale.detach().float()[None, :]
dig = torch.tensor(digit_ids, device=W_U.device)


def dla_digits(l, k, i):
    dvec = bank.saes[k][l].decoder.weight[:, i].detach().float().to(W_U.device)
    return (W_U[dig] @ dvec).cpu().numpy()


# ---- per-circuit analysis ---------------------------------------------------------------
print("\n" + "=" * 100)
out = []
for c in cids:
    l, k, i = seed_of[c]
    s = A[(A["layer"] == l) & (A["kind"] == k) & (A["index"] == i)]
    n = len(rows)
    by_role = s.groupby("role")["act"].sum() / n           # mean act per prompt at each role
    peak_role = s.sort_values("act", ascending=False).drop_duplicates("row")["role"].value_counts()
    main_role = peak_role.index[0] if len(peak_role) else "-"
    # tens dependence at the main role
    sr = s[s["role"] == main_role].groupby("row")["act"].max()
    act_by_tens = np.array([sr.reindex(np.where(tens_of == t)[0]).fillna(0).mean() if (tens_of == t).any() else np.nan for t in range(10)])
    fire_by_tens = np.array([(sr.reindex(np.where(tens_of == t)[0]).fillna(0) > 0).mean() if (tens_of == t).any() else np.nan for t in range(10)])
    v = np.nan_to_num(act_by_tens)
    sel_idx = (v.max() / max(v.mean(), 1e-9)) if v.max() > 0 else 0.0
    slope = np.corrcoef(np.arange(10)[~np.isnan(act_by_tens)], act_by_tens[~np.isnan(act_by_tens)])[0, 1] if np.isfinite(act_by_tens).sum() > 2 and v.max() > 0 else 0.0
    kind_t = "DETECTOR(%d)" % int(v.argmax()) if sel_idx > 3 else ("GRADED(%+.2f)" % slope if abs(slope) > 0.6 else "flat")
    dla = dla_digits(l, k, i)
    dla_dir = np.corrcoef(np.arange(10), dla)[0, 1]
    # members
    mem = members[c]; n_mem = sum(len(v_) for v_ in mem.values())
    n_dig = sum(1 for s_, v_ in mem.items() for j in v_ if (s_, j) in digit_latents)
    mm = A.merge(pd.DataFrame([(s_[0], s_[1], j, amps[c][s_].get(j, 1.0)) for s_, v_ in mem.items() for j in v_], columns=["layer", "kind", "index", "amp"]),
                 on=["layer", "kind", "index"])
    fired = mm.groupby(["layer", "kind", "index"]).size()
    mem_role = mm.sort_values("act", ascending=False).drop_duplicates(["row", "layer", "kind", "index"]).groupby("role").size()
    top_mem = (mm.assign(sc=mm["act"] * mm["amp"]).groupby(["layer", "kind", "index", "amp"])["sc"].mean().sort_values(ascending=False).head(8))
    print("\n%s | family %s | %d members (%d digit-family) | fires on %d/%d prompts | PEAKS at %s"
          % (skey[c], fam.get(c, "-"), n_mem, n_dig, s["row"].nunique(), n, dict(peak_role.head(3))))
    print("  mean act by role : " + " ".join("%s %.2f" % (ro, by_role.get(ro, 0.0)) for ro in ROLES))
    print("  act by tens digit: " + " ".join("%d:%.1f" % (t, x) for t, x in enumerate(act_by_tens)) + "  -> %s" % kind_t)
    print("  fires by tens    : " + " ".join("%d:%.0f%%" % (t, 100 * x) for t, x in enumerate(fire_by_tens)))
    print("  DLA over digits  : " + " ".join("%d:%+.2f" % (t, x) for t, x in enumerate(dla)) + "  (corr with digit value %+.2f)" % dla_dir)
    print("  members firing by role: %s | %d of %d members fire somewhere" % (dict(mem_role), len(fired), n_mem))
    print("  top members (act x amp): " + "; ".join("%d.%s.%d@%.1f" % (lk[0], lk[1], lk[2], lk[3]) for lk in top_mem.index))
    out.append(dict(seed=skey[c], family=int(fam.get(c, -1)), n_members=n_mem, n_digit_family=n_dig, peak_role=main_role,
                    act_by_role={ro: float(by_role.get(ro, 0.0)) for ro in ROLES}, act_by_tens=[float(x) for x in act_by_tens],
                    tens_kind=kind_t, dla_digits=[float(x) for x in dla], dla_corr=float(dla_dir)))

# ---- causal ---------------------------------------------------------------------------------
class AblateSet:
    def __init__(self, sets, posmask=None):
        self.sets = {s: torch.tensor(sorted(v), dtype=torch.long) for s, v in sets.items() if v}
        self.posmask = posmask   # [B, T] bool or None

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        idx = self.sets.get((layer_idx, kind))
        if idx is None:
            return x
        ta, ti = bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, D, dtype=x.dtype)
        keep = torch.ones_like(dense); keep[..., idx.to(dense.device)] = 0
        if self.posmask is not None:
            pm = self.posmask.to(dense.device)[:, :dense.shape[1], None]
            keep = torch.where(pm, keep, torch.ones_like(keep))
        code = dense * keep
        return x + bank.decode(code - dense, kind, layer_idx, add_bias=False).to(x.dtype)


def metric(patch_fn=None):
    """mean over prompts of (logp target - logp same-tens digit, P(>tens) - P(<=tens))."""
    m1, m2 = [], []
    with torch.no_grad():
        for b, tk in batches(rows):
            p = patch_fn(b, tk) if patch_fn else None
            res = inference.forward(tk, patcher=p, all_logits=True, grad_enabled=False, return_activations=False, tokenize_final=False)
            lg = res[1] if isinstance(res, (tuple, list)) else res
            an = torch.tensor([len(r["ids"]) - 1 for r in b], device=device); rr = torch.arange(len(b), device=device)
            lp = torch.log_softmax(lg[rr, an].float(), dim=-1); pr = lp.exp()
            for j, r in enumerate(b):
                m1.append(float(lp[j, r["target"]] - lp[j, r["contrast"]]))
                pd_ = pr[j, dig]
                m2.append(float(pd_[r["tens"] + 1:].sum() - pd_[:r["tens"] + 1].sum()))
    return float(np.mean(m1)), float(np.mean(m2))


def union(cs):
    S = defaultdict(set)
    for c in cs:
        l, k, i = seed_of[c]; S[(l, k)].add(i)
        for s, v in members[c].items():
            S[s] |= v
    return {s: v - HUBS.get(s, set()) for s, v in S.items()}


def posmask_for(b, tk, want):
    pm = torch.zeros(tk.shape, dtype=torch.bool)
    for j, r in enumerate(b):
        for p, ro in enumerate(r["roles"]):
            if ro in want:
                pm[j, p] = True
    return pm


full = metric()
print("\n" + "=" * 100)
print("FULL: logp diff %.3f | P(>tens)-P(<=tens) %.3f" % full)
S_all = union(cids)
print("ablate whole set (hubs excluded), everywhere        : %.3f | %.3f" % metric(lambda b, tk: AblateSet(S_all)))
print("ablate whole set at FIRST-YEAR digits only          : %.3f | %.3f" % metric(lambda b, tk: AblateSet(S_all, posmask_for(b, tk, {"c1", "c2", "tens", "units"}))))
print("ablate whole set at the TENS digit only             : %.3f | %.3f" % metric(lambda b, tk: AblateSet(S_all, posmask_for(b, tk, {"tens"}))))
print("ablate whole set at the FINAL '19' only             : %.3f | %.3f" % metric(lambda b, tk: AblateSet(S_all, posmask_for(b, tk, {"f1", "f2"}))))
print("ablate whole set at 'year' tokens only              : %.3f | %.3f" % metric(lambda b, tk: AblateSet(S_all, posmask_for(b, tk, {"year1", "year2"}))))
print("ablate whole set everywhere EXCEPT first-year digits: %.3f | %.3f" % metric(lambda b, tk: AblateSet(S_all, ~posmask_for(b, tk, {"c1", "c2", "tens", "units"}))))
print("\nper-circuit ablation (hubs excluded), everywhere:")
per = []
for c in cids:
    m = metric(lambda b, tk: AblateSet(union([c])))
    per.append((skey[c], m))
    print("  %-16s -> logp diff %.3f (drop %.2f) | P diff %.3f (drop %.2f)" % (skey[c], m[0], full[0] - m[0], m[1], full[1] - m[1]))
json.dump(dict(full=full, circuits=out, per_circuit=per), open(R / "gt_mechanism.json", "w"), indent=1)
print("->", R / "gt_mechanism.json")

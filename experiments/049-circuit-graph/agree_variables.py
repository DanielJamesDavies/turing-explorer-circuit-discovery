"""AGREEMENT AS A COMPUTATION OVER ITS MEMBERS: subject number vs attractor.

"The key from the books" -> the verb agrees with "key" (SUBJECT, 3 tokens
back) and must IGNORE "books" (ATTRACTOR, adjacent to the prediction).
Two competing variables, one of which must win: the cleanest "multiple
variables figuring something out" shape in the task set.

The agreement set (task_circuit_sets.json['agree'], 9 circuits) is read
as a computation:

  1. ROLE-ASSIGN EVERY MEMBER from its activations on the 256 prompts:
       SUBJ  number-dependent at the SUBJECT position (|d| between the
             singular-subject and plural-subject means, normalised)
       ATTR  number-dependent at the ATTRACTOR position
       TRANS number-dependent at the FINAL position (where the decision
             is read) although the subject is 3-4 tokens back
       OUT   direct logit effect on (plural verb - singular verb)
  2. THE NUMBER SWAP: receiver prompts and donor prompts with OPPOSITE
     subject number. Install the donor's states at the receiver's
     SUBJECT position (subject readers), and separately at the
     ATTRACTOR position (attractor readers), and read
        margin = logp(correct verb) - logp(wrong verb)
        score  = (margin_swap - margin_recv) / (margin_donor - margin_recv)
     1 = the receiver now agrees with the DONOR's number. A circuit that
     computes agreement from the subject should move under the subject
     swap and NOT under the attractor swap (the attractor is what it is
     built to ignore) — and the size of the attractor effect predicts
     the model's own agreement-attraction errors.
     Controls: size/site-matched random latents given the same values.

  PYTHONPATH=src python experiments/049-circuit-graph/agree_variables.py
Env: N_PAIRS (60)
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
N_PAIRS = int(os.environ.get("N_PAIRS", 60))
EVAL_BS = 16
rng = np.random.default_rng(0)
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
fo = pd.read_csv(R / "fanout_latents.csv")
HUBS = defaultdict(set)
for _, r in fo[fo["fanout"] >= 0.05 * len(C)].iterrows():
    HUBS[(int(r["layer"]), str(r["kind"]))].add(int(r["index"]))

# ---- data -----------------------------------------------------------------------------------
d = torch.load(D047 / "agree_clusters.pt", weights_only=False)
PREPS = {"from", "beside", "behind", "near", "under", "above", "by", "with", "in", "on", "at"}
rows = []
for i, w in enumerate(d["windows"]):
    a = int(d["anchors"][i]); ids = [int(x) for x in w[:a + 1]]
    words = [tok.decode([t]) for t in ids]
    # "The <SUBJ> <prep> the <ATTR>" — subject = token after the first 'The', attractor = final token
    prep = [p for p, wd in enumerate(words) if wd.strip().lower() in PREPS]
    if not prep:
        continue
    p_prep = prep[-1]
    subj = p_prep - 1
    attr = len(ids) - 1
    if subj < 1 or attr <= p_prep:
        continue
    rows.append(dict(ids=ids, subj=subj, attr=attr, prep=p_prep, final=attr,
                     plural=bool(d["plural"][i]), target=int(d["targets"][i]),
                     wrong=int(d["wrong_tokens"][i]), prompt=d["prompts"][i],
                     subj_word=words[subj], attr_word=words[attr]))
pl = np.array([r["plural"] for r in rows])
print("agreement prompts: %d | plural subjects %d singular %d" % (len(rows), pl.sum(), (~pl).sum()))
print("  e.g. %r  subject %r attractor %r  -> %r (not %r)"
      % (rows[0]["prompt"], rows[0]["subj_word"], rows[0]["attr_word"],
         tok.decode([rows[0]["target"]]), tok.decode([rows[0]["wrong"]])))
# does the attractor oppose the subject? (the interesting prompts)
print("  attractor-number opposition is what makes the task non-trivial; roles: subj@%d attr@final"
      % (rows[0]["subj"] - rows[0]["final"]))

sel = json.load(open(R / "task_circuit_sets.json"))["agree"]["circuits"]
cids = [cid_of[s] for s in sel]
U = defaultdict(set)
for c in cids:
    l, k, i = seed_of[c]; U[(l, k)].add(i)
    for s, v in members[c].items():
        U[s] |= v
U = {s: v - HUBS.get(s, set()) for s, v in U.items()}
tmp = defaultdict(list)
for c in cids:
    for s, dd in amps[c].items():
        for i, a in dd.items():
            tmp[(s, i)].append(a)
AMP = defaultdict(dict)
for (s, i), v in tmp.items():
    AMP[s][i] = float(np.median(v))
print("selective circuits: %s" % ", ".join(sel))
print("member universe: %d latents (hubs excluded) over %d sites" % (sum(len(v) for v in U.values()), len(U)))


def batches(rs):
    for s in range(0, len(rs), EVAL_BS):
        b = rs[s:s + EVAL_BS]
        L = max(len(r["ids"]) for r in b)
        tk = torch.tensor([r["ids"] + [0] * (L - len(r["ids"])) for r in b], dtype=torch.long, device=device)
        yield b, tk


class CaptureU:
    def __init__(self):
        self.idx = {s: torch.tensor(sorted(v), dtype=torch.long) for s, v in U.items() if v}
        self.out = {}

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        idx = self.idx.get((layer_idx, kind))
        if idx is not None:
            ta, ti = bank.encode(x, kind, layer_idx)
            dense = sparse_topk_to_dense(ta, ti, D, dtype=x.dtype)
            self.out[(layer_idx, kind)] = dense[..., idx.to(dense.device)].float().cpu().numpy()
        return x


SITES = sorted(U); COL = {}
for s in SITES:
    for i in sorted(U[s]):
        COL[(s, i)] = len(COL)
LAT = [None] * len(COL)
for (s, i), j in COL.items():
    LAT[j] = (s, i)
ROLES = ["subj", "prep", "final"]
ACT = {ro: np.zeros((len(rows), len(COL)), dtype=np.float32) for ro in ROLES}
with torch.no_grad():
    off = 0
    for b, tk in batches(rows):
        cap = CaptureU(); inference.forward(tk, patcher=cap, grad_enabled=False, return_activations=False, tokenize_final=False)
        for s, a in cap.out.items():
            cols = [COL[(s, i)] for i in sorted(U[s])]
            for j, r in enumerate(b):
                for ro in ROLES:
                    ACT[ro][off + j, cols] = a[j, r[ro]]
        off += len(b)

# ---- 1. roles ---------------------------------------------------------------------------------
W_U = inference.model.lm_head.weight.detach().float() * inference.model.transformer.norm_f.scale.detach().float()[None, :]
tg = torch.tensor([r["target"] for r in rows]); wr = torch.tensor([r["wrong"] for r in rows])
plural_verb = torch.tensor(sorted({int(t) for t, p in zip(tg, pl) if p} | {int(w) for w, p in zip(wr, pl) if not p}))
sing_verb = torch.tensor(sorted({int(t) for t, p in zip(tg, pl) if not p} | {int(w) for w, p in zip(wr, pl) if p}))
print("  plural verbs %s | singular verbs %s"
      % (" ".join(repr(tok.decode([int(t)])) for t in plural_verb[:6]),
         " ".join(repr(tok.decode([int(t)])) for t in sing_verb[:6])))


def dla_number(j):
    (s, i) = LAT[j]
    dvec = bank.saes[s[1]][s[0]].decoder.weight[:, i].detach().float().to(W_U.device)
    lg = W_U @ dvec
    return float(lg[plural_verb.to(lg.device)].mean() - lg[sing_verb.to(lg.device)].mean())


DLA = np.array([dla_number(j) for j in range(len(COL))])


def number_sel(A):
    """per latent: mean act on plural-subject vs singular-subject prompts, and a
    normalised selectivity d = (mp - ms) / (mp + ms)."""
    mp, ms = A[pl].mean(0), A[~pl].mean(0)
    den = np.maximum(mp + ms, 1e-6)
    return mp, ms, (mp - ms) / den


sel_subj = number_sel(ACT["subj"]); sel_attr = number_sel(ACT["final"]); sel_fin = number_sel(ACT["final"])
fire = {ro: (ACT[ro] > 0).mean(0) for ro in ROLES}
is_subj = (np.abs(sel_subj[2]) >= 0.3) & (fire["subj"] >= 0.3)
is_attr = (np.abs(sel_attr[2]) >= 0.3) & (fire["final"] >= 0.3)
is_out = np.abs(DLA) >= 0.2
print("\n[1] ROLES over %d member latents: SUBJECT-number readers %d | ATTRACTOR/final-number readers %d | OUTPUT (verb-number DLA) %d"
      % (len(COL), is_subj.sum(), is_attr.sum(), is_out.sum()))


def name(j):
    (l, k), i = LAT[j]
    return "%d.%s.%d" % (l, k, i)


def amp(j):
    s, i = LAT[j]
    return AMP[s].get(i, 1.0)


def show(title, mask, ro, seln, top=12):
    mp, ms, dd = seln
    js = sorted(np.where(mask)[0], key=lambda j: -abs(dd[j]) * ACT[ro][:, j].mean() * amp(j))[:top]
    print("\n  %s (%d; top %d):" % (title, mask.sum(), len(js)))
    print("  %-14s %5s %9s %9s %7s %8s" % ("member", "gain", "plural", "singular", "sel", "verb DLA"))
    for j in js:
        print("  %-14s %5.2f %9.2f %9.2f %+7.2f %+8.2f   -> %s"
              % (name(j), amp(j), mp[j], ms[j], dd[j], DLA[j], "PLURAL" if dd[j] > 0 else "SINGULAR"))


show("SUBJECT-number readers (at the subject noun)", is_subj, "subj", sel_subj)
show("FINAL-position number readers (where the verb is decided)", is_attr, "final", sel_fin)
js = sorted(np.where(is_out)[0], key=lambda j: -abs(DLA[j]) * ACT["final"][:, j].mean() * amp(j))[:10]
print("\n  OUTPUT members (%d): direct effect on (plural verb - singular verb)" % is_out.sum())
for j in js:
    print("  %-14s gain %.2f act@final %.2f | DLA %+.2f -> promotes %s"
          % (name(j), amp(j), ACT["final"][:, j].mean(), DLA[j], "PLURAL" if DLA[j] > 0 else "SINGULAR"))

# ---- 2. the swap -------------------------------------------------------------------------------
class Inject:
    def __init__(self, spec):
        self.spec = spec

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        s = (layer_idx, kind)
        if not any(s in sp for sp in self.spec):
            return x
        ta, ti = bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, D, dtype=x.dtype)
        code = dense.clone()
        for b, sp in enumerate(self.spec):
            for p, (idx, vals) in sp.get(s, {}).items():
                if p < code.shape[1]:
                    code[b, p, idx.to(code.device)] = vals.to(code.device, code.dtype)
        return x + bank.decode(code - dense, kind, layer_idx, add_bias=False).to(x.dtype)


def margins(rs, patch=None):
    out = []
    with torch.no_grad():
        for s0 in range(0, len(rs), EVAL_BS):
            b = rs[s0:s0 + EVAL_BS]
            L = max(len(r["ids"]) for r in b)
            tk = torch.tensor([r["ids"] + [0] * (L - len(r["ids"])) for r in b], dtype=torch.long, device=device)
            p = Inject(patch[s0:s0 + EVAL_BS]) if patch else None
            res = inference.forward(tk, patcher=p, all_logits=True, grad_enabled=False,
                                    return_activations=False, tokenize_final=False)
            lg = res[1] if isinstance(res, (tuple, list)) else res
            an = torch.tensor([len(r["ids"]) - 1 for r in b], device=device); rr = torch.arange(len(b), device=device)
            lp = torch.log_softmax(lg[rr, an].float(), -1)
            for j, r in enumerate(b):
                out.append(float(lp[j, r["target"]] - lp[j, r["wrong"]]))
    return np.array(out)


# pairs: receiver and donor with OPPOSITE subject number
idx_pl = [i for i, r in enumerate(rows) if r["plural"]]
idx_sg = [i for i, r in enumerate(rows) if not r["plural"]]
pairs = []
for _ in range(N_PAIRS):
    if len(pairs) % 2 == 0 and idx_sg and idx_pl:
        a_, b_ = int(rng.choice(idx_sg)), int(rng.choice(idx_pl))
    else:
        a_, b_ = int(rng.choice(idx_pl)), int(rng.choice(idx_sg))
    pairs.append((a_, b_))
recv = [rows[a_] for a_, _ in pairs]; donr = [rows[b_] for _, b_ in pairs]
m_r = margins(recv)
# the donor's margin measured with the donor's OWN number but on the receiver's frame is not
# available; use the donor prompt's margin sign-flipped onto the receiver's targets: a full
# flip means the receiver now prefers the donor's verb number, i.e. margin -> -m_r_expected.
# score = how far the receiver's margin moves toward its own negation (the donor's number).
print("\n[2] NUMBER SWAP: %d receiver/donor pairs with OPPOSITE subject number" % len(pairs))
print("    receiver margin logp(correct)-logp(wrong) = %.2f | full flip = %.2f" % (m_r.mean(), -m_r.mean()))

subj_cols = list(np.where(is_subj)[0])
attr_cols = list(np.where(is_attr)[0])
all_cols = list(range(len(COL)))
fire_subj = [j for j in all_cols if fire["subj"][j] > 0]
fire_fin = [j for j in all_cols if fire["final"][j] > 0]


def spec_for(cols, role, mode="donor"):
    """mode: donor = install the donor's values | rand = donor's values into
    site-matched RANDOM latents | zero = set these latents to 0 (harness check)."""
    sp = []
    picks = None
    if mode == "rand":
        picks = []
        for j in cols:
            s, _ = LAT[j]
            while True:
                i = int(rng.integers(D))
                if i not in U[s] and i not in HUBS.get(s, set()):
                    break
            picks.append((s, i))
    for n_, (a_, b_) in enumerate(pairs):
        by = defaultdict(lambda: ([], []))
        for m_, j in enumerate(cols):
            s, i = LAT[j] if mode != "rand" else picks[m_]
            by[s][0].append(i)
            by[s][1].append(0.0 if mode == "zero" else float(ACT[role][b_, j]))
        dd = {}
        for s, (ii, vv) in by.items():
            dd[s] = {rows[a_][role]: (torch.tensor(ii, dtype=torch.long), torch.tensor(vv))}
        sp.append(dd)
    return sp


conds = [
    ("HARNESS CHECK zero all @ final", fire_fin, "final", "zero"),
    ("HARNESS CHECK zero all @ subject", fire_subj, "subj", "zero"),
    ("SUBJECT readers @ subject pos", subj_cols, "subj", "donor"),
    ("all members @ subject pos", fire_subj, "subj", "donor"),
    ("FINAL-number readers @ final pos", attr_cols, "final", "donor"),
    ("all members @ final pos", fire_fin, "final", "donor"),
    ("random matched @ subject pos", subj_cols, "subj", "rand"),
    ("random matched @ final pos", attr_cols, "final", "rand"),
]
res = {}
print("\n  %-34s %6s %9s %8s %9s" % ("installed from donor", "n lat", "margin", "score", "flipped%"))
print("  %-34s %6s %9.2f %8.2f %8.0f%%" % ("(receiver, unpatched)", "-", m_r.mean(), 0.0, 100 * (m_r < 0).mean()))
for title, cols, role, rand in conds:
    if not cols:
        continue
    m = margins(recv, spec_for(cols, role, rand))
    score = (m - m_r) / np.where(np.abs(-2 * m_r) < 1e-6, np.nan, -2 * m_r)
    res[title] = dict(n_lat=len(cols), margin=float(m.mean()), score=float(np.nanmedian(score)),
                      flipped=float((m < 0).mean()))
    print("  %-34s %6d %9.2f %8.2f %8.0f%%" % (title, len(cols), m.mean(), np.nanmedian(score), 100 * (m < 0).mean()))
json.dump(dict(n_prompts=len(rows), n_universe=len(COL), n_pairs=len(pairs),
               roles=dict(subj=[name(j) for j in np.where(is_subj)[0]],
                          final=[name(j) for j in np.where(is_attr)[0]],
                          out=[name(j) for j in np.where(is_out)[0]]),
               recv_margin=float(m_r.mean()), swap=res),
          open(R / "agree_variables.json", "w"), indent=1)
print("\n->", R / "agree_variables.json")

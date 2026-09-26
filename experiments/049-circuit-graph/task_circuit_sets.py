"""TASK -> SET OF PRODUCTION CIRCUITS. For each capability the model has,
find which of the 15k tri-amp circuits fire at the prediction position on
the task's prompts and NOT on the other tasks' prompts, then test the set
causally on the task metric logp(target) - logp(contrast):

  ablate      zero the set's members (+ seeds) in the live stream
  ablate_nohub  same with the universal hubs excluded
  ablate_rand   random live latents matched per site (null)
  ablate_anti   the K circuits most selective for the OTHER tasks (null)
  only_amp    circuit-only: everything else zero-filled, the set's members
              at their fitted amplitudes (sufficiency, EF vs full/empty)
  only_a1     same with amplitudes = 1

Tasks (competence-filtered: keep prompts where logit(target) > logit(contrast)):
  gt      greater-than (047 gt_clusters.pt); contrast = the same tens digit
  agree   subject-verb agreement with PP distractor (047 agree_clusters.pt)
  year    "{who} published {what} in the year" -> digit-prefix token vs " of"
  list    "{a}, {b}, {c}, {d}, {e}," -> " and" vs " or"
  code    "def f(n):\\n    if n < 2" -> ":" vs ")"
  defn    "<definition> is called" -> first token of the term vs " the"

  K=15 PYTHONPATH=src python experiments/049-circuit-graph/task_circuit_sets.py
"""
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eval.ablation_faithfulness import CircuitOnlyPatcher
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
K = int(os.environ.get("K", 15))
MIN_RATE = float(os.environ.get("MIN_RATE", 0.5))
N_GEN = int(os.environ.get("N_GEN", 120))
EVAL_BS, SEQ = 16, 64
rng = random.Random(0); np.random.seed(0)

torch.set_float32_matmul_precision("high")
devices = detect_devices(); device = devices[0]
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
tok = Tokenizer()
KINDS = list(bank.kinds); D = bank.d_sae
ALL_SITES = sorted((l, k) for l in range(bank.n_layer) for k in KINDS)
inference.disable_compile()

# ---- production circuits ----------------------------------------------------
C = pd.read_parquet(T / "circuits.parquet")
M = pd.read_parquet(T / "members.parquet")
M["kind"] = M["kind"].astype(str)
seed_of = {int(r.cid): (int(r.seed_layer), str(r.seed_kind), int(r.seed_index)) for r in C.itertuples()}
skey = {c: "%d.%s.%d" % s for c, s in seed_of.items()}
members = defaultdict(lambda: defaultdict(set)); amps = defaultdict(lambda: defaultdict(dict))
for cid, l, k, i, a in zip(M["cid"].values, M["layer"].values, M["kind"].values, M["index"].values, M["amplitude"].values):
    members[int(cid)][(int(l), k)].add(int(i)); amps[int(cid)][(int(l), k)][int(i)] = float(a) if a == a else 1.0
fam = pd.read_csv(R / "circuit_families_nohub.csv").set_index("cid")["family"] if (R / "circuit_families_nohub.csv").exists() else None
fo = pd.read_csv(R / "fanout_latents.csv")
HUBS = defaultdict(set)
for _, r in fo[fo["fanout"] >= 0.05 * len(C)].iterrows():
    HUBS[(int(r["layer"]), str(r["kind"]))].add(int(r["index"]))
live_pool = {s: np.array(sorted(set(g["index"].astype(int)))) for s, g in M.groupby(["layer", "kind"], observed=True)}
# seed lookup: site -> {idx: [cids]}
seed_lookup = defaultdict(lambda: defaultdict(list))
for c, (l, k, i) in seed_of.items():
    seed_lookup[(l, k)][i].append(c)
print("circuits %d | members %d | hubs %d" % (len(C), len(M), sum(len(v) for v in HUBS.values())), flush=True)


# ---- task datasets -------------------------------------------------------------
def tok_after(prefix, word):
    a, b = tok.encode(prefix), tok.encode(prefix + word)
    return b[len(a)] if (b[:len(a)] == a and len(b) == len(a) + 1) else None


def first_of(prefix, word):
    a, b = tok.encode(prefix), tok.encode(prefix + word)
    return b[len(a)] if b[:len(a)] == a and len(b) > len(a) else None


def rows_from_prompts(prompts, target_fn, contrast_fn):
    rows = []
    for p in prompts:
        ids = tok.encode(p)
        if len(ids) > SEQ:
            continue
        t, c = target_fn(p), contrast_fn(p)
        if t is None or c is None or t == c:
            continue
        rows.append(dict(prompt=p, ids=ids, target=int(t), contrast=int(c)))
    return rows


def rows_from_047(name, contrast_of):
    d = torch.load(D047 / name, weights_only=False)
    rows = []
    for i, w in enumerate(d["windows"]):
        a = int(d["anchors"][i]); ids = [int(x) for x in w[:a + 1]]
        t = int(d["targets"][i]); c = contrast_of(d, i)
        if c is None or c == t:
            continue
        rows.append(dict(prompt=d["prompts"][i], ids=ids, target=t, contrast=int(c)))
    return rows


DIGIT_PREFIX = None
_ids = tok.encode("in the year 1905")
for j in range(1, len(_ids)):
    if tok.decode([_ids[j]]) == "1":
        DIGIT_PREFIX = _ids[j - 1]
        break
print("digit-prefix token id:", DIGIT_PREFIX, repr(tok.decode([DIGIT_PREFIX])) if DIGIT_PREFIX is not None else None)

people = ["Albert Einstein", "Isaac Newton", "Charles Darwin", "Marie Curie", "Galileo Galilei", "Nikola Tesla", "Ada Lovelace",
          "Alan Turing", "Gregor Mendel", "James Clerk Maxwell", "Niels Bohr", "Rosalind Franklin", "Louis Pasteur", "Johannes Kepler",
          "Max Planck", "Erwin Schrödinger", "Dmitri Mendeleev", "Michael Faraday", "Carl Linnaeus", "Antoine Lavoisier"]
works = ["his theory of special relativity", "the laws of motion", "a paper on natural selection", "her work on radioactivity",
         "the first telescope observations", "a design for the induction motor", "the first computer program", "a paper on computable numbers",
         "the laws of inheritance", "the equations of electromagnetism", "a model of the atom", "the structure of DNA", "a germ theory of disease",
         "the laws of planetary motion", "the quantum hypothesis", "the wave equation", "the periodic table", "the law of induction",
         "a system of classification", "the law of conservation of mass"]
year_prompts = ["%s published %s in the year" % (p, w) for p in people for w in works]
rng.shuffle(year_prompts); year_prompts = year_prompts[:N_GEN]

lists = {"colors": ["red", "blue", "green", "yellow", "orange", "purple", "black", "white", "brown", "pink"],
         "fruits": ["apples", "pears", "grapes", "plums", "cherries", "peaches", "lemons", "limes", "figs", "dates"],
         "animals": ["cats", "dogs", "horses", "cows", "sheep", "goats", "pigs", "ducks", "geese", "hens"],
         "tools": ["hammers", "saws", "drills", "nails", "screws", "wrenches", "pliers", "chisels", "files", "clamps"],
         "subjects": ["maths", "physics", "history", "chemistry", "biology", "art", "music", "geography", "law", "medicine"]}
list_prompts = []
for _ in range(N_GEN):
    ws = rng.sample(lists[rng.choice(list(lists))], 5)
    list_prompts.append("%s, %s, %s, %s, %s," % tuple(ws))

fns = ["fib", "count", "total", "solve", "check", "score", "parse", "build", "merge", "search"]
vars_ = ["n", "x", "k", "m", "i", "value", "size", "depth"]
ops = ["<", ">", "==", "<=", ">=", "!="]
code_prompts = []
for _ in range(N_GEN):
    v = rng.choice(vars_)
    code_prompts.append("def %s(%s):\n    if %s %s %d" % (rng.choice(fns), v, v, rng.choice(ops), rng.randint(0, 20)))

defs = [("The process by which plants convert sunlight into energy is called", " photosynthesis"),
        ("The process by which a cell divides into two identical cells is called", " mitosis"),
        ("The force that attracts objects toward the centre of the Earth is called", " gravity"),
        ("The study of living organisms is called", " biology"),
        ("The study of the past through written records is called", " history"),
        ("The branch of mathematics that deals with shapes and space is called", " geometry"),
        ("The smallest unit of an element that retains its properties is called", " an atom"),
        ("The process by which water changes from liquid to gas is called", " evaporation"),
        ("A word that describes a noun is called", " an adjective"),
        ("The system of government in which citizens vote for representatives is called", " democracy"),
        ("The process by which rocks are broken down by wind and water is called", " weathering"),
        ("The molecule that carries genetic information in cells is called", " DNA"),
        ("The layer of gases surrounding the Earth is called", " the atmosphere"),
        ("The study of the stars and planets is called", " astronomy"),
        ("A shape with three sides is called", " a triangle"),
        ("The change of a substance from solid directly to gas is called", " sublimation"),
        ("The organ that pumps blood around the body is called", " the heart"),
        ("The process by which animals produce offspring is called", " reproduction"),
        ("The imaginary line around the middle of the Earth is called", " the equator"),
        ("A word that expresses an action is called", " a verb"),
        ("The centre of an atom is called", " the nucleus"),
        ("The process of a liquid turning into a solid is called", " freezing"),
        ("The study of the human mind and behaviour is called", " psychology"),
        ("The gas that plants absorb from the air is called", " carbon dioxide"),
        ("The measure of how much matter is in an object is called", " mass"),
        ("A number that can only be divided by one and itself is called", " a prime"),
        ("The largest planet in the solar system is called", " Jupiter"),
        ("The process by which the body breaks down food is called", " digestion"),
        ("A person who studies rocks is called", " a geologist"),
        ("The speed of an object in a given direction is called", " velocity")]

TASKS = {
    "gt": rows_from_047("gt_clusters.pt", lambda d, i: int(d["digit_ids"][int(d["tens"][i])])),
    "agree": rows_from_047("agree_clusters.pt", lambda d, i: int(d["wrong_tokens"][i])),
    "year": rows_from_prompts(year_prompts, lambda p: DIGIT_PREFIX, lambda p: tok_after(p, " of")),
    "list": rows_from_prompts(list_prompts, lambda p: tok_after(p, " and"), lambda p: tok_after(p, " or")),
    "code": rows_from_prompts(code_prompts, lambda p: tok_after(p, ":"), lambda p: tok_after(p, ")")),
    "defn": rows_from_prompts([d[0] for d in defs], lambda p: first_of(p, dict(defs)[p]), lambda p: tok_after(p, " the")),
}


def batches(rows):
    for s in range(0, len(rows), EVAL_BS):
        b = rows[s:s + EVAL_BS]
        L = max(len(r["ids"]) for r in b)
        tk = torch.tensor([r["ids"] + [0] * (L - len(r["ids"])) for r in b], dtype=torch.long, device=device)
        an = torch.tensor([len(r["ids"]) - 1 for r in b], dtype=torch.long, device=device)
        yield b, tk, an


def metric_rows(rows, patcher=None):
    """per-row logp(target) - logp(contrast) at the anchor."""
    out = []
    with torch.no_grad():
        for b, tk, an in batches(rows):
            res = inference.forward(tk, patcher=patcher, all_logits=True, grad_enabled=False, return_activations=False, tokenize_final=False)
            lg = res[1] if isinstance(res, (tuple, list)) else res
            rr = torch.arange(tk.shape[0], device=device)
            lp = torch.log_softmax(lg[rr, an].float(), dim=-1)
            t = torch.tensor([r["target"] for r in b], device=device); c = torch.tensor([r["contrast"] for r in b], device=device)
            out.extend((lp[rr, t] - lp[rr, c]).tolist())
    return np.array(out)


# competence filter
for name in list(TASKS):
    rows = TASKS[name]
    if not rows:
        print("task %-6s: NO ROWS (tokenisation)" % name); del TASKS[name]; continue
    m = metric_rows(rows)
    keep = [r for r, v in zip(rows, m) if v > 0]
    print("task %-6s: %3d prompts, %3d competent (%.0f%%), margin on competent %.2f | e.g. %r -> %r vs %r"
          % (name, len(rows), len(keep), 100 * len(keep) / len(rows), float(np.mean(m[m > 0])) if (m > 0).any() else float("nan"),
             rows[0]["prompt"][-60:], tok.decode([rows[0]["target"]]), tok.decode([rows[0]["contrast"]])), flush=True)
    if len(keep) < 12:
        print("  -> too few competent prompts, dropping %s" % name); del TASKS[name]; continue
    TASKS[name] = keep


# ---- seed firing at the prediction position -------------------------------------
class AnchorCapture:
    def __init__(self, anchors):
        self.an = anchors; self.hits = defaultdict(set)   # (site, idx) -> {row}
        self.acts = defaultdict(dict)

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        ta, ti = bank.encode(x, kind, layer_idx)
        rr = torch.arange(x.shape[0], device=x.device)
        a = ta[rr, self.an].float().cpu().numpy(); i = ti[rr, self.an].cpu().numpy()
        for b in range(x.shape[0]):
            for v, j in zip(a[b], i[b]):
                if v > 0:
                    self.hits[((layer_idx, kind), int(j))].add(b)
                    self.acts[((layer_idx, kind), int(j))][b] = float(v)
        return x


def seed_fire_matrix(rows):
    """returns fired [n_rows, n_circuits] bool via seed lookup."""
    n = len(rows); fired = {}
    off = 0
    with torch.no_grad():
        for b, tk, an in batches(rows):
            cap = AnchorCapture(an)
            inference.forward(tk, patcher=cap, grad_enabled=False, return_activations=False, tokenize_final=False)
            for (site, j), rowset in cap.hits.items():
                for cid in seed_lookup[site].get(j, []):
                    fired.setdefault(cid, set()).update(off + r for r in rowset)
            off += len(b)
    return fired, n


fire = {}; nrows = {}
for name, rows in TASKS.items():
    fire[name], nrows[name] = seed_fire_matrix(rows)
    print("firing pass %-6s: %d circuits fire at the prediction position on >= 1 prompt" % (name, len(fire[name])), flush=True)


def rates(name):
    n = nrows[name]; other = [t for t in TASKS if t != name]
    n_other = sum(nrows[t] for t in other)
    all_c = set(fire[name]) | set().union(*(set(fire[t]) for t in other))
    rows = []
    for c in all_c:
        rt = len(fire[name].get(c, ())) / n
        rc = sum(len(fire[t].get(c, ())) for t in other) / max(n_other, 1)
        rows.append((c, rt, rc))
    return pd.DataFrame(rows, columns=["cid", "rate_task", "rate_other"])


# ---- patchers ------------------------------------------------------------------------
class AblateSet:
    def __init__(self, sets):
        self.sets = {s: torch.tensor(sorted(v), dtype=torch.long) for s, v in sets.items() if v}

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        idx = self.sets.get((layer_idx, kind))
        if idx is None:
            return x
        ta, ti = bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, D, dtype=x.dtype)
        code = dense.clone(); code[..., idx.to(dense.device)] = 0
        return x + bank.decode(code - dense, kind, layer_idx, add_bias=False).to(x.dtype)


def union_sets(cids, with_amps=True):
    S = defaultdict(set); A = defaultdict(lambda: defaultdict(list))
    for c in cids:
        l, k, i = seed_of[c]; S[(l, k)].add(i); A[(l, k)][i].append(1.0)
        for s, v in members[c].items():
            S[s] |= v
            for i in v:
                A[s][i].append(amps[c][s].get(i, 1.0))
    scales = {s: {i: float(np.mean(v)) for i, v in d.items()} for s, d in A.items()}
    return dict(S), scales


def drop_hubs(S):
    return {s: v - HUBS.get(s, set()) for s, v in S.items()}


def matched_random(S):
    out = {}
    for s, v in S.items():
        pool = live_pool.get(s)
        if pool is None:
            continue
        cand = pool[~np.isin(pool, list(v))]
        n = min(len(v), len(cand))
        out[s] = set(np.random.choice(cand, n, replace=False).tolist()) if n else set()
    return out


def only_patcher(S, scales=None):
    ks = None
    if scales is not None:
        ks = {}
        for s, d in scales.items():
            t = torch.ones(D, dtype=torch.float32)
            for i, a in d.items():
                t[i] = a
            ks[s] = t
    return CircuitOnlyPatcher(bank=bank, keep_indices={s: set(v) for s, v in S.items()}, in_scope=set(ALL_SITES),
                              seed_layer=-1, seed_kind="", seed_latent_idx=0, site_means=None, keep_scales=ks)


# ---- per task ---------------------------------------------------------------------------
report = {}
t0 = time.time()
for name, rows in TASKS.items():
    df = rates(name)
    sel = df[df["rate_task"] >= MIN_RATE].assign(score=lambda d: d["rate_task"] - d["rate_other"]).sort_values("score", ascending=False)
    top = sel.head(K)
    anti = df[df["rate_other"] >= MIN_RATE].assign(score=lambda d: d["rate_other"] - d["rate_task"]).sort_values("score", ascending=False).head(K)
    print("\n===== TASK %s: %d prompts | %d circuits fire on >= %.0f%% of prompts | top-%d selective:" % (name, len(rows), len(sel), 100 * MIN_RATE, K))
    for _, r in top.iterrows():
        c = int(r["cid"])
        print("   %-16s task %.2f other %.2f | %4d members | family %s" % (skey[c], r["rate_task"], r["rate_other"], len(M[M["cid"] == c]) if False else sum(len(v) for v in members[c].values()),
                                                                          "-" if fam is None else fam.get(c, "-")))
    cids = [int(c) for c in top["cid"]]; anti_c = [int(c) for c in anti["cid"]]
    S, scales = union_sets(cids)
    n_set = sum(len(v) for v in S.values())
    m_full = metric_rows(rows)
    res = {"n_prompts": len(rows), "n_circuits": len(cids), "n_latents": n_set, "circuits": [skey[c] for c in cids],
           "full": float(m_full.mean())}
    conds = {"ablate": AblateSet(S), "ablate_nohub": AblateSet(drop_hubs(S)), "ablate_rand": AblateSet(matched_random(drop_hubs(S))),
             "ablate_anti": AblateSet(drop_hubs(union_sets(anti_c)[0])) if anti_c else None,
             "only_amp": only_patcher(S, scales), "only_a1": only_patcher(S, None), "only_empty": only_patcher({}, None)}
    for cn, p in conds.items():
        if p is None:
            res[cn] = None; continue
        res[cn] = float(metric_rows(rows, p).mean())
    fe = res["full"] - res["only_empty"]
    res["EF_amp"] = (res["only_amp"] - res["only_empty"]) / fe if abs(fe) > 1e-6 else None
    res["EF_a1"] = (res["only_a1"] - res["only_empty"]) / fe if abs(fe) > 1e-6 else None
    print("   metric logp(target)-logp(contrast): full %.3f | ablate set %.3f | nohub %.3f | rand %.3f | anti %s | circuit-only+amps %.3f | a=1 %.3f | empty %.3f | EF_amp %s EF_a1 %s | %d latents | %.0fs"
          % (res["full"], res["ablate"], res["ablate_nohub"], res["ablate_rand"], "%.3f" % res["ablate_anti"] if res["ablate_anti"] is not None else "-",
             res["only_amp"], res["only_a1"], res["only_empty"], "%.2f" % res["EF_amp"] if res["EF_amp"] is not None else "-",
             "%.2f" % res["EF_a1"] if res["EF_a1"] is not None else "-", n_set, time.time() - t0), flush=True)
    report[name] = res
json.dump(report, open(R / "task_circuit_sets.json", "w"), indent=1)
print("\n->", R / "task_circuit_sets.json")

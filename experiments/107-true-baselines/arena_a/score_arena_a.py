"""Arena A (A8, DAN-188 / DAN-197): score Ge (hierarchical + post-hoc), Dunefsky and WCM circuits on the gpt2-small-tcz
substrate under OUR tests, plus Ge's leaf-sum recovery (Ge circuits) and Dunefsky's attribution coverage. .venv, the
site model and bank of wcm_tcz.py through 062's driver (driver_tcz.py). Other papers' circuits are only read and scored.

Units (one scored node set; curves = each method's full knob sweep, never matched-size points):
  ge_hier     run_ge.py p<j>.pt, hier[t] at each tau of the grid (knob = tau / a_t)
  ge_posthoc  the same file: S(tau) = posthoc_pool rows with attr > tau (strict, their read-out rule)
  dunefsky    run_dunefsky.py p<j>.b<nb>.i<it>.pt (knob = num_branches): every 'mlp' graph row with a latent, root
              excluded, plus the root's own (layer, latent) at an earlier token excluded (as their read-out does)
  wcm         062-driver circuits main/circuits/<key>.pt (knob = lambda, from the --wcm label), fitted alphas
Node mapping [ours, settled 2026-11-05]: Ge / Dunefsky positions are BOS-prefixed (pos 0 = BOS) -> pos 0 nodes are
dropped (BOS is a frozen live prefix for WCM) and the rest shifted to pos - 1; Kissane z latents >= 24,576 at L5 / L7
map to kind z2 (latent - 24,576). Nodes at sites that are not upstream of the target (its own site, later sites) are
dropped and counted (they cannot affect the root). The per-prompt (latent, pos) set is collapsed to latent-level
membership {(layer, kind): {latent}} for our position-collapsed tests; both counts are recorded. Baseline members are
kept at alpha = 1 (their live value); WCM members at their fitted alpha (= the production scorer).

Tests, per unit and eval set ("ho" = the target's 16 held-out strongest contexts, the production eval set; "home" =
the circuit's own prompt(s): a Ge / Dunefsky circuit on its prompt j, WCM on all K prompts, each read at the prompt's
anchor; the prompt row is the record's strong row j, identical at the anchor to the truncated prompt by causality):
  Z, A, C     free0 / freeM_topk / freeN_topk (post-Top-K read, k = d_sae) with src eval.wcm_eval's own patchers and
              fills (TapCircuitOnly, collect_site_anchors / collect_site_means, eM / eN empty baselines), computed as
              score_circuit computes them; the target-level work (a_pos, means, empty baselines) is shared per target
  nec_a       phi_sup_blind: members set to 0 in the live stream, everything else live, transcoder / SAE error LIVE
              (the real MLP recomputes); (a_pos - x) / (a_pos - e0). = the production necessity (regression check)
  nec_b       the same removal with EVERY upstream site's error (transcoder error mlp(x_in) - decode(code) and z SAE
              error) held at its CLEAN value, non-members live: diag_necessity_tcz.Diag(mem="zero",
              freeze_err={"z", "tc"}), (clean - x) / (clean - e0) with e0 the production (live-error) empty circuit, as
              the diagnosis (necdiag_summary.md) defines it [settled 2026-11-05: Ge / Dunefsky / circuit-tracer convention]
  nec_b_tc    only the transcoder error frozen (the diagnosis' f_tc_err), recorded
  pass_a/b    Z, A, C in [0.75, 1.5] and nec_a / nec_b >= 0.9 (wcm_eval BAND / NEC_MIN); None when the target is
              vacuous (wcm_eval.vacuous_v2 on the eval set)
  ge recovery (Ge units, from the runner's in-pass leaf parts: hier[t].parts, the A11 re-run posthoc[t].parts):
              interp = embed / a_t (the paper's definition, PRIMARY), nonerr = (embed + biases) / a_t, total = root of
              G' / a_t (bounded companion); all three kept [settled 2026-11-05]
  coverage    (Dunefsky units) 1 - root prune_error / root (dunefsky.md 7.3(a), from their own error nodes)
  coverage_ours  [ours, bias-aware] 1 - root prune_error / (root - root bias_error)
  R (donor)   --donor: src eval.donor_test.donor_target, unchanged, once per target on its distinct (members, alpha)
              sets (fake_circuit objects; held-out bases only, so the prompt is not part of the identity); variants
              --donor-variants (fit), 8 draws, pools A / C / R (R pool: load_or_build_pool filtered against
              --donor-records). R_pass_a / b = pass_a / b AND donor recovery A and C in BAND (donor_test.resample_pass)
  Ge recovery of WCM / Dunefsky units: --emit-ge-jobs writes the A11 jobs; ge_recovery_tcz.py (ge107) runs them;
              --summary merges ge_recovery*.jsonl (mean over the unit's prompts)
  ho_xdoc     [ours, 2026-11-06] the ho windows whose source document (record strong['doc'], the data dir's
              provenance) is in none of the circuit's training windows (XDOC_RULE); Z..pass on that subset, from the
              same per-window reads (no extra forward pass), stored as the ho row's `xdoc` dict (n, idx, note: empty /
              "n/a: no provenance"). Rows scored before it have no `xdoc`
Summary: medians per method (over sweep points), plus a per-target table (best setting per (method, target), the same
  rule for every method: max ho pass_b rate over prompts, then pass_a, then smaller n; when nothing passes, the
  q-fallback: max median q = min(fold(Z), fold(A), fold(C), nec_b), then smaller n; summary.json per_target /
  per_method_rollup) and the ho_xdoc medians (summary.json ho_xdoc).
Scopes: Ge + WCM on ARENA_A_SCOPE=a1 (z + z2 + tc); Dunefsky on a2 (tc only, attention live and undecomposed: their
node universe, dunefsky.md 7.1). One scope per process (wcm_tcz reads the scope at bank load): --scope picks the units.

Outputs (OUT): rows.jsonl (one row per unit x eval set; append + flush, a torn last line is skipped on resume; a unit is
done when its row is there), selftest.jsonl (keep-all and empty circuits per target), summary.md / summary.json
(--summary, CPU, no model; atomic tmp + replace).

  wsl, repo root:
  PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= ARENA_A_SCOPE=a1 .venv/bin/python -u \
      experiments/107-true-baselines/arena_a/score_arena_a.py --scope a1 --out <OUT> \
      --ge out/ge_smoke_up --wcm l1e-3=out/panel/wcm_l1e-3,l1e-3=out/panel/wcm_l1e-3_rerun \
      [--dunefsky out/dunefsky_smoke --scope a2] [--keys k1,k2] [--tau-stride 1] [--dry-run] [--selftest] [--donor]
  ... score_arena_a.py --summary --out <OUT>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as stats
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT_DEFAULT = HERE / "out" / "score_smoke"

D_SAE = 24576
WIDE = (5, 7)
SAME_LAYER = {"z": (), "z2": (), "tc": ("z", "z2")}          # = wcm_tcz.SAME_LAYER
KINDS = {"a1": ("z", "z2", "tc"), "a2": ("tc",)}
BAND, NEC_MIN = (0.75, 1.5), 0.9                              # = eval.wcm_eval.BAND / NEC_MIN
CTX_DIRS = (HERE / "out" / "targets_smoke" / "ctx", HERE / "out" / "panel" / "rec" / "ctx",
            HERE / "out" / "pilot" / "rec" / "ctx")
if os.environ.get("ARENA_A_CTX_DIRS"):        # the pod (run_pod.sh P6): the make_targets records dir, os.pathsep list
    CTX_DIRS = tuple(Path(p) for p in os.environ["ARENA_A_CTX_DIRS"].split(os.pathsep) if p)
K_PROMPTS = 8


# ------------------------------------------------------------------------------------------------ node mapping (pure)
def parse_key(key):
    l, k, i = key.split(".")
    return int(l), k, int(i)


def ge_site(name):
    """their hook name -> (kind, layer) of the dictionary: 'blocks.L.attn.hook_z.sae...' z, 'blocks.L.hook_mlp_out...' tc"""
    layer = int(name.split(".")[1])
    if ".attn.hook_z." in name:
        return "z", layer
    if ".hook_mlp_out." in name:
        return "tc", layer
    raise ValueError("unknown Ge site %r" % name)


def to_ours(kind, layer, latent):
    """a dictionary latent -> our (layer, kind, latent): Kissane L5 / L7 latents >= 24,576 are kind z2"""
    if kind == "z" and layer in WIDE and latent >= D_SAE:
        return layer, "z2", latent - D_SAE
    if latent >= D_SAE:
        raise ValueError("latent %d out of range at %s L%d" % (latent, kind, layer))
    return layer, kind, latent


def is_upstream(site, target, kinds=("z", "z2", "tc")):
    """the bank's upstream rule (eval.ablation_faithfulness.upstream_sites with wcm_tcz's same_layer_upstream)"""
    (l, k), (L, K) = site, target
    if k not in kinds:
        return False
    return l < L or (l == L and k in SAME_LAYER[K])


def map_nodes(latpos_raw, target, kinds=("z", "z2", "tc")):
    """[(layer, kind, pos_bos, latent)] in our kinds with BOS-prefixed positions -> dict(latpos = {(layer, kind, pos,
    latent)} with pos = pos_bos - 1, members = {(layer, kind): {latent}}, counts). Drops BOS (pos_bos 0) and nodes at
    sites that are not upstream of `target` (layer, kind)."""
    out, n_bos, n_off = set(), 0, 0
    for l, k, p, i in latpos_raw:
        if p == 0:
            n_bos += 1
            continue
        if not is_upstream((l, k), target, kinds):
            n_off += 1
            continue
        out.add((int(l), k, int(p) - 1, int(i)))
    return dict(latpos=out, members=collapse(out), n_raw=len(latpos_raw), n_bos=n_bos, n_offsite=n_off,
                n_latpos=len(out), n_latent=sum(len(v) for v in collapse(out).values()))


def collapse(latpos):
    m = {}
    for l, k, _p, i in latpos:
        m.setdefault((l, k), set()).add(i)
    return m


def ge_rows(nodes, sites):
    """run_ge node rows [n, 3] (site index, pos_bos, dictionary latent) -> [(layer, kind, pos_bos, latent)]"""
    out = []
    for s, p, i in (nodes.tolist() if hasattr(nodes, "tolist") else nodes):
        kind, layer = ge_site(sites[int(s)])
        l, k, ii = to_ours(kind, layer, int(i))
        out.append((l, k, int(p), ii))
    return out


def dunefsky_rows(rows, n_tok, root_layer, root_latent):
    """their graph rows -> [(layer, 'tc', pos_bos, latent)]: 'mlp' rows with a latent, root row excluded, the root's own
    (layer, latent) at any token excluded (their read-out drops it from the distinct set); token -1 = the last position"""
    out = []
    for r in rows:
        if r["type"] != "mlp" or r["latent"] is None or r.get("root"):
            continue
        if (int(r["layer"]), int(r["latent"])) == (root_layer, root_latent):
            continue
        t = int(r["token"])
        out.append((int(r["layer"]), "tc", n_tok - 1 if t == -1 else t, int(r["latent"])))
    return out


def _root_error(res, typ):
    root = next((r for r in res["rows"] if r.get("root")), None)
    if root is None:
        return None
    pre = root["name"][: -len("@-1")]
    pe = [e for e in res.get("error_nodes", []) if e["type"] == typ and int(e["token"]) == -1
          and e["name"].startswith(pre)]
    return float(pe[0]["contrib"]) if len(pe) == 1 else None


def dunefsky_coverage(res):
    """theirs: 1 - root prune_error / root (dunefsky.md 7.3(a)); None if the root's prune_error node is absent"""
    pe = _root_error(res, "prune_error")
    if pe is None or not res.get("root"):
        return None
    return 1.0 - pe / float(res["root"])


def dunefsky_coverage_ours(res):
    """[ours, bias-aware] 1 - root prune_error / (root - root bias_error): the coverage of the part of the root that its
    transcoder's encoder bias does not supply (their bias_error node). Not bounded by 1 either: prune_error can be < 0"""
    pe, be = _root_error(res, "prune_error"), _root_error(res, "bias_error")
    if pe is None or be is None or abs(float(res["root"]) - be) <= 1e-9:
        return None
    return 1.0 - pe / (float(res["root"]) - be)


# ------------------------------------------------------------------------------------------------ donor + Ge jobs (pure)
def fake_circuit(key, alphas):
    """a minimal circuit object for src eval.donor_test / wcm_eval.circuit_alphas: a seed node + one node per member with
    its amplitude (alpha), in sorted (site, latent) order"""
    from types import SimpleNamespace as NS
    l, k, i = parse_key(key)
    nodes = {"seed": NS(metadata=dict(role="seed", feature_id=NS(layer=l, kind=k, index=i)))}
    for (sl, sk), d in sorted(alphas.items()):
        for lat, a in sorted(d.items()):
            nodes["%d.%s.%d" % (sl, sk, lat)] = NS(metadata=dict(
                role="member", feature_id=NS(layer=sl, kind=sk, index=int(lat)), amplitude=float(a)))
    return NS(nodes=nodes, metadata={})


def compress_codes(ta, ti, kmax=None):
    """a "sparse" code (acts [..., k], idx [..., k]) whose k is the whole dictionary (ReLU at k = d_sae) -> the same code
    with k = the largest number of non-zero entries at any position (or kmax): values largest first, padded with value 0
    (index 0). Exact for donor_test.ResamplePatcher with budget K = d_sae: it only uses entries with value > 0, all of
    them fit the budget, and scatter_add of the same entries in any order is the same sum; and for sparse_topk_to_dense.
    Memory 24,576 / k times smaller (the dense-k capture held ~2.5 GB per site for 133 donor rows)."""
    import torch
    k = int((ta > 0).sum(-1).max()) if kmax is None else int(kmax)
    k = max(k, 1)
    v, j = torch.topk(ta, k, dim=-1)
    i = ti.gather(-1, j)
    return v * (v > 0), torch.where(v > 0, i, torch.zeros_like(i))


def capture_codes_compact(inference, bank, tokens, sites):
    """src donor_test.capture_codes (same batches, same encode) with each batch's code compressed (compress_codes) and
    the batches padded to one k per site; kept on the GPU (MB-scale). Installed over donor_test.capture_codes for the
    arena's dense ReLU dictionaries [ours, exact; src unchanged]."""
    import torch
    from eval import donor_test as DT
    kinds = list(bank.kinds)
    k2i = {k: i for i, k in enumerate(kinds)}
    parts = {s: ([], []) for s in sites}

    def hook(layer_idx, acts):
        for kd in kinds:
            s = (layer_idx, kd)
            if s in parts:
                ta, ti = bank.encode(acts[k2i[kd]], kd, layer_idx)
                v, i = compress_codes(ta.detach().float(), ti.detach())
                parts[s][0].append(v); parts[s][1].append(i)

    inference.disable_compile()
    try:
        with torch.no_grad():
            for s0 in range(0, int(tokens.shape[0]), DT.BATCH):
                inference.forward(tokens[s0:s0 + DT.BATCH], activations_callback=hook, return_activations=False,
                                  tokenize_final=False)
    finally:
        inference.enable_compile()
    out = {}
    for s, (a, i) in parts.items():
        k = max(int(t.shape[-1]) for t in a)
        pad = lambda t: torch.nn.functional.pad(t, (0, k - int(t.shape[-1])))  # noqa: E731
        out[s] = (torch.cat([pad(t) for t in a]), torch.cat([pad(t) for t in i]))
    return out


def unit_alphas(u):
    return u.get("alphas") or {s: {i: 1.0 for i in v} for s, v in u["members"].items() if v}


def alpha_sig(u):
    """the donor test's identity of a unit: members + alphas (it runs on the held-out bases only, so not the prompt)"""
    return json.dumps(sorted((str(s), sorted((int(i), float(a)) for i, a in d.items()))
                             for s, d in unit_alphas(u).items() if d))


def from_ours(layer, kind, latent):
    """our (layer, kind, latent) -> (Ge hook point, dictionary latent); inverse of to_ours"""
    if kind == "tc":
        return "blocks.%d.hook_mlp_out.sae.hook_feature_acts" % layer, int(latent)
    if kind in ("z", "z2"):
        return "blocks.%d.attn.hook_z.sae.hook_feature_acts" % layer, int(latent) + (D_SAE if kind == "z2" else 0)
    raise ValueError(kind)


def ge_job_nodes(u):
    """the unit's nodes in Ge's space for its A11 fixed-mask pass: Dunefsky at their graph positions (latpos, pos + 1
    for BOS) -> sorted [(hook_point, pos_bos, dict_latent)]; WCM (position-collapsed) -> sorted [(hook_point, None,
    dict_latent)], None = every non-BOS position of the prompt (ge_recovery_tcz expands it; kept compact on disk)"""
    out = set()
    if u.get("latpos") is not None:
        for l, k, p, i in u["latpos"]:
            hp, li = from_ours(l, k, i)
            out.add((hp, int(p) + 1, li))
        return sorted(out)
    for (l, k), v in u["members"].items():
        for i in v:
            out.add(from_ours(l, k, i))
    return sorted((hp, None, li) for hp, li in out)


def record_of(key):
    import torch
    for d in CTX_DIRS:
        p = Path(d) / ("%s.pt" % key)
        if p.exists():
            return torch.load(p, weights_only=False)[key], str(p)
    raise FileNotFoundError("no record for %s in %s" % (key, [str(d) for d in CTX_DIRS]))


def ge_jobs(units):
    """one A11 job per (WCM / Dunefsky unit, prompt): WCM on each of the target's K prompts, Dunefsky on its own"""
    jobs = []
    for u in units:
        if u["method"] not in ("wcm", "dunefsky"):
            continue
        l, k, i = parse_key(u["key"])
        dk, dl = ("z", i + D_SAE) if k == "z2" else (k, i)
        if u["method"] == "dunefsky":
            prs = [(u["j"], u["prompt_tokens"])]
        else:
            prs = [(q, p["tokens"]) for q, p in enumerate(prompts_for(record_of(u["key"])[0]))]
        for j, toks in prs:
            jobs.append(dict(job=u["uid"] + "|p%d" % j, uid=u["uid"], key=u["key"], j=j, method=u["method"], layer=l,
                             dict_kind=dk, dict_latent=dl, tokens=[int(t) for t in toks],
                             nodes=[list(n) for n in ge_job_nodes(u)]))
    return jobs


def ge_recovery(parts, a_t):
    """ge_core.recovery (A11 / A5), plus the bias share"""
    s = sum(parts.values())
    return dict(interp=parts["embed"] / a_t, nonerr=(parts["embed"] + parts["bias_model"] + parts["bias_dict"]) / a_t,
                total=s / a_t, bias=(parts["bias_model"] + parts["bias_dict"]) / a_t,
                err=(parts["err_tc"] + parts["err_z"]) / a_t)


def members_sig(members):
    return tuple(sorted((l, k, tuple(sorted(v))) for (l, k), v in members.items() if v))


# ------------------------------------------------------------------------------------------------ units
def ge_units(ge_dir, keys, scope, tau_stride=1):
    import torch
    if scope != "a1":
        return []
    out = []
    for p in sorted(Path(ge_dir).glob("circuits/*/p*.pt")):
        key = p.parent.name
        if keys and key not in keys:
            continue
        g = torch.load(p, weights_only=False, map_location="cpu")
        if g.get("status") != "ok":
            continue
        target = parse_key(key)[:2]
        a_t, j = float(g["a_t"]), int(g["j"])
        pool = ge_rows(g["posthoc_pool"]["nodes"], g["sites"])
        pool_attr = g["posthoc_pool"]["attr"].tolist()
        for t in range(0, len(g["hier"]), tau_stride):
            h, ph = g["hier"][t], g["posthoc"][t]
            tau = float(h["tau"])
            base = dict(key=key, j=j, knob="tau/a_t", knob_value=tau / a_t, knob_index=t, a_t=a_t, src=str(p),
                        prompt_tokens=list(g["tokens"][1:]))
            mh = map_nodes(ge_rows(h["nodes"], g["sites"]), target, KINDS[scope])
            out.append(dict(base, method="ge_hier", uid="ge_hier|%s|p%d|t%02d" % (key, j, t), alphas=None,
                            ge_n_nodes=int(h["nodes"].shape[0]), ge_rec=ge_recovery(h["parts"], a_t), **mh))
            S = [r for r, a in zip(pool, pool_attr) if a > tau]
            assert len(S) == ph["n"], (p, t, len(S), ph["n"])
            mp = map_nodes(S, target, KINDS[scope])
            out.append(dict(base, method="ge_posthoc", uid="ge_posthoc|%s|p%d|t%02d" % (key, j, t), alphas=None,
                            ge_n_nodes=int(ph["n"]), ge_rec=ge_recovery(ph["parts"], a_t), **mp))
    return out


def dunefsky_units(d_dir, keys, scope):
    import torch
    if scope != "a2":
        return []
    out = []
    for p in sorted(Path(d_dir).glob("circuits/*/p*.b*.i*.pt")):
        key = p.parent.name
        if keys and key not in keys:
            continue
        r = torch.load(p, weights_only=False, map_location="cpu")
        if r.get("status") != "ok":
            continue
        L, kind, i = parse_key(key)
        m = map_nodes(dunefsky_rows(r["rows"], len(r["tokens"]), L, i), (L, kind), KINDS[scope])
        out.append(dict(key=key, j=int(r["j"]), method="dunefsky", knob="num_branches", knob_value=r["num_branches"],
                        knob_index=int(r["num_branches"]), num_iters=r["num_iters"],
                        uid="dunefsky|%s|p%d|b%d|i%d" % (key, r["j"], r["num_branches"], r["num_iters"]),
                        alphas=None, src=str(p), coverage=dunefsky_coverage(r),
                        coverage_ours=dunefsky_coverage_ours(r), n_tc_latents_theirs=r["n_tc_latents"],
                        n_heads=len(r.get("heads", [])), n_head_nodes=len(r.get("head_nodes", [])),
                        root=float(r["root"]), prompt_tokens=[int(t) for t in r["tokens"][1:]], **m))
    return out


def wcm_units(specs, keys, scope):
    """specs: [(label, dir)]; label 'l<lambda>' -> knob lambda. Stored production rows (held strong) go with the unit."""
    import torch
    from eval.wcm_eval import circuit_alphas, circuit_key
    out = []
    for label, d in specs:
        d = Path(d)
        ev = {}
        for q in sorted((d / "main").glob("eval.shard*.jsonl")):
            for line in open(q):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("held") == "strong":
                    ev[r["seed"]] = r
        for p in sorted((d / "main" / "circuits").glob("*.pt")):
            c = torch.load(p, weights_only=False, map_location="cpu")
            if c is None:
                continue
            key = circuit_key(c)
            if keys and key not in keys:
                continue
            al = circuit_alphas(c)
            if scope == "a2" and any(k != "tc" for (_l, k) in al):
                continue                                  # an a1 fit cannot be scored in the tc-only universe
            lam = float(label[1:]) if label.startswith("l") else None
            members = {s: set(v) for s, v in al.items() if v}
            out.append(dict(key=key, j=None, method="wcm", knob="lambda", knob_value=lam, knob_index=lam, label=label,
                            uid="wcm|%s|%s|%s" % (label, d.name, key), alphas={s: dict(v) for s, v in al.items()},
                            src=str(p), members=members, latpos=None, n_raw=None, n_bos=0, n_offsite=0,
                            n_latpos=None, n_latent=sum(len(v) for v in members.values()),
                            stored={f: ev.get(key, {}).get(f) for f in (
                                "a_pos_tk", "e0_tk", "eM_topk_tk", "eN_topk_tk", "free0_tk", "freeM_topk_tk",
                                "freeN_topk_tk", "phi_sup_blind_tk", "free0_raw_tk", "phi_sup_blind_raw_tk",
                                "vacuous_v2")} if key in ev else None))
    return out


def prompts_for(rec, k=K_PROMPTS):
    """make_targets.prompts_for (copied: importing make_targets resets the driver env): the K highest-peak contexts of
    the strong part of the WCM training arm, as (strong-pool index j, anchor, tokens up to the anchor)"""
    from circuit.protocol_contexts import training_arm
    arm, _ = training_arm(rec)
    s = rec["strong"]
    idx = list(rec["D_train"]) if arm == "B" else list(s["train"])
    order = sorted(idx, key=lambda j: (-float(s["peak"][j]), j))[:k]
    return [dict(j=int(j), anchor=int(s["arg"][j]), tokens=[int(t) for t in s["pos"][j][: int(s["arg"][j]) + 1]])
            for j in order]


# ------------------------------------------------------------------------------------------------ helpers
def jl(path):
    out = []
    if Path(path).exists():
        for line in open(path):
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def ratio(x, e, a):
    return (x - e) / (a - e) if (e < a and abs(a - e) > 1e-9) else None


def nec(x, a, e):
    return (a - x) / (a - e) if a - e > 1e-9 else None


def passes(Z, A, C, n):
    if any(v is None for v in (Z, A, C, n)):
        return False
    return all(BAND[0] <= v <= BAND[1] for v in (Z, A, C)) and n >= NEC_MIN


def jsonable(u):
    drop = ("members", "latpos", "alphas", "prompt_tokens")
    return {k: v for k, v in u.items() if k not in drop}


# ------------------------------------------------------------------------------------------------ ho_xdoc (pure)
XDOC_RULE = ("ho_xdoc [ours]: the held-out strongest (ho) windows whose source document (record strong['doc'], from "
             "<data>/provenance/shard_<N>.docs.npy) is in NONE of the circuit's training windows: WCM = every "
             "strongest-train and mid-band-train row of the record (a superset of the arm's fit rows D_train + B_mid); "
             "Ge / Dunefsky = the prompt row their circuit was built on. Contrast rows are not counted (no doc ids)")


def xdoc_split(strong_doc, strong_train, strong_held, mid_doc, mid_train, n_ho, method, prompt_pool_j=None):
    """(idx, info): the positions q in 0..n_ho-1 of the ho eval set (= strong pool rows strong_held[q]) whose document
    is not a training window's document. idx None (info['note'] says why) when the record has no provenance
    (no 'doc' list, or a needed document id is -1, the record builder's 'unknown')."""
    if strong_doc is None or (method == "wcm" and mid_train and mid_doc is None):
        return None, dict(note="n/a: no provenance", n_ho=n_ho)
    hd = [int(strong_doc[j]) for j in list(strong_held)[:n_ho]]
    if method == "wcm":
        tr = [int(strong_doc[j]) for j in strong_train] + [int(mid_doc[j]) for j in (mid_train or [])]
    else:
        tr = [int(strong_doc[prompt_pool_j])]
    if len(hd) < n_ho or any(d < 0 for d in hd + tr):
        return None, dict(note="n/a: no provenance", n_ho=n_ho)
    ts = set(tr)
    idx = [q for q, d in enumerate(hd) if d not in ts]
    return idx, dict(n=len(idx), n_ho=n_ho, n_train_windows=len(tr), n_train_docs=len(ts), n_ho_docs=len(set(hd)),
                     note=None if idx else "empty: every held-out window shares a document with a training window")


def subset_metrics(V, idx, vacuous_fn=None):
    """Z / A / C / nec on the subset idx of an eval set, from its per-window reads V (lists: a_pos, e0, eM, eN, free0,
    freeM, freeN, rm_a, rm_b, rm_btc, clean), exactly as Scorer.unit forms them from the set's means"""
    if vacuous_fn is None:
        from eval.wcm_eval import vacuous_v2 as vacuous_fn
    m = {f: sum(float(V[f][q]) for q in idx) / len(idx) for f in V}
    vac = bool(vacuous_fn(m["a_pos"], [m["e0"], m["eM"], m["eN"]]))
    Z, A, C = ratio(m["free0"], m["e0"], m["a_pos"]), ratio(m["freeM"], m["eM"], m["a_pos"]), ratio(m["freeN"], m["eN"],
                                                                                                   m["a_pos"])
    na, nb, nbt = nec(m["rm_a"], m["a_pos"], m["e0"]), nec(m["rm_b"], m["clean"], m["e0"]), nec(m["rm_btc"], m["clean"],
                                                                                                m["e0"])
    return dict(n=len(idx), a_pos=m["a_pos"], e0=m["e0"], eM_topk=m["eM"], eN_topk=m["eN"], Z=Z, A=A, C=C, nec_a=na,
                nec_b=nb, nec_b_tc=nbt, vacuous=vac, pass_a=None if vac else passes(Z, A, C, na),
                pass_b=None if vac else passes(Z, A, C, nb))


# ------------------------------------------------------------------------------------------------ per-target (pure)
def setting_of(r):
    """the knob setting a row belongs to, comparable across the method's prompts: Ge = the tau grid index (tau / a_t
    differs per prompt), Dunefsky = (num_branches, num_iters), WCM = lambda"""
    if r["method"] in ("ge_hier", "ge_posthoc"):
        return ("t", r["knob_index"])
    if r["method"] == "dunefsky":
        return ("b", r["knob_value"], r.get("num_iters"))
    return ("lam", r["knob_value"] if r["knob_value"] is not None else -1.0)


def setting_label(st, rs):
    if st[0] == "t":
        kv = [r["knob_value"] for r in rs if r.get("knob_value") is not None]
        return "t%02d (tau/a_t %.3g)" % (st[1], stats.median(kv)) if kv else "t%02d" % st[1]
    if st[0] == "b":
        return "b%s i%s" % (st[1], st[2])
    return "lambda %g" % st[1]


def _rate(vals):
    v = [x for x in vals if x is not None]
    return (sum(bool(x) for x in v) / len(v)) if v else None


def _med(vals):
    v = [x for x in vals if x is not None]
    return stats.median(v) if v else None


def fold_band(x, lo=BAND[0], hi=BAND[1]):
    """a Z / A / C value folded to its distance from the pass band: min(x, lo * hi / x) for x > 0 (x itself for x <= 0).
    = lo (0.75) at both band edges, >= lo exactly inside [lo, hi], < lo outside on either side (2.0 -> 0.56)"""
    return x if x <= 0 else min(x, lo * hi / x)


def q_score(r):
    """the no-pass fallback score of one row: min(fold_band(Z), fold_band(A), fold_band(C), nec_b); None if any is
    missing. Nec b is used as is (its pass line is 0.9, no upper bound)"""
    v = [r.get(f) for f in ("Z", "A", "C", "nec_b")]
    if any(x is None for x in v):
        return None
    return min(fold_band(v[0]), fold_band(v[1]), fold_band(v[2]), v[3])


def per_target(rows):
    """one entry per (method, target) from the ho (+ home) rows. Best setting, the same rule for every method: max
    held-out pass_b rate over the setting's prompts (WCM: its circuit(s) at that lambda), then pass_a rate, then the
    smaller median n latent, then the knob order. When NO setting passes (b) or (a) on any prompt (both rates 0
    everywhere), the "q-fallback": highest median over prompts of q_score, then smaller median n, then knob order
    (best_rule 'q-fallback', else 'pass'). Vacuous targets (pass None everywhere) get no best setting."""
    last = {}
    for r in rows:                                          # a unit re-scored after a crash: the last row wins
        last[(r["uid"], r["eval_set"])] = r
    ho = [r for (u, es), r in last.items() if es == "ho"]
    home = {u: r for (u, es), r in last.items() if es == "home"}
    groups = {}
    for r in ho:
        groups.setdefault((r["method"], r["key"]), {}).setdefault(setting_of(r), []).append(r)
    order = {m: i for i, m in enumerate(("ge_hier", "ge_posthoc", "dunefsky", "wcm"))}
    out = []
    for (meth, key), by_s in sorted(groups.items(), key=lambda kv: (order.get(kv[0][0], 9), kv[0][0], kv[0][1])):
        allr = [r for rs in by_s.values() for r in rs]
        cand = []
        for st, rs in by_s.items():
            rb, ra = _rate([r["pass_b"] for r in rs]), _rate([r["pass_a"] for r in rs])
            if rb is None:
                continue
            cand.append(((-rb, -(ra or 0.0), _med([r["n_latent"] for r in rs]), st), st, _med([q_score(r) for r in rs])))
        e = dict(method=meth, key=key, n_prompts=None if meth == "wcm" else len({r["j"] for r in allr}),
                 n_settings=len(by_s), n_units=len(allr), vacuous=not cand)
        passing = [r for r in allr if r["pass_b"]]
        e.update(any_pass_b=bool(passing), n_units_pass_b=len(passing),
                 min_n_pass_b=min([r["n_latent"] for r in passing]) if passing else None)
        if cand:
            knob = lambda c: tuple(-1 if x is None else x for x in c[0][3])  # noqa: E731
            nsz = lambda c: c[0][2] if c[0][2] is not None else 0  # noqa: E731
            fallback = all(c[0][0] == 0 and c[0][1] == 0 for c in cand)
            if fallback:
                key = lambda c: (-(c[2] if c[2] is not None else -math.inf), nsz(c), knob(c))  # noqa: E731
            else:
                key = lambda c: (c[0][0], c[0][1], nsz(c), knob(c))  # noqa: E731
            st = sorted(cand, key=key)[0][1]
            rs = sorted(by_s[st], key=lambda r: (r["j"] if r["j"] is not None else -1, r["uid"]))
            hm = [home.get(r["uid"]) for r in rs]
            xd = [r.get("xdoc") for r in rs]
            xd_ok = [x for x in xd if x and x.get("n")]
            e.update(best=setting_label(st, rs), best_setting=list(st), best_n=len(rs),
                     best_rule="q-fallback" if fallback else "pass", q=_med([q_score(r) for r in rs]),
                     pass_b_rate=_rate([r["pass_b"] for r in rs]), pass_a_rate=_rate([r["pass_a"] for r in rs]),
                     n_pass_b=sum(bool(r["pass_b"]) for r in rs), n_pass_a=sum(bool(r["pass_a"]) for r in rs),
                     n_latent=_med([r["n_latent"] for r in rs]),
                     **{f: _med([r.get(f) for r in rs]) for f in ("Z", "A", "C", "nec_a", "nec_b", "R_A", "R_C", "R_R")},
                     home_pass_b_rate=_rate([h.get("pass_b") for h in hm if h]),
                     xdoc_n=_med([x["n"] for x in xd_ok]), xdoc_pass_b_rate=_rate([x.get("pass_b") for x in xd_ok]),
                     xdoc_note=(None if xd_ok else next((x.get("note") for x in xd if x), "not scored")))
            e["majority_pass_b"] = e["pass_b_rate"] > 0.5
            e["majority_pass_a"] = (e["pass_a_rate"] or 0.0) > 0.5
        out.append(e)
    return out


def per_method_rollup(pt):
    out = {}
    for e in pt:
        s = out.setdefault(e["method"], dict(targets=0, vacuous=0, scored=0, pass_b=0, pass_a=0, any_pass_b=0))
        s["targets"] += 1
        if e["vacuous"]:
            s["vacuous"] += 1
            continue
        s["scored"] += 1
        s["pass_b"] += bool(e["majority_pass_b"])
        s["pass_a"] += bool(e["majority_pass_a"])
        s["any_pass_b"] += bool(e["any_pass_b"])
    return out


# ------------------------------------------------------------------------------------------------ scorer
class Scorer:
    """the shared per-target work and the per-unit runs (model side; imported lazily)"""

    def __init__(self, scope, out_dir):
        os.environ.setdefault("STORES", str(HERE / "out" / "pilot" / "datapass"))
        os.environ.setdefault("DIAG_BATCH", "16")
        os.environ["OUT"] = str(out_dir)                  # the driver's OUT (its WCM_VERSION marker lands here)
        os.environ["CTX_DIR"] = str(CTX_DIRS[0])          # read-only: records are looked up in CTX_DIRS by record()
        os.environ.setdefault("SAE_PRELOAD", "0")
        sys.path.insert(0, str(HERE))
        import diag_necessity_tcz as DN       # registers the family, imports 062's driver (float32, no TF32)
        import torch
        self.DN, self.torch = DN, torch
        self.R = DN.driver.Runner()
        G = self.R.G
        self.inf, self.bank, self.dev = G["inference"], G["bank"], G["device"]
        self.G = G
        self.mlps = [b.mlp for b in self.inf.model.hf.transformer.h]
        import wcm_tcz as WT
        assert WT.scope() == scope, "ARENA_A_SCOPE=%s but --scope %s" % (WT.scope(), scope)
        self.real = set(WT.real_sites())
        self.T = {}

    def record(self, key):
        return record_of(key)

    def donor_pool(self, path, records_dir, run_keys=()):
        """the R pool. records_dir a directory: src donor_test.load_or_build_pool (073's random pool on the corpus,
        exact / near filtered against every record there; needs >= 32 survivors). records_dir "run" [ours, for small
        corpora]: the same random pool and the same filter (donor_test.filter_pool) against the records of the run's
        target keys only, built once at `path` (a later process, e.g. the other scope, reuses the file); >= 16 survivors
        (one donor per activating base; donor_test draws consecutive permutations past that)"""
        import torch
        from eval import donor_test as DT
        if records_dir != "run":
            return DT.load_or_build_pool(Path(path), self.G["M0"].probe_builder.loader, Path(records_dir))
        path = Path(path)
        if not path.exists():
            paths = sorted({Path(record_of(k)[1]) for k in run_keys})
            tokens, ids = DT.random_pool_073(self.G["M0"].probe_builder.loader)
            keep = DT.filter_pool(tokens, DT.iter_record_rows(paths))
            if len(keep) < 16:
                raise RuntimeError("only %d independent random donors after filtering" % len(keep))
            d = dict(tokens=tokens[keep], ids=[ids[j] for j in keep], retained_indices=keep, n_drawn=int(len(tokens)),
                     source="073 random_pool (POOL_SEED %d, %d draws, >= %d tokens)" % (
                         DT.POOL_SEED, DT.POOL_N, DT.POOL_MIN_TOKENS),
                     rule="099 exact / near, against the run's records only [ours]",
                     records=[str(p) for p in paths], n_records=len(paths), version=DT.VERSION)
            tmp = path.with_suffix(".tmp")
            torch.save(d, tmp)
            tmp.replace(path)
        d = torch.load(path, weights_only=False, map_location="cpu")
        info = {k: v for k, v in d.items() if k not in ("tokens", "ids", "retained_indices")}
        info.update(sha256=DT.file_sha256(path), n=int(d["tokens"].shape[0]))
        return d["tokens"].cpu(), info

    def donor(self, key, us, pool, variants, draws):
        """src donor_test.donor_target on the distinct (members, alphas) sets of one target: {alpha_sig: (row, raw)}"""
        from eval import donor_test as DT
        from eval.donor_test import donor_target
        DT.capture_codes = capture_codes_compact                 # exact, MB-scale codes (see compress_codes)
        circ, sigs = {}, {}
        for u in us:
            s = alpha_sig(u)
            if s not in sigs:
                sigs[s] = "s%d" % len(sigs)
                circ[sigs[s]] = fake_circuit(key, unit_alphas(u))
        rec, _ = record_of(key)
        torch = self.torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        rows, raws = donor_target(self.G, key, rec, circ, pool, variants=variants, draws=draws)
        self.donor_peak_mb = (round(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {s: (rows[a], raws[a]) for s, a in sigs.items()}

    # -------------------------------------------------------------- target-level
    def target(self, key, prompt_rows=None):
        if key in self.T:
            return self.T[key]
        torch = self.torch
        from circuit.protocol_contexts import eval_probe
        from eval.ablation_faithfulness import upstream_sites
        from eval.floors import collect_site_anchors, collect_site_means
        from eval.wcm_eval import NaturalTap, contrast_split, eval_split, read_target, site_encoder, vacuous_v2
        self.T.clear()                                                      # one target at a time (memory)
        l, k, i = parse_key(key)
        rec, rec_path = self.record(key)
        probe = eval_probe(rec, "strong", self.dev)
        pt, pa = probe.pos_tokens[:64], probe.pos_argmax[:64]
        nt = probe.neg_tokens[:64]
        n_tr = eval_split(int(pt.shape[0]), probe.metadata, "explicit")
        n_ntr = contrast_split(int(nt.shape[0]), probe.metadata, own=True)
        W, b = site_encoder(self.bank, l, k)
        w_seed, b_seed = W[i], b[i]
        UP = sorted(upstream_sites(self.bank, l, k))
        UPS = set(UP)
        means_tr, _ = collect_site_anchors(self.inf, self.bank, pt[:n_tr], UPS, pa[:n_tr], pin_position_specific=False)
        means_neg = collect_site_means(self.inf, self.bank, nt[:n_ntr], UPS) if UP else {}
        prompts = prompts_for(rec)
        if prompt_rows is not None:                                         # the runners' prompts must be these
            for pr, mine in zip(prompt_rows, prompts):
                assert pr["tokens"] == mine["tokens"] and pr["anchor"] == mine["anchor"], key
        s = rec["strong"]
        home_t = torch.stack([torch.as_tensor(s["pos"][p["j"]]) for p in prompts]).to(device=pt.device,
                                                                                         dtype=pt.dtype)
        home_a = torch.tensor([p["anchor"] for p in prompts], device=pa.device, dtype=pa.dtype)
        sets = {"ho": (pt[n_tr:], pa[n_tr:]), "home": (home_t, home_a)}
        T = dict(key=key, l=l, k=k, i=i, site=(l, k), w=w_seed, b=b_seed, UP=UP, UPS=UPS, means_tr=means_tr,
                 means_neg=means_neg, sets=sets, rec_path=rec_path, n_tr=n_tr, n_ntr=n_ntr, prompts=prompts,
                 ups_real={s_ for s_ in UP if s_ in self.real and not getattr(self.bank.saes[s_[1]][s_[0]], "null",
                                                                                   False)})
        # ho_xdoc: ho row q = strong pool row s["held"][q] (eval_probe: strongest-train, then held-out strongest)
        m_ = rec.get("mid")
        n_ho = int(pt.shape[0]) - n_tr
        held_j = list(s["held"])[:n_ho]
        map_ok = (n_tr == len(s["train"]) and len(held_j) == n_ho and bool(torch.equal(
            pt[n_tr:].cpu().long(), torch.stack([torch.as_tensor(s["pos"][j]) for j in held_j]).long())) if n_ho else
            n_tr == len(s["train"]))
        T["xdoc_rec"] = dict(strong_doc=s.get("doc"), strong_train=list(s["train"]), strong_held=held_j,
                             mid_doc=None if m_ is None else m_.get("doc"),
                             mid_train=[] if m_ is None else list(m_["train"]), n_ho=n_ho, map_ok=map_ok)
        T["dn_means"] = {}
        for nm, (tk, an) in sets.items():
            _, nat = read_target(self.inf, lambda a_, b_: NaturalTap(self.bank, (l, k), w_seed, b_seed, i), tk, an,
                                 per_seq=True)
            T[nm] = dict(a_pos=nat, e0=self.co(T, {}, tk, an), eM=self.co(T, {}, tk, an, means=means_tr, fill=True),
                         eN=self.co(T, {}, tk, an, means=means_neg, fill=True))
        ho = T["ho"]
        T["vacuous"] = vacuous_v2(float(ho["a_pos"].mean()), [float(ho[e].mean()) for e in ("e0", "eM", "eN")])
        self.T[key] = T
        return T

    def co(self, T, keep, tokens, anchors, means=None, fill=False, scales=None):
        """score_circuit's co(): circuit-only run, members at alpha x live, every other latent at every upstream site at
        the fill; per-sequence post-Top-K reads"""
        torch = self.torch
        from eval.wcm_eval import TapCircuitOnly, read_target
        kt = {s: torch.tensor(sorted(v), device=self.dev, dtype=torch.long) for s, v in keep.items() if v}
        ks = scales if keep else None
        _, tk = read_target(self.inf, lambda a, b: TapCircuitOnly(
            bank=self.bank, keep_indices=keep, in_scope=T["UPS"], seed_layer=T["l"], seed_kind=T["k"],
            seed_latent_idx=T["i"], pos_argmax=anchors[a:b], site_means=means, pin_values=None, respect_topk=fill,
            topk=int(self.bank.k), keep_tensors=kt, keep_scales=ks, w_seed=T["w"], b_seed=T["b"]), tokens, anchors,
            per_seq=True)
        return tk

    def cfp_zero(self, T, zero, tokens, anchors):
        """score_circuit's necessity run: members SET to 0 in the live stream (error live)"""
        from eval.wcm_eval import TapIntervention, read_target
        _, tk = read_target(self.inf, lambda a, b: TapIntervention(
            bank=self.bank, activator_targets={}, inhibitor_indices=zero, seed_layer=T["l"], seed_kind=T["k"],
            seed_latent_idx=T["i"], pos_argmax=anchors[a:b], w_seed=T["w"], b_seed=T["b"]), tokens, anchors,
            per_seq=True)
        return tk

    def frozen(self, T, members, tokens, anchors):
        """diag_necessity_tcz's Diag: members zeroed, non-members live, errors frozen at clean (all / tc only)"""
        torch, DN = self.torch, self.DN
        mem = {s: torch.tensor(sorted(v), device=self.dev, dtype=torch.long) for s, v in members.items()
               if v and s in T["ups_real"]}
        ctx = dict(site=T["site"], w=T["w"], b=T["b"], ups=T["ups_real"], bank=self.bank, mlps=self.mlps, members=mem,
                   alpha={}, means={}, xclean={})
        D = lambda **kw: (lambda: DN.Diag(ctx, **kw))  # noqa: E731
        sp = dict(clean=D(capture=True), rm_live=D(mem="zero"), g_err=D(mem="zero", freeze_err={"z", "tc"}),
                  f_tc_err=D(mem="zero", freeze_err={"tc"}))
        return DN.run(self.inf, ctx, tokens, anchors, sp)

    # -------------------------------------------------------------- one unit
    def unit(self, u, T):
        torch = self.torch
        D = int(self.bank.d_sae)
        members = {s: v for s, v in u["members"].items() if v}
        bad = [s for s in members if s not in T["UPS"]]
        assert not bad, (u["uid"], bad)
        if u.get("prompt_tokens") is not None:              # the runner's prompt j = our home-turf row j
            assert T["prompts"][u["j"]]["tokens"] == u["prompt_tokens"], (u["uid"], "prompt mismatch")
        alphas = u.get("alphas") or {s: {i: 1.0 for i in v} for s, v in members.items()}
        scales = {}
        for s, v in members.items():
            sv = torch.ones(D, device=self.dev, dtype=torch.float32)
            idx = sorted(v)
            sv[torch.tensor(idx, device=self.dev)] = torch.tensor([float(alphas[s][i]) for i in idx], device=self.dev)
            scales[s] = sv
        rows = []
        for nm, (tk, an) in T["sets"].items():
            if nm == "home" and u["j"] is not None:
                sel = [q for q, p in enumerate(T["prompts"]) if q == u["j"]]
            else:
                sel = list(range(int(tk.shape[0])))
            tk_, an_ = tk[sel], an[sel]
            base = {f: T[nm][f][sel] for f in ("a_pos", "e0", "eM", "eN")}
            m = {f: float(v.mean()) for f, v in base.items()}
            pw = dict(free0=self.co(T, members, tk_, an_, scales=scales),
                      freeM=self.co(T, members, tk_, an_, means=T["means_tr"], fill=True, scales=scales),
                      freeN=self.co(T, members, tk_, an_, means=T["means_neg"], fill=True, scales=scales),
                      rm_a=self.cfp_zero(T, {s: set(v) for s, v in members.items()}, tk_, an_))
            free0, freeM, freeN, rm_a = (float(pw[f].mean()) for f in ("free0", "freeM", "freeN", "rm_a"))
            fr_pw = self.frozen(T, members, tk_, an_)
            fr = {n: float(v.mean()) for n, v in fr_pw.items()}
            from eval.wcm_eval import vacuous_v2
            vac = vacuous_v2(m["a_pos"], [m["e0"], m["eM"], m["eN"]])
            Z, A, C = ratio(free0, m["e0"], m["a_pos"]), ratio(freeM, m["eM"], m["a_pos"]), ratio(freeN, m["eN"],
                                                                                               m["a_pos"])
            na = nec(rm_a, m["a_pos"], m["e0"])
            nb = nec(fr["g_err"], fr["clean"], m["e0"])
            nbt = nec(fr["f_tc_err"], fr["clean"], m["e0"])
            row = dict(jsonable(u), eval_set=nm, n_rows=len(sel), a_pos=m["a_pos"], e0=m["e0"], eM_topk=m["eM"],
                       eN_topk=m["eN"], free0_raw=free0, freeM_raw=freeM, freeN_raw=freeN, rm_a_raw=rm_a,
                       rm_b_raw=fr["g_err"], rm_btc_raw=fr["f_tc_err"], clean_diag=fr["clean"],
                       rm_live_diag=fr["rm_live"], Z=Z, A=A, C=C, nec_a=na, nec_b=nb, nec_b_tc=nbt, vacuous=vac,
                       pass_a=None if vac else passes(Z, A, C, na), pass_b=None if vac else passes(Z, A, C, nb),
                       n_upstream_sites=len(T["UP"]), n_member_sites=len(members))
            if nm == "ho" and "xdoc_rec" in T:              # the same per-window reads, subset: no extra forward pass
                try:
                    row["xdoc"] = self.xdoc(T, u, dict(
                        a_pos=base["a_pos"], e0=base["e0"], eM=base["eM"], eN=base["eN"], free0=pw["free0"],
                        freeM=pw["freeM"], freeN=pw["freeN"], rm_a=pw["rm_a"], rm_b=fr_pw["g_err"],
                        rm_btc=fr_pw["f_tc_err"], clean=fr_pw["clean"]), len(sel))
                except Exception as e:  # noqa: BLE001  (never lose a scored unit over the extra column)
                    row["xdoc"] = dict(note="error: %s: %s" % (type(e).__name__, e))
            rows.append(row)
        return rows

    def xdoc(self, T, u, V, n_rows):
        """the ho_xdoc subset of this unit's ho reads (xdoc_split) and its metrics (subset_metrics)"""
        X = T["xdoc_rec"]
        if not X["map_ok"]:
            return dict(note="n/a: held-out rows do not map onto the record's strong['held']", n_ho=X["n_ho"])
        pj = None if u["method"] == "wcm" else T["prompts"][u["j"]]["j"]
        idx, info = xdoc_split(X["strong_doc"], X["strong_train"], X["strong_held"], X["mid_doc"], X["mid_train"],
                               X["n_ho"], u["method"], pj)
        if not idx:
            return info
        V = {f: [float(x) for x in v.reshape(-1).tolist()] for f, v in V.items()}
        bad = [f for f, v in V.items() if len(v) != n_rows]
        if bad:
            return dict(info, note="error: per-window reads %s not of length %d" % (bad, n_rows))
        return dict(info, idx=idx, **subset_metrics(V, idx))

    def selftest(self, key):
        """keep-all (every latent at every real upstream site, alpha 1) and empty circuits on the held-out set"""
        torch = self.torch
        T = self.target(key)
        D = int(self.bank.d_sae)
        tk, an = T["sets"]["ho"]
        allm = {s: set(range(D)) for s in T["ups_real"]}
        sc = {s: torch.ones(D, device=self.dev) for s in allm}
        a = float(T["ho"]["a_pos"].mean())
        out = dict(key=key, a_pos=a, e0=float(T["ho"]["e0"].mean()),
                   keepall_free0=float(self.co(T, allm, tk, an, scales=sc).mean()),
                   keepall_freeM=float(self.co(T, allm, tk, an, means=T["means_tr"], fill=True, scales=sc).mean()),
                   keepall_freeN=float(self.co(T, allm, tk, an, means=T["means_neg"], fill=True, scales=sc).mean()),
                   empty_free0=float(self.co(T, {}, tk, an).mean()),
                   empty_rm=float(self.cfp_zero(T, {}, tk, an).mean()))
        out["keepall_Z"] = ratio(out["keepall_free0"], out["e0"], a)
        out["keepall_max_abs_dev"] = max(abs(out[f] - a) for f in ("keepall_free0", "keepall_freeM", "keepall_freeN"))
        return out


# ------------------------------------------------------------------------------------------------ summary (CPU)
def summary_per_target(rows, summ, f):
    """the per-target section (markdown lines); fills summ['per_target'] / summ['per_method_rollup']"""
    pt = per_target(rows)
    roll = per_method_rollup(pt)
    summ["per_target"], summ["per_method_rollup"] = pt, roll
    if not pt:
        return []
    frac = lambda k, n: "-" if n is None else "%d/%d" % (k, n)  # noqa: E731
    md = ["## Per target (one row per method x target; held-out strongest unless marked)", "",
          "Best setting = the same rule for every method: max held-out pass b rate over the setting's prompts, then "
          "pass a rate, then the smaller median n latent (Ge setting = tau grid index; Dunefsky = num_branches, "
          "num_iters; WCM = lambda, one circuit per target, prompts n/a). When no setting passes (b) or (a) on any "
          "prompt, the q-fallback picks it instead (marked 'best: q-fallback'): highest median over prompts of q = "
          "min(fold(Z), fold(A), fold(C), nec b), fold(x) = min(x, 0.75 * 1.5 / x) (= 0.75 at both band edges, "
          "below it outside the band on either side), then smaller n. Columns at the best setting: pass fractions "
          "over prompts, medians over prompts; R = donor recovery A / C / R. 'any b' = some unit at some setting passes "
          "ho (b); 'min n b' = the smallest n latent of a passing unit. Vacuous targets (wcm_eval.vacuous_v2) have no "
          "best setting.", "",
          "| method | target | prompts | settings | best setting | ho pass b | ho pass a | n latent | Z | A | C | nec a | "
          "nec b | q | R A / C / R | home pass b | ho_xdoc n / pass b | any b | min n b |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in pt:
        if e["vacuous"]:
            md.append("| %s | %s | %s | %d | vacuous | - | - | - | - | - | - | - | - | - | - | - | - | %s | %s |" % (
                e["method"], e["key"], "n/a" if e["n_prompts"] is None else e["n_prompts"], e["n_settings"],
                "yes" if e["any_pass_b"] else "no", "-" if e["min_n_pass_b"] is None else e["min_n_pass_b"]))
            continue
        nb = e["best_n"]
        rr = "-" if e["R_A"] is None else "%s / %s / %s" % (f(e["R_A"]), f(e["R_C"]), f(e["R_R"]))
        hm = "-" if e["home_pass_b_rate"] is None else "%.2f" % e["home_pass_b_rate"]
        xd = (e["xdoc_note"] or "-") if e["xdoc_n"] is None else "%s / %s" % (
            f(e["xdoc_n"], 0), "-" if e["xdoc_pass_b_rate"] is None else "%.2f" % e["xdoc_pass_b_rate"])
        md.append("| %s | %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            e["method"], e["key"], "n/a" if e["n_prompts"] is None else e["n_prompts"], e["n_settings"],
            e["best"] + (" (best: q-fallback)" if e["best_rule"] == "q-fallback" else ""),
            frac(e["n_pass_b"], nb), frac(e["n_pass_a"], nb), f(e["n_latent"], 0), f(e["Z"]), f(e["A"]), f(e["C"]),
            f(e["nec_a"]), f(e["nec_b"]), f(e["q"]), rr, hm, xd, "yes" if e["any_pass_b"] else "no",
            "-" if e["min_n_pass_b"] is None else e["min_n_pass_b"]))
    md += ["", "Roll-up: targets passing at the best setting on a majority (> half) of prompts (WCM: the circuit "
           "passes), out of the targets scored (non-vacuous).", "",
           "| method | targets | vacuous | scored | pass b (majority) | pass a (majority) | pass b at any setting |",
           "|---|---|---|---|---|---|---|"]
    for meth, s in roll.items():
        md.append("| %s | %d | %d | %d | %d / %d | %d / %d | %d / %d |" % (
            meth, s["targets"], s["vacuous"], s["scored"], s["pass_b"], s["scored"], s["pass_a"], s["scored"],
            s["any_pass_b"], s["scored"]))
    return md + [""]


def summary_xdoc(rows, summ, f):
    """the ho_xdoc section: medians per method over the units whose ho row carries a non-empty xdoc subset"""
    ho = [r for r in rows if r["eval_set"] == "ho"]
    md = ["## Held-out, same-article windows excluded (ho_xdoc)", "", XDOC_RULE + ".", ""]
    have = [r for r in ho if "xdoc" in r]
    if not ho:
        return []
    if not have:
        summ["ho_xdoc"] = dict(note="not scored: these rows predate ho_xdoc (rescore to fill it)")
        return md + ["Not scored: these rows predate the ho_xdoc column (it is computed at scoring time from the same "
                     "per-window reads, no extra forward pass; rescore to fill it).", ""]
    md += ["| method | units | with subset | empty | n/a (no provenance / other) | vacuous | subset n (median, min-max) "
           "| Z | A | C | nec a | nec b | nec b tc | pass a | pass b | ho pass b (same units) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    out = {}
    for meth in ("ge_hier", "ge_posthoc", "dunefsky", "wcm"):
        rs = [r for r in ho if r["method"] == meth]
        if not rs:
            continue
        xs = [(r, r.get("xdoc") or {}) for r in rs]
        ok = [(r, x) for r, x in xs if x.get("n")]
        nv = [(r, x) for r, x in ok if not x.get("vacuous")]
        empty = sum(1 for _r, x in xs if x.get("n") == 0)
        na = sum(1 for _r, x in xs if x.get("n") is None)
        notes = sorted({x.get("note") for _r, x in xs if x.get("n") is None and x.get("note")})
        ns = [x["n"] for _r, x in ok]
        s = dict(units=len(rs), with_subset=len(ok), empty=empty, na=na, na_notes=notes, vacuous=len(ok) - len(nv),
                 n_median=_med(ns), n_min=min(ns) if ns else None, n_max=max(ns) if ns else None,
                 **{k: _med([x.get(k) for _r, x in nv]) for k in ("Z", "A", "C", "nec_a", "nec_b", "nec_b_tc")},
                 pass_a=sum(bool(x.get("pass_a")) for _r, x in nv), pass_b=sum(bool(x.get("pass_b")) for _r, x in nv),
                 ho_pass_b_same_units=sum(bool(r.get("pass_b")) for r, _x in nv))
        out[meth] = s
        md.append("| %s | %d | %d | %d | %d%s | %d | %s | %s | %s | %s | %s | %s | %s | %d | %d | %d |" % (
            meth, s["units"], s["with_subset"], s["empty"], s["na"], (" (%s)" % "; ".join(notes)) if notes else "",
            s["vacuous"], "-" if not ns else "%s (%d-%d)" % (f(s["n_median"], 0), s["n_min"], s["n_max"]), f(s["Z"]),
            f(s["A"]), f(s["C"]), f(s["nec_a"]), f(s["nec_b"]), f(s["nec_b_tc"]), s["pass_a"], s["pass_b"],
            s["ho_pass_b_same_units"]))
    summ["ho_xdoc"] = dict(rule=XDOC_RULE, per_method=out)
    return md + [""]


def summarize(out_dir):
    rows = jl(out_dir / "rows.jsonl")
    selft = jl(out_dir / "selftest.jsonl")
    # merge: Ge recovery of WCM / Dunefsky units (ge_recovery_tcz.py, mean over the unit's prompts), the bias-aware
    # coverage (rows scored before it existed: recomputed from the Dunefsky file), the donor test (held-out rows)
    grec = {}
    for p in sorted(out_dir.glob("ge_recovery*.jsonl")):
        for r in jl(p):
            if r.get("status") == "ok":
                grec.setdefault(r["uid"], {})[r["job"]] = r
    don = {r["uid"]: r for r in jl(out_dir / "donor.jsonl") if "error" not in r}
    cov_cache = {}
    for r in rows:
        if not r.get("ge_rec") and r["uid"] in grec:
            js = list(grec[r["uid"]].values())
            r["ge_rec"] = {f: stats.mean(j["rec"][f] for j in js) for f in ("interp", "nonerr", "total", "bias", "err")}
            r["ge_rec_jobs"] = len(js)
        if r["method"] == "dunefsky" and "coverage_ours" not in r:
            if r["src"] not in cov_cache:
                import torch
                cov_cache[r["src"]] = dunefsky_coverage_ours(torch.load(r["src"], weights_only=False))
            r["coverage_ours"] = cov_cache[r["src"]]
        d = don.get(r["uid"]) if r["eval_set"] == "ho" else None
        if d:
            r.update(R_A=d["R_A"], R_C=d["R_C"], R_R=d["R_R"], R_pass_a=d["R_pass_a"], R_pass_b=d["R_pass_b"])
    f = lambda v, d=2: "-" if v is None else ("%.*f" % (d, v))  # noqa: E731
    md = ["# Arena A scorer smoke (score_arena_a.py)", "",
          "Held-out strongest (ho) unless marked; Z / A / C / nec as wcm_eval; nec_b = all upstream errors frozen at "
          "clean; pass = Z, A, C in [0.75, 1.5] and nec >= 0.9; recovery = Ge's in-pass leaf parts / a_t.", ""]
    if selft:
        md += ["## Self-test (keep-all and empty circuits, held-out)", "",
               "| target | scope | a_pos | keep-all Z | keep-all max abs dev (Z, A, C reads) | e0 | empty free0 - e0 | "
               "stored e0 |", "|---|---|---|---|---|---|---|---|"]
        for s in selft:
            md.append("| %s | %s | %.4f | %s | %.1e | %.4f | %.1e | %s |" % (
                s["key"], s.get("scope"), s["a_pos"], f(s["keepall_Z"], 6), s["keepall_max_abs_dev"], s["e0"],
                s["empty_free0"] - s["e0"], f(s.get("stored_e0"), 4)))
        md.append("")
    reg = [r for r in rows if r["method"] == "wcm" and r["eval_set"] == "ho" and r.get("stored")]
    if reg:
        md += ["## Regression: WCM column (a) vs the stored production rows (held-out strongest)", "",
               "| circuit | n | dZ | dA | dC | dnec_a | max abs |", "|---|---|---|---|---|---|---|"]
        worst = 0.0
        for r in reg:
            st_ = r["stored"]
            ds = []
            for mine, theirs in (("Z", "free0_tk"), ("A", "freeM_topk_tk"), ("C", "freeN_topk_tk"),
                                 ("nec_a", "phi_sup_blind_tk")):
                a, b = r.get(mine), st_.get(theirs)
                ds.append(None if (a is None or b is None) else (a - b) if not (a is None and b is None) else 0.0)
            mx = max([abs(d) for d in ds if d is not None] or [0.0])
            worst = max(worst, mx)
            md.append("| %s | %d | %s | %s | %s | %s | %.1e |" % (r["uid"], r["n_latent"], *(f(d, 6) for d in ds), mx))
        md += ["", "**max |diff| over the regression = %.2e (criterion 1e-4: %s)**" % (worst, "PASS" if worst <= 1e-4
                                                                                        else "FAIL"), ""]
    head, md = md, []
    for es in ("ho", "home"):
        rs = [r for r in rows if r["eval_set"] == es]
        if not rs:
            continue
        md += ["## Per unit, eval set %s" % es, "",
               "| method | target | prompt | knob | n latent | n (latent,pos) | Z | A | C | nec a | nec b | nec b tc | "
               "pass a | pass b | rec interp | rec nonerr | rec total | coverage | coverage ours | R A | R C |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(rs, key=lambda r: (r["method"], r["key"], r["j"] if r["j"] is not None else -1,
                                           r["knob_index"] if r["knob_index"] is not None else 0)):
            rec = r.get("ge_rec") or {}
            md.append("| %s | %s | %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | "
                      "%s | %s |" % (
                r["method"], r["key"], "-" if r["j"] is None else r["j"],
                ("%.3g" % r["knob_value"]) if r["knob_value"] is not None else "-", r["n_latent"],
                "-" if r["n_latpos"] is None else r["n_latpos"], f(r["Z"]), f(r["A"]), f(r["C"]), f(r["nec_a"]),
                f(r["nec_b"]), f(r["nec_b_tc"]), r["pass_a"], r["pass_b"], f(rec.get("interp")),
                f(rec.get("nonerr")), f(rec.get("total")), f(r.get("coverage")), f(r.get("coverage_ours")),
                f(r.get("R_A")), f(r.get("R_C"))))
        md.append("")
    per_unit, md = md, head
    md += ["## Medians per method and eval set (over all units of the sweep; non-vacuous units only for Z..nec)", "",
           "| method | eval set | units | vacuous | n latent | Z | A | C | nec a | nec b | nec b tc | pass a | pass b | "
           "rec interp | rec nonerr | rec total | coverage (theirs) | coverage (ours, bias-aware) | R A / C / R | "
           "R pass a / b |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    summ = {}
    for meth, es in [(m_, e_) for m_ in ("ge_hier", "ge_posthoc", "dunefsky", "wcm") for e_ in ("ho", "home")]:
        allr = [r for r in rows if r["method"] == meth and r["eval_set"] == es]
        rs = [r for r in allr if not r["vacuous"]]
        if not allr:
            continue

        def med(fn):
            v = [fn(r) for r in rs]
            v = [x for x in v if x is not None]
            return stats.median(v) if v else None
        s = dict(units=len(allr), vacuous=len(allr) - len(rs), n_latent=med(lambda r: r["n_latent"]),
                 Z=med(lambda r: r["Z"]), A=med(lambda r: r["A"]), C=med(lambda r: r["C"]),
                 nec_a=med(lambda r: r["nec_a"]), nec_b=med(lambda r: r["nec_b"]), nec_b_tc=med(lambda r: r["nec_b_tc"]),
                 pass_a=sum(bool(r["pass_a"]) for r in rs), pass_b=sum(bool(r["pass_b"]) for r in rs),
                 rec_interp=med(lambda r: (r.get("ge_rec") or {}).get("interp")),
                 rec_nonerr=med(lambda r: (r.get("ge_rec") or {}).get("nonerr")),
                 rec_total=med(lambda r: (r.get("ge_rec") or {}).get("total")), coverage=med(lambda r: r.get("coverage")),
                 coverage_ours=med(lambda r: r.get("coverage_ours")), R_A=med(lambda r: r.get("R_A")),
                 R_C=med(lambda r: r.get("R_C")), R_R=med(lambda r: r.get("R_R")),
                 R_n=sum(r.get("R_A") is not None or r.get("R_pass_a") is not None for r in rs),
                 R_pass_a=sum(bool(r.get("R_pass_a")) for r in rs), R_pass_b=sum(bool(r.get("R_pass_b")) for r in rs))
        summ["%s|%s" % (meth, es)] = s
        rcol = "-" if not s["R_n"] else "%s / %s / %s" % (f(s["R_A"]), f(s["R_C"]), f(s["R_R"]))
        rpass = "-" if not s["R_n"] else "%d / %d" % (s["R_pass_a"], s["R_pass_b"])
        md.append("| %s | %s | %d | %d | %s | %s | %s | %s | %s | %s | %s | %d | %d | %s | %s | %s | %s | %s | %s | %s |"
                  % (meth, es, s["units"], s["vacuous"], f(s["n_latent"], 0), f(s["Z"]), f(s["A"]), f(s["C"]),
                     f(s["nec_a"]), f(s["nec_b"]), f(s["nec_b_tc"]), s["pass_a"], s["pass_b"], f(s["rec_interp"]),
                     f(s["rec_nonerr"]), f(s["rec_total"]), f(s["coverage"]), f(s["coverage_ours"]), rcol, rpass))
    md += [""] + summary_per_target(rows, summ, f) + summary_xdoc(rows, summ, f) + per_unit
    for name, text in (("summary.md", "\n".join(md) + "\n"), ("summary.json", json.dumps(summ, indent=1) + "\n")):
        tmp = out_dir / (name + ".tmp")
        tmp.write_text(text)
        tmp.replace(out_dir / name)
    print("\n".join(md))


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--scope", default=os.environ.get("ARENA_A_SCOPE", "a1"), choices=("a1", "a2"))
    ap.add_argument("--ge", default="")
    ap.add_argument("--dunefsky", default="")
    ap.add_argument("--wcm", default="", help="label=dir,... (label l<lambda>)")
    ap.add_argument("--keys", default="")
    ap.add_argument("--tau-stride", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="keep-all + empty circuits for every target in the run")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--stop-epoch", type=float, default=0, help="no new target after this unix time")
    ap.add_argument("--donor", action="store_true", help="also the donor test R (src eval.donor_test.donor_target)")
    ap.add_argument("--donor-variants", default="fit", help="comma list of fit, a1, pinned")
    ap.add_argument("--donor-draws", type=int, default=8)
    ap.add_argument("--donor-pool", default="", help="the R pool file [OUT/random_pool.pt; built when missing]")
    ap.add_argument("--donor-records", default=str(CTX_DIRS[0]), help="records the R pool is filtered against")
    ap.add_argument("--emit-ge-jobs", action="store_true",
                    help="write OUT/ge_jobs.<scope>.jsonl (A11 jobs for ge_recovery_tcz.py) and exit; no model")
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.summary:
        return summarize(out_dir)
    os.environ["ARENA_A_SCOPE"] = a.scope
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    torch.set_num_threads(a.threads)
    keys = set(a.keys.split(",")) if a.keys else None
    if not (a.dry_run or a.emit_ge_jobs):
        sc = Scorer(a.scope, out_dir)                   # registers the family (circuits unpickle with its FeatureID)
    else:
        sys.path.insert(0, str(HERE))
        import wcm_tcz as WT
        WT.register()
    units = []
    if a.ge:
        units += ge_units(a.ge, keys, a.scope, a.tau_stride)
    if a.dunefsky:
        units += dunefsky_units(a.dunefsky, keys, a.scope)
    if a.wcm:
        units += wcm_units([tuple(x.split("=", 1)) for x in a.wcm.split(",") if x], keys, a.scope)
    if a.emit_ge_jobs:
        jobs = ge_jobs(units)
        path = out_dir / ("ge_jobs.%s.jsonl" % a.scope)
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(j) + "\n" for j in jobs))
        tmp.replace(path)
        print("ge jobs: %d (from %d units) -> %s" % (len(jobs), sum(u["method"] in ("wcm", "dunefsky") for u in units),
                                                    path), flush=True)
        return
    done = {(r["uid"], r["eval_set"]) for r in jl(out_dir / "rows.jsonl")}
    todo = [u for u in units if (u["uid"], "home") not in done]          # "home" is written after "ho"
    sdone = {(r["key"], r.get("scope")) for r in jl(out_dir / "selftest.jsonl")}
    ddone = {r["uid"] for r in jl(out_dir / "donor.jsonl") if "error" not in r}
    by_t = {}
    for u in todo:
        by_t.setdefault(u["key"], []).append(u)
    if a.selftest:                                                      # targets whose self-test is still missing
        for u in units:
            if (u["key"], a.scope) not in sdone:
                by_t.setdefault(u["key"], [])
    if a.donor:                                                         # targets with donor rows still missing
        for u in units:
            if u["uid"] not in ddone:
                by_t.setdefault(u["key"], [])
        print("donor: %d units without a donor row" % sum(u["uid"] not in ddone for u in units), flush=True)
    n_sig = len({(u["key"], u["j"], members_sig(u["members"]), json.dumps(sorted(
        (str(s), sorted(d.items())) for s, d in (u["alphas"] or {}).items()))) for u in todo})
    print("score_arena_a: scope %s, %d units (%d done, %d to score in %d targets; %d distinct node sets)" % (
        a.scope, len(units), len(units) - len(todo), len(todo), len(by_t), n_sig), flush=True)
    for key, us in sorted(by_t.items()):
        cnt = {}
        for u in us:
            cnt[u["method"]] = cnt.get(u["method"], 0) + 1
        print("  %-12s %s  n_latent %s..%s" % (key, cnt or "(self-test only)", min([u["n_latent"] for u in us] or [0]),
                                               max([u["n_latent"] for u in us] or [0])), flush=True)
    if a.dry_run:
        return
    fh = open(out_dir / "rows.jsonl", "a")
    fs = open(out_dir / "selftest.jsonl", "a")
    stored_of = {u["key"]: u["stored"] for u in units if u.get("stored")}
    ho_of = {r["uid"]: r for r in jl(out_dir / "rows.jsonl") if r["eval_set"] == "ho"}
    if a.donor:
        from eval.donor_test import in_band
        pool, pool_info = sc.donor_pool(a.donor_pool or (out_dir / "random_pool.pt"), a.donor_records,
                                        sorted({u["key"] for u in units}))
        print("donor pool: %d of %s contexts after the filter against %s records (%s)" % (
            int(pool.shape[0]), pool_info.get("n_drawn"), pool_info.get("n_records"), pool_info.get("sha256", "")[:12]),
            flush=True)
        fd, fdr = open(out_dir / "donor.jsonl", "a"), open(out_dir / "donor_raw.jsonl", "a")
        variants = tuple(v for v in a.donor_variants.split(",") if v)
    for key, us in sorted(by_t.items()):
        if a.stop_epoch and time.time() >= a.stop_epoch:
            print("stop epoch reached: not starting %s" % key, flush=True)
            break
        t0 = time.time()
        T = sc.target(key)
        print("[%s] %s target ready (%.0fs): a_pos ho %.4f e0 %.4f vacuous %s" % (
            time.strftime("%H:%M:%S"), key, time.time() - t0, float(T["ho"]["a_pos"].mean()),
            float(T["ho"]["e0"].mean()), T["vacuous"]), flush=True)
        if a.selftest and (key, a.scope) not in sdone:
            s = sc.selftest(key)
            st_ = stored_of.get(key)
            s["stored_e0"] = None if st_ is None else st_.get("e0_tk")
            s["stored_a_pos"] = None if st_ is None else st_.get("a_pos_tk")
            s["scope"] = a.scope
            fs.write(json.dumps(s) + "\n")
            fs.flush()
            print("  selftest: keep-all Z %s (max abs dev %.1e), empty free0 - e0 %.1e, stored e0 %s" % (
                s["keepall_Z"], s["keepall_max_abs_dev"], s["empty_free0"] - s["e0"], s["stored_e0"]), flush=True)
        cache = {}
        for u in us:
            sig = (u["j"], members_sig(u["members"]),
                   json.dumps(sorted((str(s_), sorted(d.items())) for s_, d in (u["alphas"] or {}).items())))
            t1 = time.time()
            if sig in cache:
                rows = [dict(r, **jsonable(u), reused=True) for r in cache[sig]]
            else:
                rows = sc.unit(u, T)
                cache[sig] = rows
            for r in rows:
                fh.write(json.dumps(r) + "\n")
                if r["eval_set"] == "ho":
                    ho_of[r["uid"]] = r
            fh.flush()
            r = rows[0]
            fmt = lambda v: "-" if v is None else "%.3f" % v  # noqa: E731
            print("  %-40s n %5d | Z %s A %s C %s nec a %s b %s | pass a %s b %s%s | %.1fs" % (
                u["uid"], u["n_latent"], fmt(r["Z"]), fmt(r["A"]), fmt(r["C"]), fmt(r["nec_a"]), fmt(r["nec_b"]),
                r["pass_a"], r["pass_b"], " (reused)" if r.get("reused") else "", time.time() - t1), flush=True)
        if a.donor:
            dus = [u for u in units if u["key"] == key and u["uid"] not in ddone]
            if dus:
                t1 = time.time()
                try:
                    res = sc.donor(key, dus, pool, variants, a.donor_draws)
                except Exception as e:  # noqa: BLE001
                    for u in dus:
                        fd.write(json.dumps(dict(uid=u["uid"], key=key, error="%s: %s" % (type(e).__name__, e))) + "\n")
                    fd.flush()
                    print("  donor %s: ERROR %s" % (key, e), flush=True)
                    continue
                written = set()
                for u in dus:
                    s = alpha_sig(u)
                    drow, draw_ = res[s]
                    ho = ho_of.get(u["uid"]) or {}
                    rf = drow["recovery"].get("fit", {}) if "fit" in drow["recovery"] else {}
                    ok_r = bool(in_band(rf.get("A")) and in_band(rf.get("C")))
                    out = dict(jsonable(u), donor=drow, R_A=rf.get("A"), R_C=rf.get("C"), R_R=rf.get("R"),
                               R_pass_a=None if ho.get("pass_a") is None else bool(ho["pass_a"] and ok_r),
                               R_pass_b=None if ho.get("pass_b") is None else bool(ho["pass_b"] and ok_r),
                               donor_sig_shared=s in written, donor_peak_mb=sc.donor_peak_mb,
                               donor_codes="compact (capture_codes_compact, exact)")
                    fd.write(json.dumps(out) + "\n")
                    if s not in written:
                        fdr.write(json.dumps(dict(draw_, uid=u["uid"], key=key)) + "\n")
                    written.add(s)
                fd.flush()
                fdr.flush()
                print("  donor %s: %d units, %d distinct sets, %.0fs, peak allocated %s MB" % (
                    key, len(dus), len(res), time.time() - t1, sc.donor_peak_mb), flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("score_arena_a: done", flush=True)


if __name__ == "__main__":
    main()

"""Exp 132 stage 1: the IOI ceiling on GPT-2 small + OAI v5 32k SAEs under exp 081's scorer, unchanged.

Every variant is scored with eval.wcm_eval.score_task exactly as 081's engine.score calls it (same 5,000 held-out IOI
items, same A / C means from engine.means, alpha = 1 everywhere), only the member set and the ablation scope vary:

  full_all              every latent at every one of the 48 sites, scope = 48 sites (sanity: must be the full model)
  full_attnmlp_in48     every latent at the 24 attn + mlp sites, NO resid / resid_mid members; scope = 48 sites
  full_resid2_in48      every latent at the 24 resid + resid_mid sites, NO attn / mlp members; scope = 48 sites
  full_resid_in48       every latent at the 12 resid sites only; scope = 48
  scope_attnmlp         scope = the 24 attn + mlp sites (resid / resid_mid run LIVE, unreplaced), every latent there
  scope_resid           scope = the 12 resid sites (the other 36 run LIVE), every latent there
  active_all            every latent active (> 0) at a real prompt token of a training item (081's RND pool,
                        out_v4/gpt2-small/active/ioi.json.gz), scope 48
  active_scope_attnmlp  the active pool's attn + mlp latents, scope = attn + mlp (resid / resid_mid live)

  VARIANTS=a,b,...  (default: all, in the order above)
Writes out/ceiling.jsonl (one row per variant, the scorer's full row) and rewrites out/ceiling.md after each.
"""
import gzip
import json
import os
import time

import common132 as K

ORDER = ["full_all", "scope_attnmlp", "full_attnmlp_in48", "full_resid2_in48", "active_all", "active_scope_attnmlp",
         "full_resid_in48", "scope_resid", "active_scope_resid", "active_scope_resid2", "oracle_heldout",
         "oracle_union"]
DESC = {
    "full_all": "all 32,768 latents x 48 sites, scope 48",
    "scope_attnmlp": "all latents at attn+mlp, scope attn+mlp (resid, resid_mid LIVE)",
    "full_attnmlp_in48": "all latents at attn+mlp, resid+resid_mid members none, scope 48",
    "full_resid2_in48": "all latents at resid+resid_mid, attn+mlp members none, scope 48",
    "full_resid_in48": "all latents at resid only, scope 48",
    "scope_resid": "all latents at resid, scope resid (other 36 LIVE)",
    "active_all": "training-active pool (081 RND pool), scope 48",
    "active_scope_attnmlp": "training-active pool at attn+mlp, scope attn+mlp (resid, resid_mid LIVE)",
    "active_scope_resid": "training-active pool at resid, scope resid (other 36 LIVE)",
    "active_scope_resid2": "training-active pool at resid+resid_mid, scope resid+resid_mid (attn, mlp LIVE)",
    "oracle_heldout": "ORACLE: latents active on the held-out prompts themselves, scope 48",
    "oracle_union": "ORACLE: training-active pool + held-out-active latents, scope 48",
}
ACTIVE = K.D081 / "out_v4" / "gpt2-small" / "active" / "ioi.json.gz"


def full(eng, kinds):
    D = int(eng.bank.d_sae)
    return {s: {i: 1.0 for i in range(D)} for s in eng.sites if s[1] in kinds}


def active(eng, kinds):
    d = json.loads(gzip.open(ACTIVE, "rt").read())
    out = {}
    for k, v in d.items():
        l, kd = k.split("/")
        if kd in kinds and v:
            out[(int(l), kd)] = {int(i): 1.0 for i in v}
    return out


def active_rows(eng, B, r0, r1):
    """{site: sorted latents active (> 0) at a real prompt token of rows r0:r1} (engine.active_latents, any rows)."""
    import torch
    k2i = {k: i for i, k in enumerate(eng.kinds)}
    live, cur = {}, {}

    def hook(layer_idx, activations):
        for kd in eng.kinds:
            s = (layer_idx, kd)
            ta, ti = eng.bank.encode(activations[k2i[kd]], kd, layer_idx)
            m = (ta > 0) & cur["v"].to(ta.device)[..., None]
            if s not in live:
                live[s] = torch.zeros(int(eng.bank.d_sae), dtype=torch.bool, device=ti.device)
            live[s][ti[m].long()] = True

    eng.inf.disable_compile()
    try:
        with torch.no_grad():
            for s0 in range(r0, r1, 64):
                s1 = min(r1, s0 + 64)
                cur["v"] = B["valid"][s0:s1]
                eng.inf.forward(B["tokens"][s0:s1], activations_callback=hook, return_activations=False,
                                tokenize_final=False)
    finally:
        eng.inf.enable_compile()
    return {s: set(live[s].nonzero().flatten().cpu().tolist()) for s in eng.sites if s in live}


def oracle(eng, B, name):
    """ORACLE pools (not legitimate circuits: they read the held-out prompts): latents active on the held-out items,
    and the union with the training-active pool."""
    ho = active_rows(eng, B, B["n_train"], int(B["tokens"].shape[0]))
    out = {s: {i: 1.0 for i in v} for s, v in ho.items() if v}
    if name == "oracle_union":
        for s, d in active(eng, K.KINDS).items():
            out.setdefault(s, {}).update(d)
    return out


def build(eng, name, B=None):
    if name in ("oracle_heldout", "oracle_union"):
        return oracle(eng, B, name), None
    A, AM, R2 = K.KINDS, ("attn", "mlp"), ("resid", "resid_mid")
    if name == "full_all":
        return full(eng, A), None
    if name == "scope_attnmlp":
        return full(eng, AM), K.scope_of(eng, AM)
    if name == "full_attnmlp_in48":
        return full(eng, AM), None
    if name == "full_resid2_in48":
        return full(eng, R2), None
    if name == "full_resid_in48":
        return full(eng, ("resid",)), None
    if name == "scope_resid":
        return full(eng, ("resid",)), K.scope_of(eng, ("resid",))
    if name == "active_all":
        return active(eng, A), None
    if name == "active_scope_attnmlp":
        return active(eng, AM), K.scope_of(eng, AM)
    if name == "active_scope_resid":
        return active(eng, ("resid",)), K.scope_of(eng, ("resid",))
    if name == "active_scope_resid2":
        return active(eng, R2), K.scope_of(eng, R2)
    raise KeyError(name)


def write_md(rows):
    L = ["# Exp 132 stage 1: IOI ceiling under the 081 scorer (unchanged)", "",
         "GPT-2 small + OAI v5 32k SAEs, 48 sites; 5,000 held-out IOI items; 081's A / C means; alpha = 1; task metric "
         "= logit(IO) - logit(S). Pass = Z, A, C in [0.75, 1.5] and nec >= 0.9.",
         "SAE errors: score_task (CircuitOnlyPatcher, keep_error_sites=None) keeps every in-scope site's error LIVE in "
         "every frame (error recomputed from the patched stream at that site: x - decode(encode(x))). There is no frame "
         "in this scorer where errors are ablated. Sites outside the scope run live (not re-encoded).", "",
         "| variant | what | n latents | scope sites | full | e0 (Z empty) | Z | A | C | nec | acc Z | pass |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["variant"], DESC.get(r["variant"], ""), r.get("n"), r.get("n_sites_scope"), K.f2(r.get("task_full")),
            K.f2(r.get("task_e0")), K.f2(r.get("task_free0")), K.f2(r.get("task_freeM_topk")),
            K.f2(r.get("task_freeN_topk")), K.f2(r.get("task_nec")), K.f2(r.get("task_acc_free0")), r.get("task_pass")))
    L += ["", "Raw means (task metric, nats): see out/ceiling.jsonl (task_*_raw, task_eM_topk, task_eN_topk).", ""]
    (K.OUT / "ceiling.md").write_text("\n".join(L) + "\n")


def main():
    want = [v for v in os.environ.get("VARIANTS", ",".join(ORDER)).split(",") if v]
    p = K.OUT / "ceiling.jsonl"
    rows = []
    if p.exists():
        rows = [json.loads(l) for l in open(p)]
    done = {r["variant"] for r in rows}
    eng, B, _ = K.setup()
    for name in want:
        if name in done:
            continue
        al, scope = build(eng, name, B)
        t = time.time()
        r = K.score(eng, B, al, scope)
        rec = dict(variant=name, desc=DESC[name], **{k: v for k, v in r.items()
                                                     if isinstance(v, (int, float, str, bool)) or v is None
                                                     or k == "per_kind"})
        K.append(p, rec)
        rows.append(rec)
        write_md(sorted(rows, key=lambda x: ORDER.index(x["variant"])))
        K.log("%-22s n=%-8d full %.3f e0 %.3f Z %s A %s C %s nec %s acc %s pass %s (%.0fs)" % (
            name, r["n"], r["task_full"], r["task_e0"], K.f2(r["task_free0"]), K.f2(r["task_freeM_topk"]),
            K.f2(r["task_freeN_topk"]), K.f2(r["task_nec"]), K.f2(r["task_acc_free0"]), r["task_pass"],
            time.time() - t))
    K.log("DONE")


if __name__ == "__main__":
    main()

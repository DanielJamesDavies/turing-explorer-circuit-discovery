"""Arena C diagnostic: is a WCM task circuit just "every latent that fires"?

On SFC's short templates only a few thousand (site, latent) pairs are ever active, and keeping every active latent
(SAE errors kept) reproduces the model exactly under the zero ablation. So a circuit's Z = 1 is only informative if
the circuit is much smaller than the active set. For each circuit file this prints, per scope site and in total:
  active_heldout   latents active (> 0) at any position of any held-out clean prompt (what Z is scored on)
  active_train     the same over the training clean prompts; active_any = clean + patch, train + held-out
  members          the circuit's latents
  cover            |members ∩ active_heldout| / |active_heldout|: the share of the held-out active set kept
  members_never    members never active on any prompt of the task (they act only through the A / C fills)

Run in WSL with the main venv:
  ./.venv/bin/python experiments/107-true-baselines/arena_c/active_cover.py --task simple <circuit.json> [...]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wcm_marks as WM  # noqa: E402


@torch.no_grad()
def active_sets(inf, bank, tokens, sites, kinds):
    """{site: set of latent ids active (> 0) at any position of any row}."""
    k2i = {k: i for i, k in enumerate(kinds)}
    want = set(sites)
    out = {s: torch.zeros(bank.d_sae, dtype=torch.bool) for s in sites}

    def hook(layer, acts):
        for kd in kinds:
            s = (layer, kd)
            if s in want:
                ta, ti = bank.encode(acts[k2i[kd]], kd, layer)
                on = torch.zeros(bank.d_sae, dtype=torch.bool, device=ta.device)
                on[ti[ta > 0]] = True
                out[s] |= on.cpu()

    for s0 in range(0, int(tokens.shape[0]), 64):
        inf.forward(tokens[s0:s0 + 64], activations_callback=hook, return_activations=False, tokenize_final=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--mode", default="sfc")
    ap.add_argument("--sites", default="noembed")
    ap.add_argument("circuits", nargs="+")
    a = ap.parse_args()
    C, _ = WM._import_081()
    T = WM.build_task(a.task, a.mode)
    rows, n = T["rows"], T["split_meta"]["n_train"]
    inf, bank = WM.load_model()
    dev = inf.device
    tk = lambda key, rs: torch.tensor([r[key] for r in rs], dtype=torch.long, device=dev)   # noqa: E731
    sites = WM.scope_sites(a.sites)
    kinds = list(WM.KINDS)
    act = {nm: active_sets(inf, bank, tk(key, rs), sites, kinds) for nm, key, rs in (
        ("train", "tokens", rows[:n]), ("heldout", "tokens", rows[n:]),
        ("patch_train", "cf_tokens", rows[:n]), ("patch_heldout", "cf_tokens", rows[n:]))}
    any_ = {s: act["train"][s] | act["heldout"][s] | act["patch_train"][s] | act["patch_heldout"][s] for s in sites}
    tot = lambda d: int(sum(int(v.sum()) for v in d.values()))   # noqa: E731
    print("task %s (%d train / %d held-out prompts, length %s): active latents over %d sites: held-out %d, train %d, "
          "any prompt %d" % (a.task, n, len(rows) - n, T["prompt_len_train"], len(sites), tot(act["heldout"]),
                             tot(act["train"]), tot(any_)))
    for path in a.circuits:
        rec = json.loads(Path(path).read_text())
        al = C.alphas_from_json(rec["alphas"])
        mem = {s: torch.zeros(bank.d_sae, dtype=torch.bool) for s in sites}
        for s, d in al.items():
            if s in mem:
                mem[s][list(d)] = True
        n_mem = tot(mem)
        kept_ho = tot({s: mem[s] & act["heldout"][s] for s in sites})
        never = tot({s: mem[s] & ~any_[s] for s in sites})
        print("  %s λ %g %s: members %d | held-out active kept %d of %d (cover %.3f) | members never active %d"
              % (rec.get("means"), rec["lam"], Path(path).name, n_mem, kept_ho, tot(act["heldout"]),
                 kept_ho / max(1, tot(act["heldout"])), never))
        worst = sorted(sites, key=lambda s: -int(act["heldout"][s].sum()))[:4]
        print("    per site (held-out active / kept): " + ", ".join(
            "%s %d/%d" % (WM.sfc_name(s), int(act["heldout"][s].sum()), int((mem[s] & act["heldout"][s]).sum()))
            for s in worst))


if __name__ == "__main__":
    main()

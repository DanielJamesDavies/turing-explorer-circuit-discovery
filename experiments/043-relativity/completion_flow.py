"""COMPLETION FLOW: which latents cause a specific fact completion?

For a cloze prompt whose greedy completion is a known fact (e.g.
"...special relativity in the year" -> "1905"):
  1. forward the prompt; record every ACTIVE latent at every site at
     the final position AND at up to two marked subject positions
     (fact recall literature: subject enrichment happens there);
  2. take the top candidates by activation per site;
  3. knock each candidate out (zero at its site, all positions,
     encode-modify-decode delta) and measure the drop in the answer
     token's log-probability;
  4. print the causal chain ordered by layer -- the internal flow --
     and flag any latent we already know (validated seeds / chain).

  PYTHONPATH=src python .../completion_flow.py
"""
import sys

sys.path.insert(0, "src")
import torch

from hardware import detect_devices, should_compile
from model.hooks import multi_patch
from model.inference import Inference
from model.tokenizer import Tokenizer
from pipeline.component_index import component_idx, split_component_idx
from sae.bank import SAEBank
from sae.dense import sparse_topk_to_dense

KNOWN = {(11, 18699): "Einstein-phrase seed", (29, 3736): "relativity seed",
         (20, 23753): "rel-corrections seed", (26, 455): "Doppler seed",
         (29, 4523): "relativity#2 seed", (35, 13633): "boson seed",
         (5, 17106): "relativistic-stem", (14, 2777): "special-relativity",
         (4, 36431): "Einstein-phrase-1905"}

PROBES = [
    ("Einstein published his theory of special relativity in the year",
     ["relativity", "Einstein"]),
    ("The theory of relativity was developed by Albert",
     ["relativity", "Albert"]),
    ("General relativity describes gravity not as a force but as the curvature of",
     ["relativity", "gravity"]),
]
PER_SITE = 3


class ZeroLatent:
    def __init__(self, bank, site, lat):
        self.bank, self.site, self.lat = bank, site, lat

    def __call__(self, model):
        return multi_patch(model, self.tf)

    def tf(self, layer_idx, kind, x):
        if (layer_idx, kind) != self.site:
            return x
        ta, ti = self.bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, self.bank.d_sae, dtype=x.dtype)
        code = dense.clone()
        code[..., self.lat] = 0.0
        out = self.bank.decode(code - dense, kind, layer_idx, add_bias=False)
        return x + out.to(x.dtype)


def main():
    devices = detect_devices()
    device = devices[0]
    inference = Inference(device=device, compile=should_compile())
    bank = SAEBank(devices=devices, load_decoders=True,
                   compile=should_compile())
    tok = Tokenizer()
    model = inference.model
    n_kinds = len(bank.kinds)
    inference.disable_compile()

    for prompt, subjects in PROBES:
        ids = [1] + [i for i in tok.encode(prompt) if i != 1]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        T = idx.shape[1]
        # subject positions: last sub-token of each subject word
        sub_pos = []
        for w in subjects:
            wids = [i for i in tok.encode(" " + w)
                    if i > 2 and i != 29871]
            for p in range(T - 1, 0, -1):
                if ids[p] == wids[-1]:
                    sub_pos.append(p)
                    break
        read_pos = sorted(set(sub_pos + [T - 1]))

        # 1-2. active latents at the read positions
        cand = {}

        def cb(layer_idx, activations):
            with torch.no_grad():
                for ki, kind in enumerate(bank.kinds):
                    ta, ti = bank.encode(activations[ki], kind, layer_idx)
                    for p in read_pos:
                        v, ii = ta[0, p, :].float(), ti[0, p, :]
                        top = v.topk(min(PER_SITE, v.shape[0]))
                        for val, j in zip(top.values.tolist(),
                                          top.indices.tolist()):
                            if val <= 0:
                                continue
                            key = ((layer_idx, kind), int(ii[j]))
                            cand[key] = max(cand.get(key, 0.0), val)

        with torch.no_grad():
            inference.forward(idx, num_gen=1, tokenize_final=False,
                              activations_callback=cb,
                              return_activations=False)
            logits, _ = model(idx)
            if logits.dim() == 3:
                logits = logits[:, -1, :]
            lp = torch.log_softmax(logits.float(), -1)[0]
            ans = int(lp.argmax())
            base_lp = float(lp[ans])
        ans_txt = tok.decode([ans])
        print("\n=== %r -> %r (logp %.3f) | %d candidate latents"
              % (prompt[-48:], ans_txt, base_lp, len(cand)), flush=True)

        # 3. knockouts
        drops = []
        for (site, lat), act in cand.items():
            p = ZeroLatent(bank, site, lat)
            with torch.no_grad(), p(model):
                lg, _ = model(idx)
                if lg.dim() == 3:
                    lg = lg[:, -1, :]
                lp2 = torch.log_softmax(lg.float(), -1)[0]
            drops.append((base_lp - float(lp2[ans]), site, lat, act))
        drops.sort(reverse=True)

        print("%-10s %-7s %8s %8s  %s"
              % ("site", "latent", "dlogp", "act", "known-as"))
        for d, site, lat, act in drops[:14]:
            comp = component_idx(site[0], bank.kinds.index(site[1]), n_kinds)
            tag = KNOWN.get((comp, lat), "")
            print("L%-2d %-6s %-7d %8.3f %8.2f  %s"
                  % (site[0], site[1], lat, d, act, tag))
    inference.enable_compile()


if __name__ == "__main__":
    main()

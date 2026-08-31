"""ML fact analysis: logit-lens depth profile + causal latent knockouts
for three well-completed ML probes. (Same instruments as the relativity
year analysis, pointed at the rarer domain.)

  PYTHONPATH=src python .../ml_flow.py
"""
import sys

sys.path.insert(0, "src")
import torch

from hardware import detect_devices, should_compile
from model.hooks import multi_patch
from model.inference import Inference
from model.tokenizer import Tokenizer
from pipeline.component_index import component_idx
from sae.bank import SAEBank
from sae.dense import sparse_topk_to_dense

PROBES = [
    "To minimise the loss function, neural networks update parameters in the direction of steepest descent, a method known as gradient",
    "Support vector machines separate classes by finding the hyperplane with the maximum",
    "To evaluate generalisation, the dataset is split into a training set and a",
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
    inference.disable_compile()
    norm_f = model.transformer.norm_f
    W_U = model.lm_head.weight

    for prompt in PROBES:
        ids = [1] + [i for i in tok.encode(prompt) if i != 1]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        resids, cand = {}, {}

        def cb(layer_idx, acts):
            resids[layer_idx] = acts[2][:, -1, :].detach()
            with torch.no_grad():
                for ki, kind in enumerate(bank.kinds):
                    ta, ti = bank.encode(acts[ki], kind, layer_idx)
                    v, ii = ta[0, -1, :].float(), ti[0, -1, :]
                    top = v.topk(min(PER_SITE, v.shape[0]))
                    for val, j in zip(top.values.tolist(),
                                      top.indices.tolist()):
                        if val > 0:
                            cand[((layer_idx, kind), int(ii[j]))] = val

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
        ans_txt = tok.decode([ans]) or repr(ans)
        print("\n=== %r -> %r (logp %.3f)" % (prompt[-46:], ans_txt, base_lp))

        print("-- logit lens:")
        with torch.no_grad():
            for L in sorted(resids):
                h = norm_f(resids[L].to(W_U.dtype))
                lg = (h @ W_U.T).float()
                p = torch.softmax(lg, -1)[0]
                rank = int((p > p[ans]).sum()) + 1
                top3 = [tok.decode([int(t)]) or "?"
                        for t in lg.topk(3).indices[0].tolist()]
                print("  L%-3d P=%8.5f rank %-5d %s"
                      % (L, float(p[ans]), rank,
                         " | ".join(repr(t) for t in top3)))

        print("-- knockouts (top 10 of %d candidates):" % len(cand))
        drops = []
        for (site, lat), act in cand.items():
            pz = ZeroLatent(bank, site, lat)
            with torch.no_grad(), pz(model):
                lg, _ = model(idx)
                if lg.dim() == 3:
                    lg = lg[:, -1, :]
                lp2 = torch.log_softmax(lg.float(), -1)[0]
            drops.append((base_lp - float(lp2[ans]), site, lat, act))
        drops.sort(reverse=True)
        for d, site, lat, act in drops[:10]:
            print("  L%-2d %-6s %-7d dlogp %7.3f  act %6.2f"
                  % (site[0], site[1], lat, d, act))
    inference.enable_compile()


if __name__ == "__main__":
    main()

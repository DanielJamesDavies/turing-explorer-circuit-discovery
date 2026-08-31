"""Show any latent's own top contexts (by comp/lat), decoded from the
training data, with the peak token marked. The audit ranks edges by
causal weight; this names them.

  PYTHONPATH=src python .../topctx_show.py 26/33903 32/12332 ...
"""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")
from data.loader import DataLoader
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from model.tokenizer import Tokenizer
from pipeline.component_index import component_idx, split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/"
                "Runs/20260531-152059-37117a33/20260531-152059-37117a33")
TOPK = int(os.environ.get("TOPK", 3))
CTX = int(os.environ.get("CTX", 9))
NSEQ = int(os.environ.get("NSEQ", 24))     # top contexts read per latent


def render_profile(tok, seq, prof):
    """Render the WHOLE sequence with per-token activation markers:
    [[tok=v]] at >=50% of peak, [tok=v] at >=15%, plain otherwise. The
    firing PROFILE across tokens is data, not decoration: one hot token
    reads as a token detector, a ramp across a clause as a context
    accumulator, and the secondary firing tokens are often the
    interpretive key."""
    peak = float(max(float(prof.max()), 1e-9))
    parts = []
    for i, t in enumerate(seq):
        if t < 0:
            continue
        w = tok.decode([int(t)])
        v = float(prof[i]) if i < prof.shape[0] else 0.0
        if v >= 0.5 * peak:
            parts.append("[[%s=%.0f]]" % (w, v))
        elif v >= 0.15 * peak:
            parts.append("[%s=%.0f]" % (w, v))
        else:
            parts.append(w)
    return "".join(parts).replace(chr(10), " ")


def main():
    targets = [tuple(int(x) for x in a.split("/")) for a in sys.argv[1:]]
    load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
    from store.context import top_ctx
    tok = Tokenizer()
    devices = detect_devices()
    device = devices[0]
    inference = Inference(device=device, compile=should_compile())
    bank = SAEBank(devices=devices, load_decoders=False,
                   compile=should_compile())
    n_kinds = len(bank.kinds)
    loader = DataLoader(device=device, pin_memory=is_fast_memory())

    for comp, lat in targets:
        layer, ki = split_component_idx(comp, n_kinds)
        knd = bank.kinds[ki]
        ids = [int(x) for x in
               top_ctx.ctx_seq_idx[comp, lat].tolist() if x > 0][:NSEQ]
        seqs, order = {}, []
        for b_ids, b_toks in loader.get_batches_by_ids(ids):
            ids_l = b_ids.tolist() if torch.is_tensor(b_ids) else list(b_ids)
            for sid, row in zip(ids_l, b_toks.tolist()):
                seqs[int(sid)] = row
                order.append(int(sid))
        best = []
        for s0 in range(0, len(order), 16):
            chunk = order[s0:s0 + 16]
            toks = torch.tensor([[max(t, 0) for t in seqs[s]] for s in chunk],
                                dtype=torch.long, device=device)

            def cb(layer_idx, activations):
                if layer_idx != layer:
                    return
                with torch.no_grad():
                    ta, ti = bank.encode(activations[ki], knd, layer_idx)
                    hit = (ti == lat)
                    if not bool(hit.any()):
                        return
                    vals = torch.where(hit, ta.float(),
                                       torch.zeros_like(ta.float())).amax(-1)
                    for b, sid in enumerate(chunk):
                        v = float(vals[b].max())
                        if v > 0:
                            best.append((v, sid, vals[b].cpu()))
            inference.forward(toks, num_gen=1, tokenize_final=False,
                              activations_callback=cb,
                              return_activations=False)
        best.sort(key=lambda t: -t[0])
        print("=== comp %d (L%d %s) latent %d -- fires %d/%d contexts"
              % (comp, layer, knd, lat, len(best), len(order)))
        for v, sid, prof in best[:TOPK]:
            print("  peak %.2f | %s"
                  % (v, render_profile(tok, seqs[sid], prof)))
        if not best:
            print("  (silent in its own top contexts?)")
        print()


if __name__ == "__main__":
    main()

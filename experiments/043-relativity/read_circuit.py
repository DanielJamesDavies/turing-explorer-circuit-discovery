"""Read a relativity circuit as a concept graph, describing every member
by ITS OWN stored top contexts.

Each latent already has 64 top-activating sequences in top_ctx (shape
[36, 40960, 64]) -- that is what the store is for. An earlier version
of this script sampled random training text and looked for members
firing in it, which is strictly worse: a relativity-specific member is
rare in random text, so its "top contexts" came back as noise, while a
generic high-frequency member looked confident. Reading each member's
own top_ctx removes that bias entirely and is far cheaper (one pass
over the union of members' contexts, ~1k sequences, instead of
thousands of random ones).

Note the circuit itself was fitted on the SEED'S top_ctx/mid_ctx
sequences at the seed's argmax position (ProbeDatasetBuilder), so seed
and members are described on the same footing as they were fitted.

  PYTHONPATH=src python .../read_circuit.py 29/3736
  env: TOPM (members shown, default 20), TOPK (contexts each, 3),
       ARM (triamp400)
"""
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")
from hardware import detect_devices, should_compile        # noqa: E402
from model.inference import Inference                      # noqa: E402
from model.tokenizer import Tokenizer                      # noqa: E402
from pipeline.component_index import component_idx         # noqa: E402
from pipeline.discovery_artifacts import load_discovery_artifacts  # noqa: E402
from sae.bank import SAEBank                               # noqa: E402

HERE = Path(__file__).parent
RR = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
          "20260531-152059-37117a33/20260531-152059-37117a33")
TOPM = int(os.environ.get("TOPM", 20))
TOPK = int(os.environ.get("TOPK", 3))
BATCH = int(os.environ.get("BATCH", 16))
CTX = int(os.environ.get("CTX", 9))


def main():
    comp0, lat0 = (int(x) for x in sys.argv[1].split("/"))
    arm = os.environ.get("ARM", "triamp400")
    rec = None
    for line in open(HERE / "members.jsonl"):
        r = json.loads(line)
        if (r["comp_idx"], r["latent"], r["arm"]) == (comp0, lat0, arm):
            rec = r
    assert rec, "no %s members for %d/%d" % (arm, comp0, lat0)

    load_discovery_artifacts(RR, candidates_path=RR / "candidates.pt")
    from store.context import top_ctx                      # noqa: E402
    from data.loader import DataLoader                     # noqa: E402

    tok = Tokenizer()
    devices = detect_devices(); device = devices[0]
    inference = Inference(device=device, compile=should_compile())
    bank = SAEBank(devices=devices, load_decoders=False, compile=should_compile())
    n_kinds = len(bank.kinds)
    loader = DataLoader(device=device)

    members = []
    for site, d in rec["alphas"].items():
        lyr, knd = site.split("/")
        for i, a in d.items():
            members.append((float(a), int(lyr), knd, int(i)))
    members.sort(reverse=True)
    shown = members[:TOPM]

    # (comp, latent) -> its own top_ctx sequence ids
    targets = [(comp0, lat0)] + [
        (component_idx(l, bank.kinds.index(k), n_kinds), i)
        for _a, l, k, i in shown]
    own_ids, all_ids = {}, set()
    for c, lat in targets:
        ids = [int(x) for x in top_ctx.ctx_seq_idx[c, lat].tolist() if x > 0]
        own_ids[(c, lat)] = set(ids)
        all_ids |= set(ids)
    all_ids = sorted(all_ids)
    print("reading %d members via their OWN top contexts "
          "(%d unique sequences)" % (len(shown), len(all_ids)), flush=True)

    seqs, order = {}, []
    for b_ids, b_toks in loader.get_batches_by_ids(all_ids):
        ids_l = b_ids.tolist() if torch.is_tensor(b_ids) else list(b_ids)
        for sid, row in zip(ids_l, b_toks.tolist()):
            seqs[int(sid)] = row
            order.append(int(sid))

    want = {}
    for c, lat in targets:
        want.setdefault(c, []).append(lat)
    best = {}
    for s0 in range(0, len(order), BATCH):
        chunk = order[s0:s0 + BATCH]
        toks = torch.tensor([[max(t, 0) for t in seqs[s]] for s in chunk],
                            dtype=torch.long, device=device)

        def cb(layer_idx, activations):
            with torch.no_grad():
                for ki, knd in enumerate(bank.kinds):
                    c = component_idx(layer_idx, ki, n_kinds)
                    if c not in want:
                        continue
                    ta, ti = bank.encode(activations[ki], knd, layer_idx)
                    ta = ta.float()
                    for lat in want[c]:
                        hit = (ti == lat)
                        if not bool(hit.any()):
                            continue
                        vals = torch.where(hit, ta, torch.zeros_like(ta)).amax(-1)
                        for b, sid in enumerate(chunk):
                            if sid not in own_ids[(c, lat)]:
                                continue        # only ITS OWN contexts
                            v, p = vals[b].max(0)
                            if float(v) > 0:
                                best.setdefault((c, lat), []).append(
                                    (float(v), sid, int(p)))
        inference.forward(toks, num_gen=1, tokenize_final=False,
                          activations_callback=cb, return_activations=False)

    def render(c, lat, k=TOPK):
        hits = sorted(best.get((c, lat), []), reverse=True)[:k]
        out = []
        for v, sid, p in hits:
            s = seqs[sid]
            lo, hi = max(0, p - CTX), min(len(s), p + CTX + 1)
            cl = lambda a, b: tok.decode([t for t in s[a:b] if t >= 0])
            out.append("      %6.2f  %s[[%s]]%s"
                       % (v, cl(lo, p), cl(p, p + 1), cl(p + 1, hi)))
        return out or ["      (no positive activation in its own contexts)"]

    print("\nSEED comp %d latent %d -- circuit n=%d (%s)"
          % (comp0, lat0, rec["n"], arm))
    print("\n".join(render(comp0, lat0, 3)))
    print("\ntop %d of %d members by fitted alpha "
          "(remainder not shown, not absent):\n" % (len(shown), len(members)))
    for a, lyr, knd, i in shown:
        c = component_idx(int(lyr), bank.kinds.index(knd), n_kinds)
        print("  alpha=%.2f  L%d %s latent %d" % (a, lyr, knd, i))
        print("\n".join(render(c, i)))
        print()


if __name__ == "__main__":
    main()

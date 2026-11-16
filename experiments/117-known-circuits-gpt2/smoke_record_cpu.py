"""117 CPU smoke of the fit-stage plumbing (no fit): Runner bring-up on the production stores, task_record on a small
scan for one attn-out L9 latent, the record's shapes, a fold, and training_probe / eval_probe / eval_reference.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -X utf8 experiments/117-known-circuits-gpt2/smoke_record_cpu.py
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lib117 as L  # noqa: E402

os.environ["OUT"] = str(L.OUT / "smoke_wcm")
L.setup_env()
import torch  # noqa: E402
import driver  # noqa: E402

R = driver.Runner()
loader = R.G["M0"].probe_builder.loader
n_corpus = int(sum(loader._shard_sequence_counts))
print("corpus windows:", n_corpus)
task = L.load_task("ioi")
key = os.environ.get("KEY")
if not key:                     # the strongest attn-out L9 latent at END on the first IOI training prompt
    rs = [task["rows"][i] for i in task["split"]["train"][:32]]
    tok0 = L.pad_batch([r["tokens"] for r in rs], R.G["inference"].device)
    cap = {}
    kinds = list(R.G["bank"].kinds)

    def cb(li, acts):
        if li == 9:
            cap["x"] = acts[kinds.index("attn")]
    with torch.no_grad():
        R.G["inference"].forward(tok0, activations_callback=cb, return_activations=False, tokenize_final=False)
    ta, ti = R.G["bank"].encode(cap["x"], "attn", 9)
    from collections import Counter
    cnt = Counter()
    for j, r in enumerate(rs):
        e = len(r["tokens"]) - 1
        cnt.update(int(x) for x, a in zip(ti[j, e].tolist(), ta[j, e].tolist()) if a > 0)
    key = "9.attn.%d" % cnt.most_common(1)[0][0]
print("key", key)
io = int(os.environ["IO"]) if os.environ.get("IO") else None
rec = L.task_record(R.G["inference"], R.G["bank"], loader, n_corpus, key, task, "end", n_scan=96, n_held_scan=96,
                    io_token=io)
print({k: (v if not isinstance(v, (dict, torch.Tensor, list)) else type(v).__name__) for k, v in rec.items()})
if rec.get("strong") is not None:
    s = rec["strong"]
    print("strong pos", tuple(s["pos"].shape), "train", len(s["train"]), "held", len(s["held"]), "peak[:3]",
          s["peak"][:3].tolist(), "arg[:3]", s["arg"][:3].tolist())
    print("neg", tuple(rec["neg"].shape), "D_train", len(rec["D_train"]), "mid", rec.get("mid") is not None)
    from circuit.protocol_contexts import eval_probe, eval_reference, training_probe
    tp = training_probe(rec, torch.device("cpu"), R.train_arm(rec)[0])
    ep = eval_probe(L.fold(rec, "strong", 1) or rec, "strong", torch.device("cpu"))
    a, c = eval_reference(rec, torch.device("cpu"))
    print("training_probe", tuple(tp.pos_tokens.shape), tp.metadata, "eval_probe", tuple(ep.pos_tokens.shape),
          ep.metadata, "ref", tuple(a.shape), tuple(c.shape))
    print("decoded row 0:", R.G["inference"].tokenizer.decode(s["pos"][0].tolist())[:160] if hasattr(
        R.G["inference"], "tokenizer") else s["pos"][0][:24].tolist())
print("SMOKE_RECORD_OK")

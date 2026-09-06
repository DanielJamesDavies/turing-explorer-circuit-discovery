"""IOI (Indirect Object Identification, Wang et al. 2022) task dataset for
TuringLLM, in the behaviour_clusters.pt schema used by
044-behaviours/behaviour_runner.py, plus the task-competence check.

  ABBA: "Then, {A} and {B} went to the {place}. {B} gave a {obj} to" -> A
  BABA: "Then, {B} and {A} went to the {place}. {B} gave a {obj} to" -> A
(IO = A, S = B, S2 = the repeated B.) Names/places/objects are checked to
be single tokens in context (prefix-diff encoding). Every prompt is
scored by the intact model: p(IO), p(S), logit diff (IO - S), argmax
correctness. The dataset keeps prompts where logit_diff > 0 (the model
actually does IOI there) so the fitted circuit explains a behaviour the
model has; the full stats are printed either way.

  N=256 PYTHONPATH=src python experiments/047-known-circuits/make_ioi.py
"""
import os
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")
from hardware import detect_devices, should_compile
from model.inference import Inference
from model.tokenizer import Tokenizer

HERE = Path(__file__).parent
SEQ = 64
ANCHOR = SEQ - 2
N = int(os.environ.get("N", 256))
SEED = int(os.environ.get("SEED", 0))

NAMES = ["Mary", "John", "Tom", "James", "Dan", "Martin", "Amy", "Joseph",
         "Jim", "Peter", "Paul", "Bob", "Alice", "Sarah", "Emma", "David",
         "Lucy", "Anna", "Kate", "Mark", "Steve", "Laura", "Jack", "Sam",
         "Mike", "Ben", "Emily", "Chris", "Andrew", "Lisa", "Karen", "Jane",
         "Ryan", "Eric", "Adam", "Alex", "Kevin", "Brian", "Sophie", "Jason"]
PLACES = ["store", "school", "park", "office", "hospital", "garden",
          "station", "restaurant", "house", "market", "beach", "library"]
OBJECTS = ["drink", "book", "ring", "bone", "necklace", "kiss", "snack",
           "basketball", "computer", "letter", "gift", "coat"]
TEMPLATES = [
    "Then, {X} and {Y} went to the {place}. {S} gave a {obj} to",
    "When {X} and {Y} got a {obj} at the {place}, {S} decided to give it to",
    "After {X} and {Y} went to the {place}, {S} gave a {obj} to",
    "While {X} and {Y} were working at the {place}, {S} gave a {obj} to",
]


def single_token_in_context(tok, prefix, word):
    a = tok.encode(prefix)
    b = tok.encode(prefix + " " + word)
    return b[:len(a)] == a and len(b) == len(a) + 1, (b[len(a)] if len(b) > len(a) else None)


def main():
    device = detect_devices()[0]
    inference = Inference(device=device, compile=should_compile())
    tok = Tokenizer()
    inference.disable_compile()
    rng = random.Random(SEED)

    # vocabulary filter: single-token names in a neutral context
    ok_names = [n for n in NAMES if single_token_in_context(tok, "Then,", n)[0]]
    print("single-token names in context: %d/%d" % (len(ok_names), len(NAMES)))

    rows = []
    while len(rows) < N:
        io, s = rng.sample(ok_names, 2)
        place, obj = rng.choice(PLACES), rng.choice(OBJECTS)
        tpl = rng.choice(TEMPLATES)
        abba = rng.random() < 0.5
        x, y = (io, s) if abba else (s, io)
        prompt = tpl.format(X=x, Y=y, S=s, place=place, obj=obj)
        ok_io, io_tok = single_token_in_context(tok, prompt, io)
        ok_s, s_tok = single_token_in_context(tok, prompt, s)
        if not (ok_io and ok_s):
            continue
        pt = tok.encode(prompt)
        if len(pt) > ANCHOR + 1:
            continue
        win = [0] * (ANCHOR + 1 - len(pt)) + pt
        rows.append({"prompt": prompt, "win": win, "io": io_tok, "s": s_tok,
                     "order": "ABBA" if abba else "BABA"})

    # score with the intact model (in the padded frame the fit will use)
    lds, pios, pss, correct = [], [], [], 0
    with torch.no_grad():
        for i in range(0, len(rows), 16):
            chunk = rows[i:i + 16]
            tk = torch.tensor([r["win"] for r in chunk], dtype=torch.long,
                              device=device)
            logits, _ = inference.model(tk)
            if logits.dim() == 3:
                logits = logits[:, -1, :]
            lg = logits.float()
            p = torch.softmax(lg, -1)
            for b, r in enumerate(chunk):
                ld = float(lg[b, r["io"]] - lg[b, r["s"]])
                r["logit_diff"] = ld
                r["p_io"], r["p_s"] = float(p[b, r["io"]]), float(p[b, r["s"]])
                r["argmax_io"] = bool(int(lg[b].argmax()) == r["io"])
                lds.append(ld); pios.append(r["p_io"]); pss.append(r["p_s"])
                correct += int(r["argmax_io"])
    n = len(rows)
    ld_pos = sum(1 for x in lds if x > 0)
    print("IOI on TuringLLM (%d prompts): mean logit diff IO-S %.3f | "
          "logit_diff>0: %d/%d (%.0f%%) | argmax==IO: %d/%d (%.0f%%) | "
          "mean p(IO) %.3f  mean p(S) %.3f"
          % (n, sum(lds) / n, ld_pos, n, 100 * ld_pos / n, correct, n,
             100 * correct / n, sum(pios) / n, sum(pss) / n))
    for order in ("ABBA", "BABA"):
        sub = [r for r in rows if r["order"] == order]
        print("  %s: n=%d | mean logit diff %.3f | argmax==IO %.0f%%"
              % (order, len(sub), sum(r["logit_diff"] for r in sub) / len(sub),
                 100 * sum(r["argmax_io"] for r in sub) / len(sub)))
    for r in rows[:4]:
        print("   e.g. %r -> IO %r (ld %.2f, p %.2f)" % (
            r["prompt"][-60:], tok.decode([r["io"]]), r["logit_diff"], r["p_io"]))

    keep = [r for r in rows if r["logit_diff"] > 0]
    torch.save({"windows": [r["win"] + [r["io"]] for r in keep],
                "targets": [r["io"] for r in keep],
                "s_tokens": [r["s"] for r in keep],
                "probs": [r["p_io"] for r in keep],
                "logit_diffs": [r["logit_diff"] for r in keep],
                "orders": [r["order"] for r in keep],
                "prompts": [r["prompt"] for r in keep],
                "assign": torch.zeros(len(keep), dtype=torch.long),
                "coherence": torch.ones(1), "sizes": torch.tensor([len(keep)]),
                "anchor": ANCHOR}, HERE / "ioi_clusters.pt")
    print("-> ioi_clusters.pt (%d prompts with logit_diff > 0)" % len(keep))


if __name__ == "__main__":
    main()

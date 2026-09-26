"""ACRONYM CAPABILITY PROBE.

The census + inspect found 8.attn.23585 firing at the "(" of
"Dual Language Immersion (DLI)" with distal dependence on 'D'(-6),
'ual'(-5), 'Language'(-4) — the acronym signature. Before any circuit
work, the prerequisite: CAN the model form acronyms, on names it has
not memorised?

  1. three-word capitalised names -> after "(", is the top token the
     first word's initial? (and after that initial, the second?)
  2. the same with the words SHUFFLED between names (novel combinations
     the corpus cannot contain) — real computation must survive this
  3. per-word occlusion of the BEHAVIOUR: replace word 1 / 2 / 3 with
     another name's word and see which output letter changes. A genuine
     acronym algorithm moves letter i when word i changes and not
     otherwise.

  PYTHONPATH=src python experiments/049-circuit-graph/acronym_probe.py
"""
import json
import os
from pathlib import Path

import numpy as np
import torch

from hardware import detect_devices, should_compile
from model.inference import Inference
from model.tokenizer import Tokenizer
from pipeline.discovery_artifacts import load_discovery_artifacts

HERE = Path(__file__).parent
R = Path(os.environ.get("OUT", str(HERE / "results_full")))
torch.set_float32_matmul_precision("high")
load_discovery_artifacts("outputs", candidates_path="outputs/candidates.pt")
devices = detect_devices(); device = devices[0]
inference = Inference(device=device, compile=should_compile())
inference.disable_compile()
tok = Tokenizer()

# Three-word names in the style the corpus uses. Words chosen so each initial
# is a distinct letter; the shuffled condition recombines them into names that
# cannot have been memorised.
NAMES = [
    ("Dual", "Language", "Immersion"), ("Central", "Processing", "Unit"),
    ("Digital", "Signal", "Processing"), ("General", "Data", "Protection"),
    ("Random", "Access", "Memory"), ("National", "Health", "Service"),
    ("Global", "Positioning", "System"), ("Human", "Resource", "Management"),
    ("Public", "Health", "England"), ("Machine", "Learning", "Research"),
    ("Applied", "Behaviour", "Analysis"), ("Early", "Warning", "System"),
    ("Modern", "Language", "Association"), ("Quality", "Assurance", "Team"),
    ("Federal", "Reserve", "Bank"), ("Advanced", "Placement", "Program"),
]
FRAMES = ["The {} ({}", "{} ({}", "known as the {} ({}", "This is the {} ({}"]
rng = np.random.default_rng(0)


def first_id(prefix, word):
    """token id the model must emit next if it continues `prefix` with `word`."""
    base = tok.encode(prefix)
    full = tok.encode(prefix + word)
    return full[len(base)] if len(full) > len(base) else None


def top_tokens(prefix, n=5):
    ids = torch.tensor([tok.encode(prefix)], dtype=torch.long)
    with torch.no_grad():
        res = inference.forward(ids.to(device), all_logits=True, grad_enabled=False,
                                return_activations=False, tokenize_final=False)
    lg = (res[1] if isinstance(res, (tuple, list)) else res)[0, -1].float()
    lp = torch.log_softmax(lg, -1)
    top = torch.topk(lp, n)
    return [(tok.decode([int(i)]), float(v)) for i, v in zip(top.indices, top.values)], lp


def letters_of(name):
    return [w[0] for w in name]


def score_name(name, frame, prefix_letters=""):
    """log p of the NEXT acronym letter given the letters emitted so far."""
    text = frame.format(" ".join(name), prefix_letters)
    want = letters_of(name)[len(prefix_letters)]
    tid = first_id(text, want)
    if tid is None:
        return None, None, None
    top, lp = top_tokens(text)
    return float(lp[tid]), top[0][0], want


print("=" * 100)
print("[1] REAL NAMES: after '(', is the top token the first word's initial?")
hit1 = hit2 = n1 = n2 = 0
rows = []
for name in NAMES:
    lp1, top1, want1 = score_name(name, FRAMES[0])
    if lp1 is None:
        continue
    # the tokenizer merges common acronyms ('RAM', 'CP', 'HR', 'ML', 'AB'), so a merged
    # token STARTING with the wanted initial is a correct continuation, not a miss.
    n1 += 1; ok1 = top1.strip().startswith(want1); hit1 += ok1
    lp2, top2, want2 = score_name(name, FRAMES[0], prefix_letters=want1)
    ok2 = lp2 is not None and top2.strip() == want2
    if lp2 is not None:
        n2 += 1; hit2 += ok2
    rows.append(dict(name=" ".join(name), want=want1, top=top1, lp=lp1, ok=bool(ok1),
                     want2=want2, top2=top2, lp2=lp2, ok2=bool(ok2)))
    print("  %-42s -> want %r top %-12r logp %6.2f %s | 2nd letter want %r top %-10r %s"
          % (" ".join(name), want1, top1, lp1, "OK" if ok1 else "  ", want2, top2, "OK" if ok2 else ""))
print("  first letter %d/%d | second letter %d/%d" % (hit1, n1, hit2, n2))

print("\n[2] SHUFFLED names (novel combinations — memorisation impossible):")
hits = n = 0
srows = []
for _ in range(16):
    a, b, c = rng.integers(0, len(NAMES), 3)
    name = (NAMES[a][0], NAMES[b][1], NAMES[c][2])
    if len({w[0] for w in name}) < 3:
        continue
    lp1, top1, want1 = score_name(name, FRAMES[0])
    if lp1 is None:
        continue
    n += 1; ok = top1.strip().startswith(want1); hits += ok
    srows.append(dict(name=" ".join(name), want=want1, top=top1, lp=lp1, ok=bool(ok)))
    print("  %-42s -> want %r top %-12r logp %6.2f %s" % (" ".join(name), want1, top1, lp1, "OK" if ok else ""))
print("  shuffled first letter %d/%d" % (hits, n))

print("\n[3] SINGLE-VARIABLE SWEEP: words 2-3 FIXED, word 1 varies over many initials.")
print("    The behavioural test the circuit work needs: does the emitted token track W1's")
print("    initial while everything else is held constant? (the gt tens manipulation's analogue)")
W1 = ["Dual", "Central", "Random", "Quality", "Federal", "Modern", "Applied", "Global",
      "Public", "Hybrid", "Vertical", "Eastern", "Joint", "Basic", "Legal", "Native"]
TAILS = [("Language", "Immersion"), ("Processing", "Unit"), ("Access", "Memory"),
         ("Signal", "Analysis"), ("Health", "Service")]
sweeps = []
for tail in TAILS:
    ok = n = 0; line = []
    for w in W1:
        name = (w,) + tail
        lp, top, want = score_name(name, FRAMES[0])
        if lp is None:
            continue
        n += 1
        # credit a merged acronym token too: 'RAM' for Random..., 'CP' for Central...
        hit = top.strip().startswith(want)
        ok += hit
        line.append((w, want, top, round(lp, 2), bool(hit)))
    sweeps.append(dict(tail=" ".join(tail), n=n, ok=ok, rows=line))
    print("  ... %s (  %d/%d track W1's initial)" % (" ".join(tail), ok, n))
    print("      " + "  ".join("%s->%s%s" % (w, repr(t).strip("'"), "" if h else "[x want %s]" % g)
                               for w, g, t, _, h in line))
tot_ok = sum(s["ok"] for s in sweeps); tot_n = sum(s["n"] for s in sweeps)
print("  OVERALL: %d/%d (%.0f%%) of single-word substitutions move the output to W1's initial"
      % (tot_ok, tot_n, 100 * tot_ok / max(tot_n, 1)))
json.dump(dict(real=rows, shuffled=srows, sweeps=sweeps,
               acc_first=hit1 / max(n1, 1), acc_second=hit2 / max(n2, 1), acc_shuffled=hits / max(n, 1),
               acc_sweep=tot_ok / max(tot_n, 1)),
          open(R / "acronym_probe.json", "w"), indent=1)
print("\n->", R / "acronym_probe.json")

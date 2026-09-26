"""CAN GEMMA-2-2B DO TWO-HOP? The precondition for asking whether a
circuit shows internal reasoning.

A two-hop prompt names entity A, requires an intermediate ("bridge")
entity B that is NEVER written, and asks for a property of B:

  "The capital of the state containing Dallas is"  ->  Austin
       A = Dallas,  bridge = Texas (unsaid),  answer = Austin

Each item is probed in three forms, so a two-hop success cannot be an
artefact of the model simply knowing A -> answer by association:

  two_hop   A given, bridge unsaid            (the reasoning case)
  one_hop   bridge given explicitly           (ceiling: pure retrieval)
  hop1      A given, asked for the BRIDGE     (does it know hop 1 at all?)

Reported per family: argmax accuracy, mean logit diff vs a plausible
wrong answer, and p(answer). A family is usable for the reasoning
experiment when hop1 and one_hop are strong AND two_hop is clearly
positive.

  python experiments/051-gemma-reasoning/probe_twohop.py
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
MODEL_ID = os.environ.get("GEMMA_MODEL", "unsloth/gemma-2-2b")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(dev).eval()


def enc(t):
    return tok(t, return_tensors="pt")["input_ids"][0].tolist()


def tok_after(prefix, word):
    a, b = enc(prefix), enc(prefix + word)
    return b[len(a)] if (b[:len(a)] == a and len(b) == len(a) + 1) else None


def first_after(prefix, word):
    """first token of `word` after `prefix` (answers may be multi-token)."""
    a, b = enc(prefix), enc(prefix + word)
    return b[len(a)] if (b[:len(a)] == a and len(b) > len(a)) else None


def last_logits(ids):
    with torch.no_grad():
        return model(torch.tensor([ids], device=dev)).logits[0, -1].float()


# (city A, bridge B, answer, wrong answer)  — bridge never appears in two_hop
US = [("Dallas", "Texas", "Austin", "Boston"), ("Houston", "Texas", "Austin", "Denver"),
      ("Miami", "Florida", "Tallahassee", "Austin"), ("Orlando", "Florida", "Tallahassee", "Denver"),
      ("Chicago", "Illinois", "Springfield", "Austin"), ("Detroit", "Michigan", "Lansing", "Austin"),
      ("Seattle", "Washington", "Olympia", "Austin"), ("Portland", "Oregon", "Salem", "Austin"),
      ("Atlanta", "Georgia", "Atlanta", "Denver"), ("Phoenix", "Arizona", "Phoenix", "Denver")]
COUNTRY = [("Munich", "Germany", "Berlin", "Paris"), ("Hamburg", "Germany", "Berlin", "Madrid"),
           ("Milan", "Italy", "Rome", "Paris"), ("Naples", "Italy", "Rome", "Berlin"),
           ("Barcelona", "Spain", "Madrid", "Rome"), ("Seville", "Spain", "Madrid", "Berlin"),
           ("Lyon", "France", "Paris", "Rome"), ("Marseille", "France", "Paris", "Berlin"),
           ("Osaka", "Japan", "Tokyo", "Beijing"), ("Kyoto", "Japan", "Tokyo", "Seoul"),
           ("Shanghai", "China", "Beijing", "Tokyo"), ("Mumbai", "India", "Delhi", "Tokyo"),
           ("Rio", "Brazil", "Brasilia", "Lima"), ("Toronto", "Canada", "Ottawa", "Lima")]
LANG = [("Kyoto", "Japan", "Japanese", "German"), ("Osaka", "Japan", "Japanese", "French"),
        ("Munich", "Germany", "German", "French"), ("Hamburg", "Germany", "German", "Italian"),
        ("Milan", "Italy", "Italian", "German"), ("Naples", "Italy", "Italian", "Spanish"),
        ("Lyon", "France", "French", "German"), ("Barcelona", "Spain", "Spanish", "German"),
        ("Shanghai", "China", "Chinese", "Japanese"), ("Moscow", "Russia", "Russian", "Polish")]

FAMILIES = {
    "state_capital": dict(items=US,
                          two_hop="Q: What is the capital of the US state containing the city of {a}?\nA: The capital is",
                          one_hop="Q: What is the capital of the US state of {b}?\nA: The capital is",
                          hop1="Q: Which US state contains the city of {a}?\nA: The state of"),
    "country_capital": dict(items=COUNTRY,
                            two_hop="Q: What is the capital of the country containing the city of {a}?\nA: The capital is",
                            one_hop="Q: What is the capital of {b}?\nA: The capital is",
                            hop1="Q: Which country contains the city of {a}?\nA: The country of"),
    "country_language": dict(items=LANG,
                             two_hop="Q: What is the main language of the country containing the city of {a}?\nA: The language is",
                             one_hop="Q: What is the main language of {b}?\nA: The language is",
                             hop1="Q: Which country contains the city of {a}?\nA: The country of"),
}

report = {}
print("family            form      n   acc   margin  p(ans)   example")
for fam, spec in FAMILIES.items():
    for form in ("hop1", "one_hop", "two_hop"):
        rows = []
        for a, b, ans, wrong in spec["items"]:
            prompt = spec[form].format(a=a, b=b)
            # hop1's answer is the bridge; the others' is the property
            target_s, wrong_s = (" " + b, " " + spec["items"][0][1]) if form == "hop1" else (" " + ans, " " + wrong)
            if form == "hop1" and b == spec["items"][0][1]:
                wrong_s = " " + [x[1] for x in spec["items"] if x[1] != b][0]
            ti = first_after(prompt, target_s)
            tc = first_after(prompt, wrong_s)
            if ti is None or tc is None or ti == tc:
                continue
            rows.append((prompt, ti, tc, target_s))
        if not rows:
            print("%-17s %-8s  --  (tokenisation)" % (fam, form)); continue
        acc = margin = ptar = 0.0
        for prompt, ti, tc, _ in rows:
            lg = last_logits(enc(prompt))
            acc += int(lg.argmax() == ti); margin += float(lg[ti] - lg[tc])
            ptar += float(torch.softmax(lg, -1)[ti])
        m = len(rows)
        report["%s.%s" % (fam, form)] = dict(n=m, acc=acc / m, margin=margin / m, p=ptar / m)
        print("%-17s %-8s %3d  %.2f  %+6.2f   %.3f   %r -> %r"
              % (fam, form, m, acc / m, margin / m, ptar / m,
                 rows[0][0].split("\n")[0][-52:], rows[0][3]))
    print()

json.dump(report, open(HERE / "twohop_competence.json", "w"), indent=1)
usable = [f for f in FAMILIES
          if report.get("%s.two_hop" % f, {}).get("margin", -9) > 1.0
          and report.get("%s.one_hop" % f, {}).get("acc", 0) >= 0.6
          and report.get("%s.hop1" % f, {}).get("acc", 0) >= 0.5]
print("USABLE for the reasoning experiment (hop1 and one_hop known, two_hop positive): %s"
      % (", ".join(usable) or "none"))
print("->", HERE / "twohop_competence.json")

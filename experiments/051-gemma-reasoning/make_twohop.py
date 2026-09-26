"""TWO-HOP DATASETS for the reasoning experiment.

Writes:
  twohop_rows.json    prompts for the circuit fit (048 fitter, ROWS=...):
                      "capital of the state containing Dallas" -> Austin,
                      contrast = another state's capital. The BRIDGE
                      (Texas) never appears in the prompt.
  onehop_rows.json    the same targets with the bridge GIVEN explicitly
                      ("capital of Texas") — the control for step 3: a
                      bridge feature should be needed when the bridge must
                      be COMPUTED, less so when it is handed over.
  bridge_probes.json  per bridge entity, short prompts ENDING with that
                      entity, so a member's bridge selectivity can be read
                      at the final position.

  N=400 python experiments/051-gemma-reasoning/make_twohop.py
"""
import json
import os
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).parent
# OUT: where the three json files go. Token ids are tokenizer-specific, so a
# dataset for another model must NOT overwrite this folder's Gemma-2 files.
OUT = Path(os.environ.get("OUT", str(HERE)))
OUT.mkdir(parents=True, exist_ok=True)
MODEL_ID = os.environ.get("GEMMA_MODEL", "unsloth/gemma-2-2b")
tok = AutoTokenizer.from_pretrained(MODEL_ID)


def enc(t):
    return tok(t, return_tensors="pt")["input_ids"][0].tolist()


def first_after(prefix, word):
    """First token of `word` after `prefix` (answers may be multi-token)."""
    a, b = enc(prefix), enc(prefix + word)
    return b[len(a)] if (b[:len(a)] == a and len(b) > len(a)) else None


# (city, bridge, answer)
STATES = [("Dallas", "Texas", "Austin"), ("Houston", "Texas", "Austin"),
          ("San Antonio", "Texas", "Austin"), ("Miami", "Florida", "Tallahassee"),
          ("Orlando", "Florida", "Tallahassee"), ("Tampa", "Florida", "Tallahassee"),
          ("Chicago", "Illinois", "Springfield"), ("Detroit", "Michigan", "Lansing"),
          ("Seattle", "Washington", "Olympia"), ("Portland", "Oregon", "Salem"),
          ("Los Angeles", "California", "Sacramento"), ("San Diego", "California", "Sacramento"),
          ("San Francisco", "California", "Sacramento"), ("Buffalo", "New York", "Albany"),
          ("Rochester", "New York", "Albany"), ("Philadelphia", "Pennsylvania", "Harrisburg"),
          ("Pittsburgh", "Pennsylvania", "Harrisburg"), ("Cleveland", "Ohio", "Columbus"),
          ("Cincinnati", "Ohio", "Columbus"), ("Memphis", "Tennessee", "Nashville"),
          ("New Orleans", "Louisiana", "Baton Rouge")]
COUNTRIES = [("Munich", "Germany", "Berlin"), ("Hamburg", "Germany", "Berlin"),
             ("Frankfurt", "Germany", "Berlin"), ("Cologne", "Germany", "Berlin"),
             ("Milan", "Italy", "Rome"), ("Naples", "Italy", "Rome"), ("Turin", "Italy", "Rome"),
             ("Venice", "Italy", "Rome"), ("Barcelona", "Spain", "Madrid"),
             ("Seville", "Spain", "Madrid"), ("Valencia", "Spain", "Madrid"),
             ("Lyon", "France", "Paris"), ("Marseille", "France", "Paris"),
             ("Nice", "France", "Paris"), ("Bordeaux", "France", "Paris"),
             ("Osaka", "Japan", "Tokyo"), ("Kyoto", "Japan", "Tokyo"), ("Nagoya", "Japan", "Tokyo"),
             ("Shanghai", "China", "Beijing"), ("Shenzhen", "China", "Beijing"),
             ("Mumbai", "India", "Delhi"), ("Bangalore", "India", "Delhi"),
             ("Rio de Janeiro", "Brazil", "Brasilia"), ("Toronto", "Canada", "Ottawa"),
             ("Montreal", "Canada", "Ottawa"), ("Vancouver", "Canada", "Ottawa"),
             ("Istanbul", "Turkey", "Ankara"), ("Sydney", "Australia", "Canberra"),
             ("Melbourne", "Australia", "Canberra")]
LANGS = {"Germany": "German", "Italy": "Italian", "Spain": "Spanish", "France": "French",
         "Japan": "Japanese", "China": "Chinese", "India": "Hindi", "Brazil": "Portuguese",
         "Canada": "English", "Turkey": "Turkish", "Australia": "English"}

TWO_TPL = {
    "state_capital": ["Q: What is the capital of the US state containing the city of {a}?\nA: The capital is",
                      "The capital city of the US state that contains {a} is",
                      "Fact: the capital of the state where {a} is located is"],
    "country_capital": ["Q: What is the capital of the country containing the city of {a}?\nA: The capital is",
                        "The capital city of the country that contains {a} is",
                        "Fact: the capital of the country where {a} is located is"],
    "country_language": ["Q: What is the main language of the country containing the city of {a}?\nA: The language is",
                         "The main language of the country that contains {a} is",
                         "Fact: the language spoken in the country where {a} is located is"],
}
ONE_TPL = {
    "state_capital": ["Q: What is the capital of the US state of {b}?\nA: The capital is",
                      "The capital city of the US state of {b} is",
                      "Fact: the capital of {b} is"],
    "country_capital": ["Q: What is the capital of {b}?\nA: The capital is",
                        "The capital city of {b} is", "Fact: the capital of {b} is"],
    "country_language": ["Q: What is the main language of {b}?\nA: The language is",
                         "The main language of {b} is", "Fact: the language spoken in {b} is"],
}
BRIDGE_TPL = ["The state of {b}", "He grew up in {b}", "a detailed map of {b}",
              "the government of {b}", "travelling south through {b}", "the history of {b}",
              "people who live in {b}", "the economy of {b}"]

FAMILIES = {"state_capital": [(a, b, ans) for a, b, ans in STATES],
            "country_capital": [(a, b, ans) for a, b, ans in COUNTRIES],
            "country_language": [(a, b, LANGS[b]) for a, b, ans in COUNTRIES if b in LANGS]}


def build(which):
    tpls = TWO_TPL if which == "two" else ONE_TPL
    rows, seen = [], set()
    for fam, items in FAMILIES.items():
        answers = sorted({ans for _, _, ans in items})
        for a, b, ans in items:
            wrong = next(x for x in answers if x != ans)
            for tpl in tpls[fam]:
                prompt = tpl.format(a=a, b=b)
                if prompt in seen:
                    continue
                ti, tc = first_after(prompt, " " + ans), first_after(prompt, " " + wrong)
                if ti is None or tc is None or ti == tc:
                    continue
                seen.add(prompt)
                rows.append({"prompt": prompt, "target": ti, "contrast": tc,
                             "meta": {"family": fam, "city": a, "bridge": b, "answer": ans}})
    return rows


two, one = build("two"), build("one")
json.dump(two, open(OUT / "twohop_rows.json", "w"), indent=1)
json.dump(one, open(OUT / "onehop_rows.json", "w"), indent=1)
bridges = sorted({r["meta"]["bridge"] for r in two})
probes = {b: [t.format(b=b) for t in BRIDGE_TPL if not (t.startswith("The state") and b not in
                                                        {x[1] for x in STATES})] for b in bridges}
json.dump(probes, open(OUT / "bridge_probes.json", "w"), indent=1)
print("two-hop rows %d | one-hop rows %d | bridges %d (%s)"
      % (len(two), len(one), len(bridges), ", ".join(bridges[:8]) + " ..."))
print("by family:", {f: sum(1 for r in two if r["meta"]["family"] == f) for f in FAMILIES})
print("e.g. %r -> %r vs %r" % (two[0]["prompt"][-60:].replace("\n", "\\n"),
                               tok.decode([two[0]["target"]]), tok.decode([two[0]["contrast"]])))
print("bridge probe e.g.", probes[bridges[0]][:3])

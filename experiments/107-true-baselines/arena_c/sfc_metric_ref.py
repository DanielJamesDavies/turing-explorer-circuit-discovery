"""Arena C (DAN-194): reference values from SFC's OWN code, for test_wcm_marks.py. Run in the SFC env (~/venvs/sfc107,
nnsight 0.3.7), never in ours. Reads third_party/feature-circuits read-only (sys.path only; cwd unchanged).

For one SVA task: SFC's load_examples (their function, data_loading_utils.py) for the discovery set (<task>_train,
100, min length) and the evaluation set (<task>_test, 40, min length); then, on the first N evaluation examples, SFC's
evaluation metric exactly as ablation.py:162-187 builds it (answer index = tokenizer(answer).input_ids[-1];
m = logit[-1, clean] - logit[-1, patch], computed inside model.trace on the clean prompts). Prints one JSON line.

  ~/venvs/sfc107/bin/python sfc_metric_ref.py --task simple --n 8
"""
import argparse
import json
import os
import sys

FC = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../third_party/feature-circuits"))
sys.path.insert(0, FC)

import torch as t  # noqa: E402
from nnsight import LanguageModel  # noqa: E402
from data_loading_utils import load_examples  # noqa: E402  (theirs)

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="simple")
ap.add_argument("--n", type=int, default=8)
a = ap.parse_args()

model = LanguageModel("EleutherAI/pythia-70m-deduped", attn_implementation="eager", torch_dtype=t.float32,
                      device_map="cpu", dispatch=True)
disc = load_examples(os.path.join(FC, "data", "%s_train.json" % a.task), 100, model, use_min_length_only=True)
ev = load_examples(os.path.join(FC, "data", "%s_test.json" % a.task), 40, model, use_min_length_only=True)
examples = ev[:a.n]
clean_inputs = [e["clean_prefix"] for e in examples]
clean_answer_idxs = t.tensor([model.tokenizer(e["clean_answer"]).input_ids[-1] for e in examples], dtype=t.long)
patch_answer_idxs = t.tensor([model.tokenizer(e["patch_answer"]).input_ids[-1] for e in examples], dtype=t.long)


def metric_fn(model):                                         # ablation.py:176-180, verbatim
    logits = model.output.logits[:, -1, :]
    return -t.gather(logits, dim=-1, index=patch_answer_idxs.view(-1, 1)).squeeze(
        -1
    ) + t.gather(logits, dim=-1, index=clean_answer_idxs.view(-1, 1)).squeeze(-1)


with t.no_grad():
    with model.trace(clean_inputs):
        metric = metric_fn(model).save()
        last = model.output.logits[:, -1, :].save()
print("JSON" + json.dumps(dict(
    task=a.task, discovery_prefixes=[e["clean_prefix"] for e in disc], eval_prefixes=[e["clean_prefix"] for e in ev],
    eval_answers=[[e["clean_answer"], e["patch_answer"]] for e in ev],
    metric=[float(x) for x in metric.value], last_logits_head=[[float(x) for x in r[:16]] for r in last.value])))

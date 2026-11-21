"""129 E (CPU only, forward passes): does each 117 circuit compute its target the way the task requires, or does it
only reproduce the activation on the prompts it was scored on?

Discriminating controls on held-out task prompts (first N_PROMPTS), target pre-activation at the anchor, in the full
model and in the corpus / task circuits' Z frames (117 circuits, errors live, c_heads.frame_pass with grad off):
  IOI-NM (per-name latents, IO = name n): (a) prompts with n as IO (the scored condition); (b) the SAME prompts with n
          put in BOTH S slots (S1, S2) and the original IO name kept: the name is present but is the subject, so the
          IOI algorithm (S-inhibition) says the name movers should NOT copy it; (c) the original prompts (n absent).
  IOI-SI / IOI-IND: (a) the IOI prompt; (b) its ABC counterfactual (cf_tokens: three distinct names, no repeated
          subject), anchor at the same role (END / S2).
  GT-MLP: Spearman rho(pre at END, YY) over the prompts, plus the mean.
The full model gives the reference contrast; a circuit that implements the task should reproduce (a) - (b), not just (a).

  CUDA_VISIBLE_DEVICES= ./.venv/bin/python -I experiments/129-known-circuits-diagnosis/e_func.py -> out/e_func.json/.md
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c_heads as C  # noqa: E402  (sets CUDA_VISIBLE_DEVICES="", imports lib117 as C.L)
import torch  # noqa: E402

L = C.L
N = int(os.environ.get("N_PROMPTS", "64"))
BS = 16


def pre_of(inf, bank, key, toks, anchors, mem):
    out = []
    for s0 in range(0, len(toks), BS):
        tok = L.pad_batch(toks[s0:s0 + BS], torch.device("cpu"))
        anc = torch.tensor(anchors[s0:s0 + BS])
        roles = [["x"] * tok.shape[1]] * tok.shape[0]
        out.append(C.frame_pass(inf, bank, key, tok, anc, roles, members=mem, grad=False)["pre"])
    return torch.cat(out)


def spearman(a, b):
    import numpy as np
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "16")))
    from config import config
    from model.inference import Inference
    from sae.public_bank import PublicSAEBank
    config.model.family = "gpt2-small"
    inf = Inference(torch.device("cpu"), family="gpt2-small")
    inf.disable_compile(); inf.enable_compile = lambda: None
    bank = PublicSAEBank(device=torch.device("cpu"), family="gpt2-small", preload=False)
    sel = json.loads((C.E117 / "out" / "targets.json").read_text())
    keys = [t for t in (C.E117 / "out" / "targets.txt").read_text().split() if t]
    if os.environ.get("KEYS"):
        keys = os.environ["KEYS"].split(",")
    tasks = {}
    res = []
    md = ["# 129 E: discriminating controls (target pre-activation at the anchor, mean over %d held-out prompts)" % N, ""]
    for key in keys:
        pname = next(p for p, d in sel["pathways"].items() if any(c["key"] == key for c in d["chosen"]))
        pw = L.PATHWAYS[pname]
        task = tasks.setdefault(pw["task"], L.load_task(pw["task"]))
        ent = next(c for c in sel["pathways"][pname]["chosen"] if c["key"] == key)
        io = ent.get("io_token")
        conds = {}
        if pname == "IOI-NM":
            items, toks = L.task_items(task, "held", "end", io)
            items, toks = items[:N], toks[:N]
            rows = [task["rows"][i] for i in items]
            anc = [len(t) - 1 for t in toks]
            as_s = []
            for r in rows:
                t = list(r["tokens"])
                for j, ro in enumerate(r["roles"]):
                    if ro in ("s1", "s2"):
                        t[j] = int(io)
                as_s.append(t)
            conds = {"name_as_IO": (toks, anc), "name_as_S": (as_s, anc),
                     "name_absent": ([list(r["tokens"]) for r in rows], anc)}
        elif pname in ("IOI-SI", "IOI-IND"):
            role = pw["role"]
            items, toks = L.task_items(task, "held", role)
            pairs = [(i, t) for i, t in zip(items, toks)
                     if L.role_pos({"roles": task["rows"][i]["cf_roles"]}, role) is not None][:N]
            rows = [task["rows"][i] for i, _ in pairs]
            conds = {"ioi": ([t for _, t in pairs], [L.role_pos(r, role) for r in rows]),
                     "abc_cf": ([list(r["cf_tokens"]) for r in rows],
                                [L.role_pos({"roles": r["cf_roles"]}, role) for r in rows])}
        else:
            items, toks = L.task_items(task, "held", "end")
            items, toks = items[:N], toks[:N]
            rows = [task["rows"][i] for i in items]
            conds = {"gt": (toks, [len(t) - 1 for t in toks])}
            yy = [int(r["meta"]["yy"]) for r in rows]
        frames = [("full", None)]
        for cname, sub in (("corpus", "main"), ("task", "task")):
            cp = C.E117 / "out" / "wcm" / sub / "circuits" / ("%s.pt" % key)
            if cp.exists():
                c = torch.load(cp, weights_only=False, map_location="cpu")
                if c is not None:
                    frames.append((cname, L.circuit_members(c)))
        row = dict(seed=key, pathway=pname, frames={})
        for fname, mem in frames:
            d = {}
            for cn, (tk, an) in conds.items():
                p = pre_of(inf, bank, key, tk, an, mem)
                d[cn] = float(p.mean())
                d[cn + "_fire"] = float((p > 0).float().mean())
                if pname == "GT-MLP":
                    d["rho_yy"] = spearman(p.numpy(), yy)
            row["frames"][fname] = d
            print("[129 E] %s %s %s" % (key, fname, json.dumps({k: round(v, 3) for k, v in d.items()})), flush=True)
        res.append(row)
    (HERE / "out" / "e_func.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for r in res:
        md.append("- **%s** (%s): " % (r["seed"], r["pathway"]) + "; ".join(
            "%s: %s" % (f, ", ".join("%s %.2f" % (k, v) for k, v in d.items())) for f, d in r["frames"].items()))
    (HERE / "out" / "e_func.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()

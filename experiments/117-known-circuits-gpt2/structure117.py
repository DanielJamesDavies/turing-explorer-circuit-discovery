"""117 stage 3 (GPU): attribution structure of each fitted circuit on held-out task prompts, with nulls.

Per target and circuit (corpus = the production circuit, task = the task-record circuit), on the first N_PROMPTS
held-out-split task items (names / nouns disjoint from training), prompt only (the anchor is the prompt's END / S2):
  frame "circuit"  the circuit's Z frame, fitted alphas (lib117.attribution)
  frame "null"     N_DRAWS random matched sets: per site the circuit's member count drawn from the latents active (> 0)
                   at a real token of the first 512 training-split prompts at that site, the circuit's alphas permuted
                   within the site; same frame and attribution
  frame "full"     every latent live (the unedited model), once per target
Rows: out/wcm/dan215/structure.jsonl, one per (seed, circuit, frame, draw): mass [[layer, kind, head, role, |attr|]...],
signed [[layer, kind, role, attr]...], mean pre-activation in the frame and the natural one (full frame).

  N_PROMPTS=96 N_DRAWS=20 PYTHONPATH=src python -X utf8 -u experiments/117-known-circuits-gpt2/structure117.py
"""
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lib117 as L  # noqa: E402

N_PROMPTS = int(os.environ.get("N_PROMPTS", "96"))
N_DRAWS = int(os.environ.get("N_DRAWS", "20"))
BS = int(os.environ.get("STRUCT_BS", "8"))
OUT = Path(os.environ.get("OUT", str(L.OUT / "wcm")))
SEED = 20261115


def active_pool(inf, bank, task, n=512):
    kinds = list(bank.kinds)
    rows = [task["rows"][i] for i in list(task["split"]["train"])[:n]]
    live = {}
    for s0 in range(0, len(rows), 32):
        rb = rows[s0:s0 + 32]
        tok = L.pad_batch([r["tokens"] for r in rb], inf.device)
        valid = torch.zeros(tok.shape, dtype=torch.bool, device=inf.device)
        for j, r in enumerate(rb):
            valid[j, :len(r["tokens"])] = True

        def cb(li, acts):
            for kd in kinds:
                ta, ti = bank.encode(acts[kinds.index(kd)], kd, li)
                m = (ta > 0) & valid[..., None]
                s = (li, kd)
                if s not in live:
                    live[s] = torch.zeros(int(bank.d_sae), dtype=torch.bool, device=ti.device)
                live[s][ti[m].long()] = True
        inf.disable_compile()
        try:
            with torch.no_grad():
                inf.forward(tok, activations_callback=cb, return_activations=False, tokenize_final=False)
        finally:
            inf.enable_compile()
    return {s: v.nonzero().flatten().cpu().numpy() for s, v in live.items()}


def run_frame(inf, bank, key, batches, members):
    mass, signed, pres, tks = {}, {}, [], []
    for tok, anc, roles in batches:
        m, s, pre, tk = L.attribution(inf, bank, key, tok, anc, roles, members=members)
        for k, v in m.items():
            mass[k] = mass.get(k, 0.0) + v
        for k, v in s.items():
            signed[k] = signed.get(k, 0.0) + v
        pres.append(pre); tks.append(tk)
        torch.cuda.empty_cache()
    return (dict(mass=[[k[0], k[1], k[2], k[3], v] for k, v in mass.items()],
                 signed=[[k[0], k[1], k[2], v] for k, v in signed.items()]),
            float(torch.cat(pres).mean()), float(torch.cat(tks).mean()))


def main():
    from config import config
    from model.inference import Inference
    from sae.public_bank import PublicSAEBank
    config.model.family = "gpt2-small"
    torch.set_float32_matmul_precision("high")
    dev = torch.device(os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    inf = Inference(dev, family="gpt2-small")
    bank = PublicSAEBank(device=dev, family="gpt2-small", preload=False)
    # compile off for the whole process: re-wrapping every block in torch.compile on each attribution call leaked file
    # descriptors (job 2 died with "Too many open files" / CUDA driver error after ~2,000 toggles)
    inf.disable_compile()
    inf.enable_compile = lambda: None
    sel = json.loads((L.OUT / "targets.json").read_text())
    keys = [t for t in (L.OUT / "targets.txt").read_text().split() if t]
    if os.environ.get("KEYS"):
        keys = [k for k in os.environ["KEYS"].split(",") if k]
    path = Path(os.environ.get("STRUCT_OUT", str(OUT / "dan215" / "structure.jsonl")))
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line)
                done.add((r["seed"], r["circuit"], r["frame"], r["draw"]))
            except (ValueError, KeyError):
                continue
    fh = open(path, "a")
    tasks, pools = {}, {}
    for key in keys:
        pname = next(p for p, d in sel["pathways"].items() if any(c["key"] == key for c in d["chosen"]))
        pw = L.PATHWAYS[pname]
        if pw["task"] not in tasks:
            tasks[pw["task"]] = L.load_task(pw["task"])
            pools[pw["task"]] = active_pool(inf, bank, tasks[pw["task"]])
        task, pool = tasks[pw["task"]], pools[pw["task"]]
        ent = next(c for c in sel["pathways"][pname]["chosen"] if c["key"] == key)
        items, toks = L.task_items(task, "held", pw["role"], ent.get("io_token"))
        items, toks = items[:N_PROMPTS], toks[:N_PROMPTS]
        batches = []
        for s0 in range(0, len(items), BS):
            rb = [task["rows"][i] for i in items[s0:s0 + BS]]
            tok = L.pad_batch(toks[s0:s0 + BS], dev)
            anc = torch.tensor([L.role_pos(r, pw["role"]) for r in rb], device=dev)
            batches.append((tok, anc, [L.roles_of(r) for r in rb]))
        t0 = time.time()
        if (key, "-", "full", 0) not in done:
            res, pre, tk = run_frame(inf, bank, key, batches, None)
            fh.write(json.dumps(dict(seed=key, pathway=pname, circuit="-", frame="full", draw=0, pre=pre, tk=tk,
                                     n=None, n_prompts=len(items), **res)) + "\n"); fh.flush()
        for cname in ("corpus", "task"):
            cp = OUT / ("main" if cname == "corpus" else "task") / "circuits" / ("%s.pt" % key)
            if not cp.exists():
                continue
            c = torch.load(cp, weights_only=False)
            if c is None:
                continue
            mem = L.circuit_members(c)
            n = sum(len(v) for v in mem.values())
            if (key, cname, "circuit", 0) not in done:
                res, pre, tk = run_frame(inf, bank, key, batches, mem)
                fh.write(json.dumps(dict(seed=key, pathway=pname, circuit=cname, frame="circuit", draw=0, pre=pre,
                                         tk=tk, n=n, n_prompts=len(items), **res)) + "\n"); fh.flush()
            rng = np.random.default_rng([SEED, zlib.crc32(key.encode()), 0 if cname == "corpus" else 1])
            for dr in range(N_DRAWS):
                draw_mem = {}
                for s, dct in mem.items():
                    pl = pool.get(s, np.array([], dtype=np.int64))
                    k = min(len(dct), len(pl))
                    if k == 0:
                        continue
                    lat = rng.choice(pl, size=k, replace=False)
                    al = rng.permutation(np.array(list(dct.values()), dtype=float))[:k]
                    draw_mem[s] = {int(a): float(b) for a, b in zip(lat, al)}
                if (key, cname, "null", dr) in done:
                    continue
                res, pre, tk = run_frame(inf, bank, key, batches, draw_mem)
                fh.write(json.dumps(dict(seed=key, pathway=pname, circuit=cname, frame="null", draw=dr, pre=pre,
                                         tk=tk, n=n, n_prompts=len(items), **res)) + "\n"); fh.flush()
        print("[117 struct] %s done in %.0fs" % (key, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

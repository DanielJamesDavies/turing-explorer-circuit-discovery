"""Staged VRAM ledger — measure, don't estimate (plan D2.7).

Walks the pipeline's memory stages and prints a ledger of allocated /
reserved / peak at each, so the "one sequence takes 16 GB" mystery is
decomposed into named, rerunnable measurements:

  S0  CUDA context baseline
  S1  model load                       (param bytes by dtype)
  S2  SAE bank load                    (param bytes by dtype)
  S3  plain inference forward          (no grad)
  S4  inference + SAE passthroughs     (no grad — the eval configuration)
  S5  backward on the model only       (grad forward + backward, no SAEs)
  S6  discovery-config backward        (SAE graph instrument, grads w.r.t.
      upstream latent anchors) — swept over #instrumented sites and batch
  S7  S5/S6 repeated under config.discovery.autocast_bf16 (--autocast)

Synthetic tokens only — no run artifacts needed. Optional --snapshot dumps
torch's allocation history for the S6 peak stage (view with torch memory-viz).

Run from the repo root:
    PYTHONPATH=src python -m debug.vram_ledger [--batches 1 2 4]
        [--site-counts 2 8 17 26 35] [--autocast] [--snapshot] [--out DIR]
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from config import config
from hardware import detect_devices, is_fast_memory, should_compile

GB = 1024 ** 3
VOCAB, T = 50304, 64
SEED_LATENT = 123


def mem(device):
    torch.cuda.synchronize(device)
    return (torch.cuda.memory_allocated(device) / GB,
            torch.cuda.memory_reserved(device) / GB)


def param_bytes_by_dtype(module):
    out = defaultdict(int)
    for p in module.parameters():
        out[str(p.dtype)] += p.numel() * p.element_size()
    for b in module.buffers():
        out[str(b.dtype)] += b.numel() * b.element_size()
    return {k: round(v / GB, 3) for k, v in out.items()}


class Ledger:
    def __init__(self, device, collect=False):
        self.device = device
        self.collect = collect
        self.rows = []

    def stage(self, name, fn=None, note=""):
        if self.collect:
            gc.collect()           # the leak kill-test: break cycles NOW
        torch.cuda.empty_cache()
        a0, r0 = mem(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        t0 = time.time()
        extra = fn() if fn else None
        secs = time.time() - t0
        a1, r1 = mem(self.device)
        peak = torch.cuda.max_memory_allocated(self.device) / GB
        row = {"stage": name, "alloc_before": round(a0, 3),
               "alloc_after": round(a1, 3), "delta": round(a1 - a0, 3),
               "peak": round(peak, 3), "reserved": round(r1, 3),
               "secs": round(secs, 2), "note": note, "extra": extra}
        self.rows.append(row)
        print("%-42s alloc %6.3f->%6.3f GB (d %+6.3f) | peak %6.3f | "
              "reserved %6.3f | %5.1fs %s"
              % (name, a0, a1, a1 - a0, peak, r1, secs, note), flush=True)
        return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--site-counts", type=int, nargs="+",
                    default=[2, 8, 17, 26, 35])
    ap.add_argument("--autocast", action="store_true",
                    help="repeat grad stages under autocast_bf16")
    ap.add_argument("--snapshot", action="store_true",
                    help="dump allocation history for the largest S6 stage")
    ap.add_argument("--gc", action="store_true",
                    help="gc.collect() between stages — the leak kill-test")
    ap.add_argument("--out", default="dev-notes/data/vram-ledger-2026-07-31")
    args = ap.parse_args()

    devices = detect_devices()
    device = devices[0]
    assert device.type == "cuda", "ledger requires CUDA"
    led = Ledger(device, collect=args.gc)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens = {b: torch.randint(0, VOCAB, (b, T), device=device)
              for b in args.batches}

    led.stage("S0 cuda context",
              lambda: torch.zeros(1, device=device).item() and None)

    # S1 model
    from model.inference import Inference
    holder = {}

    def load_model():
        holder["inference"] = Inference(device=device, compile=should_compile())
        return param_bytes_by_dtype(holder["inference"].model)
    led.stage("S1 model load", load_model)
    inference = holder["inference"]

    # S2 bank
    from sae.bank import SAEBank

    def load_bank():
        holder["bank"] = SAEBank(devices=devices,
                                 load_decoders=is_fast_memory(),
                                 compile=should_compile())
        tot = defaultdict(int)
        for kind, per_layer in holder["bank"].saes.items():
            for sae in per_layer:
                if sae is None:
                    continue
                for k, v in param_bytes_by_dtype(sae).items():
                    tot[k] += v
        return {k: round(v, 3) for k, v in tot.items()}
    led.stage("S2 SAE bank load", load_bank)
    bank = holder["bank"]

    # S3 plain forward (largest batch)
    B = max(args.batches)

    def plain_fwd():
        inference.forward(tokens[B], grad_enabled=False,
                          return_activations=False, tokenize_final=False)
    inference.disable_compile()
    led.stage("S3 forward no-grad (B=%d)" % B, plain_fwd)

    # S4 forward with SAE passthroughs, no grad (eval configuration)
    from circuit.instrument.sae_graph import SAEGraphInstrument

    def sae_fwd():
        instrument = SAEGraphInstrument(bank)
        with torch.no_grad():
            inference.forward(tokens[B], patcher=instrument,
                              grad_enabled=False, return_activations=False,
                              tokenize_final=False)
        del instrument
    led.stage("S4 forward + SAE passthroughs (B=%d)" % B, sae_fwd)

    # S5 backward on model only
    def model_bwd(autocast_flag=False):
        config.discovery.autocast_bf16 = autocast_flag
        _, logits, _ = inference.forward(tokens[B], grad_enabled=True,
                                         return_activations=False,
                                         all_logits=True,
                                         tokenize_final=False)
        loss = logits.float().mean()
        loss.backward()
        config.discovery.autocast_bf16 = False
    led.stage("S5 model-only backward (B=%d)" % B, model_bwd)

    # S6 discovery-config backward: instrument first N sites, grads at anchors
    all_sites = [(l, k) for l in range(bank.n_layer) for k in bank.kinds
                 if bank.saes[k][l] is not None]

    def discovery_bwd(n_sites, batch, autocast_flag=False):
        sites = set(all_sites[:n_sites])
        seed_layer, seed_kind = all_sites[min(n_sites, len(all_sites) - 1)]
        sae = bank.saes[seed_kind][seed_layer]
        w_seed = sae.encoder.weight[SEED_LATENT].detach()
        b_seed = sae._get_bias_eff()[SEED_LATENT].detach()
        instrument = SAEGraphInstrument(bank)
        orig = instrument.transform
        seed_pre = []

        def tap(layer_idx, kd, x):
            if layer_idx == seed_layer and kd == seed_kind:
                seed_pre.append(x @ w_seed.to(x.device, x.dtype)
                                + b_seed.to(x.device, x.dtype))
                return x
            if (layer_idx, kd) not in sites:
                return x
            return orig(layer_idx, kd, x)

        instrument.transform = tap
        config.discovery.autocast_bf16 = autocast_flag
        inference.forward(tokens[batch], patcher=instrument, grad_enabled=True,
                          return_activations=False, tokenize_final=False)
        config.discovery.autocast_bf16 = False
        metric = seed_pre[0].float().mean()
        graph = instrument.graph
        anchors = [graph.get_latents(*s)[0].act
                   for s in sorted(graph.activations) if s in sites]
        if anchors:
            torch.autograd.grad(metric, anchors, allow_unused=True)
        instrument.release()   # deterministic teardown (mirrors production sites)
        del instrument, graph, anchors, seed_pre

    snapshot_target = (max(args.site_counts), max(args.batches))
    for n_sites in args.site_counts:
        if n_sites > len(all_sites):
            continue
        for batch in args.batches:
            do_snap = (args.snapshot
                       and (n_sites, batch) == snapshot_target)
            if do_snap:
                torch.cuda.memory._record_memory_history(max_entries=200000)
            led.stage("S6 discovery bwd (sites=%d, B=%d)" % (n_sites, batch),
                      lambda n=n_sites, b=batch: discovery_bwd(n, b))
            if do_snap:
                snap = out_dir / "s6_snapshot.pickle"
                torch.cuda.memory._dump_snapshot(str(snap))
                torch.cuda.memory._record_memory_history(enabled=None)
                print("  snapshot -> %s (view: pytorch.org/memory_viz)" % snap,
                      flush=True)

    if args.autocast:
        led.stage("S7 model-only backward AUTOCAST (B=%d)" % B,
                  lambda: model_bwd(True))
        for n_sites in args.site_counts:
            if n_sites > len(all_sites):
                continue
            for batch in args.batches:
                led.stage("S7 discovery bwd AUTOCAST (sites=%d, B=%d)"
                          % (n_sites, batch),
                          lambda n=n_sites, b=batch: discovery_bwd(n, b, True))

    out = out_dir / "ledger.json"
    out.write_text(json.dumps(led.rows, indent=1), encoding="utf-8")
    print("\nwrote %s" % out, flush=True)


if __name__ == "__main__":
    main()

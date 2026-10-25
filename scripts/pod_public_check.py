"""Public-model pod helpers for scripts/pod_bootstrap_public.sh (DAN-39 B7). Two subcommands:

  download   fetch (or, with --check-only, just verify in the local HF cache) every file the 070 adapters and the
             public SAE bank load: the HF checkpoints + tokenizers of EleutherAI/pythia-70m and gpt2, and every
             dictionary of model/families.py (Pythia: EleutherAI/sae-pythia-70m-32k, 18 sites; GPT-2:
             jbloom/GPT2-Small-OAI-v5-32k-{attn-out,resid-mid,mlp-out,resid-post}-SAEs, 48 sites, weights + cfg.json),
             plus the small wikitext test parquet the adapter tests tokenise (otherwise they fall back to random
             ids). The file lists come from FAMILIES[...].sae.files, i.e. exactly what the loaders open. Resumable
             (the HF cache skips complete files). Exit 1 if anything is missing afterwards.
  gpu        on --device (default cuda:0): the HF forward and every SAE encode, compared with the same on CPU
             (max |logit diff|; top-k index agreement per site). Prints a table; exit 1 on a load error, a site
             that cannot be loaded, or top-k agreement below --min-agree.

  PYTHONPATH=src .venv/bin/python scripts/pod_public_check.py download [--check-only] [--families pythia-70m ...]
  PYTHONPATH=src .venv/bin/python scripts/pod_public_check.py gpu [--device cuda:0]
"""
import argparse
import os
import sys
import time

PUBLIC = ("pythia-70m", "gpt2-small")
MODEL_FILES = ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
               "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]
WIKITEXT = ("Salesforce/wikitext", "wikitext-103-raw-v1/test-00000-of-00001.parquet")


def sae_files(fam):
    """{repo: [files]} for every (kind, layer) dictionary of a family, as the bank opens them."""
    out = {}
    for layer in range(fam.n_layer):
        for kind in fam.kinds:
            repo, weights, cfg = fam.sae.files(kind, layer)
            out.setdefault(repo, []).extend([weights] + ([cfg] if cfg else []))
    return out


def _cached(repo, filename, repo_type="model"):
    from huggingface_hub import try_to_load_from_cache
    p = try_to_load_from_cache(repo, filename, repo_type=repo_type)
    return p if isinstance(p, str) and os.path.exists(p) else None


def download(args):
    from huggingface_hub import list_repo_files, snapshot_download
    from model.families import FAMILIES
    want = []                                                     # (repo, repo_type, file, required)
    for name in args.families:
        fam = FAMILIES[name]
        model_files = MODEL_FILES
        if not args.check_only:
            present = set(list_repo_files(fam.hf_model_id))
            model_files = [f for f in MODEL_FILES if f in present]
        want += [(fam.hf_model_id, "model", f, f in ("config.json", "model.safetensors", "tokenizer.json",
                                                     "tokenizer_config.json")) for f in model_files]
        for repo, files in sae_files(fam).items():
            want += [(repo, "model", f, True) for f in files]
    want.append((WIKITEXT[0], "dataset", WIKITEXT[1], False))
    if not args.check_only:
        by_repo = {}
        for repo, rtype, f, _req in want:
            by_repo.setdefault((repo, rtype), []).append(f)
        for (repo, rtype), files in by_repo.items():
            t0 = time.time()
            snapshot_download(repo_id=repo, repo_type=rtype, allow_patterns=files, max_workers=args.workers)
            print("  %-55s %3d files  %.0f s" % (repo, len(files), time.time() - t0), flush=True)
    missing, total = [], 0
    for repo, rtype, f, req in want:
        p = _cached(repo, f, rtype)
        if p is None:
            if req:
                missing.append("%s/%s" % (repo, f))
            elif rtype == "dataset":
                print("  note: %s/%s not cached (the adapter tests then use random token ids)" % (repo, f))
            continue
        total += os.path.getsize(p)
    n_sae = {name: sum(len([x for x in fs if not x.endswith("cfg.json")])
                       for fs in sae_files(FAMILIES[name]).values()) for name in args.families}
    print("HF files: %d listed, %.2f GB in the cache; dictionaries per family: %s"
          % (len(want), total / 1e9, ", ".join("%s %d" % kv for kv in n_sae.items())))
    if missing:
        print("MISSING (%d):\n  %s" % (len(missing), "\n  ".join(missing)))
        return 1
    print("download check: OK")
    return 0


def gpu(args):
    import torch
    from model.families import FAMILIES
    from model.inference import Inference
    from sae.bank import SAEBank
    dev = torch.device(args.device)
    cpu = torch.device("cpu")
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("no CUDA device")
        return 1
    print("torch %s, CUDA %s, %d GPU(s)%s" % (
        torch.__version__, torch.version.cuda, torch.cuda.device_count(),
        (", " + torch.cuda.get_device_name(dev)) if dev.type == "cuda" else ""))
    bad = 0
    for name in args.families:
        fam = FAMILIES[name]
        g = torch.Generator().manual_seed(0)
        tokens = torch.randint(0, 50000, (2, 64), generator=g)
        t0 = time.time()
        with torch.no_grad():
            ref = Inference(cpu, family=name).model.hf(tokens).logits.float()
            inf = Inference(dev, family=name)
            out = inf.model.hf(tokens.to(dev)).logits.float().cpu()
        dlogit = (out - ref).abs().max().item()
        del inf
        bank_c = SAEBank(device=cpu, family=name, preload=False)
        bank_g = SAEBank(device=dev, family=name, preload=False)
        agree, n_sites, worst = [], 0, None
        for layer in range(fam.n_layer):
            for kind in fam.kinds:
                x = torch.randn(256, fam.d_model, generator=g)
                with torch.no_grad():
                    _, ic = bank_c.encode(x, kind, layer)
                    _, ig = bank_g.encode(x.to(dev), kind, layer)
                ic, ig = ic.cpu(), ig.cpu()
                k = ic.shape[-1]
                same = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(ic, ig)) / (k * ic.shape[0])
                agree.append(same)
                if worst is None or same < worst[0]:
                    worst = (same, "%d.%s" % (layer, kind))
                n_sites += 1
                bank_c.release(kind, layer)
                bank_g.release(kind, layer)
        mean = sum(agree) / len(agree)
        ok = n_sites == fam.n_layer * len(fam.kinds) and worst[0] >= args.min_agree
        bad += not ok
        print("%-11s %s | max |logit diff| vs CPU %.2e | %d SAE sites on %s, top-k agreement mean %.4f, worst %.4f "
              "(%s) | %.0f s | %s" % (name, fam.hf_model_id, dlogit, n_sites, dev, mean, worst[0], worst[1],
                                      time.time() - t0, "OK" if ok else "FAIL"), flush=True)
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--check-only", action="store_true", help="no network: verify the local HF cache only")
    d.add_argument("--families", nargs="+", default=list(PUBLIC), choices=list(PUBLIC))
    d.add_argument("--workers", type=int, default=8)
    gp = sub.add_parser("gpu")
    gp.add_argument("--device", default="cuda:0")
    gp.add_argument("--families", nargs="+", default=list(PUBLIC), choices=list(PUBLIC))
    gp.add_argument("--min-agree", type=float, default=0.95,
                    help="minimum mean share of top-k indices shared with CPU at every site (TF32 / reduction-order "
                         "ties may swap a few near-tied latents)")
    args = ap.parse_args(argv)
    return download(args) if args.cmd == "download" else gpu(args)


if __name__ == "__main__":
    sys.exit(main())

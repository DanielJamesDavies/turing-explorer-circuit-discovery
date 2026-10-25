"""TARGET FRAME for the public-model runs (DAN-9 / DAN-40 / DAN-41, pod plan B3): a SEEDED draw, written once and frozen.

Final frame (Daniel, 2026-10-24): 512 targets per model, seed 20261024 (chain.sh SEED), --frame fired.
  pythia-70m  32 per component x 16 = 512 (attn 0 and mlp 0 have no upstream SAE site: parallel residual)
  gpt2-small  512 over 47 sites = 10-11 per site (attn 0 has no upstream site); the 42 sites that get an 11th target
              are chosen by one seeded permutation (allocate)
Site eligibility is model.families.family_upstream_sites (the rule discovery.skip_no_upstream applies), so no SAE is
loaded.

The frame:
  --frame fired  (b, the reported frame) a latent is eligible iff, in the scanned corpus (the data pass's own pass-1
                 latent_stats.seq_count = the number of distinct firing 64-token windows; the corpus is exact-deduped),
                   it fires in >= --min-seqs (200) windows             (pre-stated; replaces the earlier >= 8)
                   and in no more than --hub-frac (50%) of the windows  (pre-stated HUB CAP: a latent that fires in
                                                                         most windows has almost no silent contrast)
                 The window total is the store's data_pass_manifest.json n_sequences (or --n-windows). The analogue of
                 TuringLLM's survival conditioning, to be stated as such; the manifest counts, per component, the
                 latents excluded by each rule (dead = 0 windows, below_min = 1..min-1, hub = above the cap)
  --frame all    (a) every latent of the dictionary (kept for the record; the fit side skips dead / rare latents)

Pre-drawn subsets (every one stratified as evenly as possible over the eligible components; a remainder goes to
components in a seeded order):
  pilot       30, DISJOINT from main (the next latents of each component's permutation), for the lambda pilot (DAN-42),
              so the main run stays confirmatory (DAN-8)
  nulls       128, a subset of main: the DAN-181 harder nulls (084) and the 073 resample subset (D11). Written
              SUBSAMPLE FIRST: its first 32 (a seeded stratified quarter) get the 2nd null draw (084 SUB_N = 32,
              SUB_DRAWS = 2); the manifest records nulls_sub_n
  stable      32, a subset of main (stable core)
  unweighted  128, a subset of main, Pythia only (the unweighted arm at the C3 lambda); not written for GPT-2
The alpha = 1 re-score (DAN-20) uses all of main.

Deterministic: every draw is numpy's PCG64 on SeedSequence([seed, stream, comp]), so the same family, frame, seed and
latent_stats give byte-identical files, and a component's main draw depends only on its own quota (not on the pilot or
subset sizes). This is NOT the data pass's candidates.pt (stratified_random: unseeded torch.randperm). Lists are
written in the 062 order (crc32 shuffle); the manifest records counts per component, the md5 of every list and of the
latent_stats input, and no time stamp or path. Existing lists that differ are never overwritten (frozen before any
fit); --force overrides.

  PYTHONPATH=src python experiments/082-public-runs/make_targets_public.py --family pythia-70m --frame fired \
      --seed 20261024 --stores outputs_public/pythia-70m
  -> experiments/082-public-runs/targets_<family>_{main,pilot,nulls,stable[,unweighted]}.txt + _manifest.json
"""
import argparse
import hashlib
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from model.families import FAMILIES, family_upstream_sites  # noqa: E402

N_MAIN = {"pythia-70m": 512, "gpt2-small": 512}               # Daniel, 2026-10-24
N_PILOT, N_STABLE, N_NULLS, N_NULLS_SUB = 30, 32, 128, 32
N_UNWEIGHTED = {"pythia-70m": 128, "gpt2-small": 0}          # the unweighted arm runs on Pythia only
MIN_SEQS = 200                                               # distinct firing windows (pre-stated eligibility)
HUB_FRAC = 0.5                                               # hub cap: exclude latents firing in > 50% of windows
SEED = 20261024
FRAMES = {"all": "(a) uniform over all latents",
          "fired": "(b) uniform over latents firing in >= min_seqs windows and in <= hub_frac of the windows"}
STREAM = {"main": 0, "pilot": 1, "unweighted": 2, "stable": 3, "nulls": 4, "nulls_sub": 5, "quota": 6}
SUBSETS = ("pilot", "unweighted", "stable", "nulls")         # pilot from main's remainder; the rest inside main


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rng(seed, *path):
    """One independent stream. SeedSequence pads short entropy with zeros, so [s, 2] and [s, 2, 0] would collide:
    per-component subset streams therefore use comp + 1, never 0."""
    return np.random.default_rng(np.random.SeedSequence([int(seed)] + [int(p) for p in path]))


def order(keys):
    """062's deterministic shuffle (make_targets.py), so every shard sees every depth."""
    return sorted(keys, key=lambda k: (zlib.crc32(k.encode()), k))


def components(family):
    """[(comp_idx, layer, kind, eligible)] in comp order; eligible = the site has at least one upstream SAE site."""
    fam = FAMILIES[family]
    return [(layer * len(fam.kinds) + j, layer, kind, bool(family_upstream_sites(fam, layer, kind)))
            for layer in range(fam.n_layer) for j, kind in enumerate(fam.kinds)]


def allocate(total, capacity, gen):
    """Spread `total` over the components as evenly as possible, never above a component's capacity. The remainder
    (and any capacity shortfall) goes to components in the order of one seeded permutation."""
    comps = sorted(capacity)
    alloc = {c: 0 for c in comps}
    tie = [comps[i] for i in gen.permutation(len(comps))]
    left = min(int(total), sum(capacity.values()))
    while left:
        open_ = [c for c in tie if alloc[c] < capacity[c]]
        share = left // len(open_)
        if share == 0:                                   # fewer left than open components: one each, in seeded order
            for c in open_[:left]:
                alloc[c] += 1
            break
        for c in open_:
            add = min(share, capacity[c] - alloc[c])
            alloc[c] += add
            left -= add
    return alloc


def eligibility(sc, frame, min_seqs, hub_frac, n_windows):
    """(eligible latent indices, exclusion counts) of one component. sc = its seq_count row (None for frame 'all'
    without stores). Hub = fires in more than hub_frac * n_windows windows (strictly more)."""
    if sc is None:
        return None, {}
    sc = np.asarray(sc)
    dead = sc == 0
    below = (sc > 0) & (sc < min_seqs)
    hub = (sc > hub_frac * n_windows) if n_windows else np.zeros_like(dead)
    counts = dict(dead=int(dead.sum()), below_min=int(below.sum()), hub=int((hub & ~dead & ~below).sum()))
    keep = ~(dead | below | hub)
    return np.flatnonzero(keep), counts


def draw(family, frame, seed, seq_count=None, n_windows=None, n_main=None, min_seqs=MIN_SEQS, hub_frac=HUB_FRAC,
         n_pilot=N_PILOT, n_stable=N_STABLE, n_nulls=N_NULLS, n_nulls_sub=N_NULLS_SUB, n_unweighted=None):
    """The frame draw. seq_count: [C, d_sae] firing-window counts (latent_stats.pt), required for frame 'fired'
    (with n_windows, the corpus window total, for the hub cap), optional for 'all' (then only used to count the
    expected skips). Returns ({list name: [keys]}, manifest)."""
    if family not in N_MAIN:
        raise ValueError("unknown public family %r; known: %s" % (family, sorted(N_MAIN)))
    if frame not in FRAMES:
        raise ValueError("frame must be one of %s" % sorted(FRAMES))
    fam = FAMILIES[family]
    comps = components(family)
    if seq_count is not None:
        seq_count = np.asarray(seq_count)
        if seq_count.shape != (len(comps), fam.sae.d_sae):
            raise ValueError("seq_count shape %s != grid (%d, %d)" % (seq_count.shape, len(comps), fam.sae.d_sae))
    elif frame == "fired":
        raise ValueError("frame 'fired' needs latent_stats (seq_count)")
    if frame == "fired" and not n_windows:
        raise ValueError("frame 'fired' needs n_windows (the corpus window total) for the hub cap")
    n_main = int(N_MAIN[family] if n_main is None else n_main)
    n_unw = int(N_UNWEIGHTED[family] if n_unweighted is None else n_unweighted)
    site = {c: "%d.%s" % (layer, kind) for c, layer, kind, _ in comps}
    elig = [c for c, _, _, ok in comps if ok]
    pools, per = {}, {}
    for c in elig:
        keep, excl = eligibility(None if seq_count is None else seq_count[c], frame, min_seqs, hub_frac, n_windows)
        pools[c] = np.arange(fam.sae.d_sae) if frame == "all" else keep
        per[c] = dict(comp_idx=c, site=site[c], frame=int(len(pools[c])))
        if excl:
            per[c]["excluded"] = excl
    quota = allocate(n_main, {c: int(len(pools[c])) for c in elig}, rng(seed, STREAM["quota"]))
    even = n_main // len(elig)
    main, rest = {}, {}
    for c in elig:
        perm = rng(seed, STREAM["main"], c).permutation(pools[c])
        q = quota[c]
        main[c], rest[c] = sorted(int(i) for i in perm[:q]), [int(i) for i in perm[q:]]
        per[c].update(quota=q, main=len(main[c]), short=max(0, even - len(main[c])))
        if seq_count is not None:
            sc = seq_count[c][main[c]] if main[c] else np.zeros(0)
            per[c]["main_below_min_seqs"] = int((sc < min_seqs).sum())
            if n_windows:
                per[c]["main_above_hub_cap"] = int((sc > hub_frac * n_windows).sum())
    lists = {"main": order([("%s.%d" % (site[c], i)) for c in elig for i in main[c]])}
    want = {"pilot": n_pilot, "unweighted": n_unw, "stable": n_stable, "nulls": n_nulls}
    picks = {}
    for name in SUBSETS:
        if not want[name]:
            continue
        cap = {c: len(rest[c]) if name == "pilot" else len(main[c]) for c in elig}
        alloc = allocate(want[name], cap, rng(seed, STREAM[name]))
        for c in elig:
            if name == "pilot":                                  # the next latents of the same permutation: disjoint
                pick = sorted(rest[c][:alloc[c]])
            else:
                pick = sorted(int(i) for i in rng(seed, STREAM[name], c + 1).choice(main[c], size=alloc[c], replace=False))
            per[c][name] = len(pick)
            picks[(name, c)] = pick
        if name != "nulls":
            lists[name] = order(["%s.%d" % (site[c], i) for c in elig for i in picks[(name, c)]])
    # nulls: its own stratified subsample (the 2nd-draw targets) is written first
    sub_n = 0
    if n_nulls:
        alloc_sub = allocate(n_nulls_sub, {c: len(picks[("nulls", c)]) for c in elig}, rng(seed, STREAM["nulls_sub"]))
        sub, rest_n = [], []
        for c in elig:
            perm = [int(i) for i in rng(seed, STREAM["nulls_sub"], c + 1).permutation(picks[("nulls", c)])]
            sub += ["%s.%d" % (site[c], i) for i in sorted(perm[:alloc_sub[c]])]
            rest_n += ["%s.%d" % (site[c], i) for i in sorted(perm[alloc_sub[c]:])]
            per[c]["nulls_sub"] = alloc_sub[c]
        lists["nulls"] = order(sub) + order(rest_n)
        sub_n = len(sub)
    excl_tot = {}
    for c in elig:
        for k, v in per[c].get("excluded", {}).items():
            excl_tot[k] = excl_tot.get(k, 0) + v
    manifest = dict(
        family=family, frame=frame, frame_option=FRAMES[frame], seed=int(seed), n_main_requested=n_main,
        min_seqs=int(min_seqs), hub_frac=float(hub_frac), n_windows=None if n_windows is None else int(n_windows),
        hub_cap_windows=None if not n_windows else float(hub_frac * n_windows),
        d_sae=int(fam.sae.d_sae), n_layer=int(fam.n_layer), kinds=list(fam.kinds),
        subset_sizes_requested=dict(want, nulls_sub=n_nulls_sub), nulls_sub_n=sub_n,
        rng="numpy PCG64 on SeedSequence([seed, stream, comp]); streams %s (subset choice uses comp + 1)" % STREAM,
        eligible=[site[c] for c in elig],
        excluded={site[c]: "no upstream SAE site" for c, _, _, ok in comps if not ok},
        latents_excluded=excl_tot,
        counts={k: len(v) for k, v in lists.items()},
        components=[per[c] for c in elig],
        shortfalls={site[c]: per[c]["short"] for c in elig if per[c]["short"]},
        relations=dict(pilot_disjoint_from_main=True, nulls_subset_of_main=True, stable_subset_of_main=True,
                       unweighted_subset_of_main=bool(n_unw), nulls_order="the first nulls_sub_n get the 2nd draw",
                       resample_073_uses="nulls", alpha1_uses="main"),
    )
    if seq_count is not None:
        manifest["main_below_min_seqs"] = sum(per[c]["main_below_min_seqs"] for c in elig)
        if n_windows:
            manifest["main_above_hub_cap"] = sum(per[c]["main_above_hub_cap"] for c in elig)
    return lists, manifest


def write(out_dir, family, lists, manifest, force=False):
    """Write the lists + manifest. Refuses to change an existing frozen list unless force (identical = no-op)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = {name: "\n".join(keys) + "\n" for name, keys in lists.items()}
    paths = {name: out_dir / ("targets_%s_%s.txt" % (family, name)) for name in texts}
    changed = [str(p) for name, p in paths.items() if p.exists() and p.read_bytes() != texts[name].encode()]
    if changed and not force:
        raise SystemExit("refusing to overwrite frozen target lists that differ (use --force): %s" % changed)
    for name, p in paths.items():
        with open(p, "w", newline="\n", encoding="utf-8") as fh:
            fh.write(texts[name])
    manifest = dict(manifest, files={p.name: md5(p) for p in paths.values()})
    mpath = out_dir / ("targets_%s_manifest.json" % family)
    with open(mpath, "w", newline="\n", encoding="utf-8") as fh:
        fh.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return paths, mpath


def load_seq_count(stores):
    import torch
    path = os.path.join(stores, "latent_stats.pt")
    ls = torch.load(path, map_location="cpu", weights_only=False)
    return ls["seq_count"].numpy(), md5(path)


def load_n_windows(stores):
    """The scanned corpus's window total: data_pass_manifest.json n_sequences (written by run_data_pass.py)."""
    p = os.path.join(stores, "data_pass_manifest.json")
    if not os.path.exists(p):
        return None
    return int(json.load(open(p))["n_sequences"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--family", required=True, choices=sorted(N_MAIN))
    ap.add_argument("--frame", required=True, choices=sorted(FRAMES), help="fired = the reported frame (b)")
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--stores", help="outputs_public/<family> (latent_stats.pt, data_pass_manifest.json)")
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--n-windows", type=int, help="corpus window total (default: the store's data_pass_manifest)")
    ap.add_argument("--n-main", type=int)
    ap.add_argument("--min-seqs", type=int, default=MIN_SEQS)
    ap.add_argument("--hub-frac", type=float, default=HUB_FRAC)
    ap.add_argument("--n-pilot", type=int, default=N_PILOT)
    ap.add_argument("--n-stable", type=int, default=N_STABLE)
    ap.add_argument("--n-nulls", type=int, default=N_NULLS)
    ap.add_argument("--n-nulls-sub", type=int, default=N_NULLS_SUB)
    ap.add_argument("--n-unweighted", type=int)
    ap.add_argument("--force", action="store_true", help="overwrite frozen lists that differ")
    a = ap.parse_args(argv)
    seq_count, ls_md5, n_win = None, None, a.n_windows
    if a.stores:
        seq_count, ls_md5 = load_seq_count(a.stores)
        n_win = n_win or load_n_windows(a.stores)
    elif a.frame == "fired":
        ap.error("--frame fired needs --stores (latent_stats.pt from pass 1)")
    if a.frame == "fired" and not n_win:
        ap.error("--frame fired needs the window total (data_pass_manifest.json in --stores, or --n-windows)")
    lists, manifest = draw(a.family, a.frame, a.seed, seq_count, n_win, a.n_main, a.min_seqs, a.hub_frac, a.n_pilot,
                           a.n_stable, a.n_nulls, a.n_nulls_sub, a.n_unweighted)
    manifest["latent_stats_md5"] = ls_md5
    paths, mpath = write(a.out, a.family, lists, manifest, a.force)
    print("%s, frame %s, seed %d: %s" % (a.family, FRAMES[a.frame], a.seed,
                                         ", ".join("%s %d" % kv for kv in manifest["counts"].items())))
    print("excluded (no upstream SAE site): %s" % ", ".join(manifest["excluded"]))
    if manifest["latents_excluded"]:
        print("latents excluded from the frame (%s windows total; >= %d windows, hub cap %.0f%%): %s" % (
            n_win, a.min_seqs, 100 * a.hub_frac, manifest["latents_excluded"]))
        for c in manifest["components"]:
            print("  %-12s frame %6d | dead %6d below_min %6d hub %5d | main %d" % (
                c["site"], c["frame"], c["excluded"]["dead"], c["excluded"]["below_min"], c["excluded"]["hub"],
                c["main"]))
    if manifest["shortfalls"]:
        print("WARNING: components short of the even quota: %s" % manifest["shortfalls"])
    if "main_below_min_seqs" in manifest:
        print("main targets below %d firing windows (expected skips under frame all): %d" % (
            a.min_seqs, manifest["main_below_min_seqs"]))
    for p in list(paths.values()) + [mpath]:
        print("wrote", p)


if __name__ == "__main__":
    main()

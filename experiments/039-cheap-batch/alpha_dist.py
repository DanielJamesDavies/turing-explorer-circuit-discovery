"""CHEAP BATCH A: what do the fitted amplitudes actually look like?

Daniel's holiday question 1 ("what is the amplification distribution?"),
which feeds question 2 (negative amplification -- alpha < 1 mass is the
attenuation story) and question 3 (are any members held at alpha ~ 0?).

Sources with FULL per-member alphas:
  * Llama TopK skip-transcoders: ours_llama_members.jsonl (triamp400)
  * GemmaScope JumpReLU SAEs: ours_gemma_members_t2.jsonl
    (triamp400 AND coact_amp -- the baseline's alphas for contrast)
The home bank never dumped per-member alphas; its logged summary
(alpha_p90 ~ 2.0-2.7, alpha_max 3.5-12.3 across coact-era fits) is
quoted for completeness but not histogrammed.

Also folds in Daniel's question 5 (circuit selectivity): members as a
fraction of the live pool, from the logged anchor_support rows.

  python alpha_dist.py     (pure CPU, reads jsonl only)
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
D = HERE.parent
SRC = {
    ("llama-topk", "triamp400"):
        D / "035-transcoder-compare/ours_llama_members.jsonl",
    ("gemma-jumprelu", "triamp400"):
        D / "037-gemmascope/ours_gemma_members_t2.jsonl",
    ("gemma-jumprelu", "coact_amp"):
        D / "037-gemmascope/ours_gemma_members_t2.jsonl",
}
ROWS = {
    "llama-topk": D / "035-transcoder-compare/ours_llama_rows.jsonl",
    "gemma-jumprelu": D / "037-gemmascope/ours_gemma_rows_t2.jsonl",
}


def quantile(xs, q):
    xs = sorted(xs)
    i = q * (len(xs) - 1)
    lo = int(i)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (i - lo)


def main():
    print("%-16s %-10s %6s | %6s %6s %6s %6s | %6s %6s %6s %6s %6s"
          % ("bank", "arm", "n_all", "p10", "med", "p90", "max",
             "<0.1", "<0.5", "<1", "0.9-1.1", ">2"))
    for (bank, arm), path in SRC.items():
        alphas = []
        per_seed = []
        for line in open(path):
            r = json.loads(line)
            if r.get("arm", "triamp400") != arm:
                continue
            a = [float(v) for d in r["alphas"].values() for v in d.values()]
            alphas += a
            per_seed.append((r["layer"], r["latent"], a))
        if not alphas:
            continue
        n = len(alphas)
        f = lambda pred: 100.0 * sum(pred(x) for x in alphas) / n
        print("%-16s %-10s %6d | %6.2f %6.2f %6.2f %6.2f | %5.1f%% %5.1f%% "
              "%5.1f%% %5.1f%% %5.1f%%"
              % (bank, arm, n, quantile(alphas, .1), quantile(alphas, .5),
                 quantile(alphas, .9), max(alphas),
                 f(lambda x: x < 0.1), f(lambda x: x < 0.5),
                 f(lambda x: x < 1.0),
                 f(lambda x: 0.9 <= x <= 1.1), f(lambda x: x > 2.0)))
    print("\n(home bank: no per-member dump; logged summaries alpha_p90 "
          "2.0-2.7, alpha_max 3.5-12.3)")

    print("\nSELECTIVITY (question 5): circuit vs the live pool")
    print("%-16s  seed          n   live_pool   in-circuit%%   excluded%%"
          % "bank")
    for bank, path in ROWS.items():
        rows = [json.loads(l) for l in open(path)]
        pool = {(r["layer"], r["latent"]): r["n"] for r in rows
                if r.get("arm") == "anchor_support"}
        for r in rows:
            if r.get("arm") != "triamp400":
                continue
            k = (r["layer"], r["latent"])
            if k not in pool:
                continue
            pct = 100.0 * r["n"] / pool[k]
            print("%-16s  L%-2d %-8d %5d %10d %10.3f%% %10.2f%%"
                  % (bank, k[0], k[1], r["n"], pool[k], pct, 100 - pct))


if __name__ == "__main__":
    main()

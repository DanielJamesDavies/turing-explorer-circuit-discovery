"""Verify the pipeline smoke's stored circuits carry tri-amp
amplitudes (the persistence fix) and print the acceptance metrics."""
import sys

import torch

cs = torch.load("outputs/circuits/discovered_circuits.shard0.pt",
                weights_only=False)
print(len(cs), "circuits stored")
ok = True
for c in cs.values():
    md = c.metadata
    amps = [n.metadata["amplitude"] for n in c.nodes.values()
            if "amplitude" in n.metadata]
    non_seed = [n for n in c.nodes.values()
                if n.metadata.get("role") != "seed"]
    print("seed c%s/%s | %s | nodes %d (non-seed %d) | with-amp %d | "
          "amp %.3f..%.3f | amp_stats %s | cf %.3f sup %.3f"
          % (md.get("seed_comp"), md.get("seed_latent"),
             md.get("discovery_method"), len(c.nodes), len(non_seed),
             len(amps), min(amps) if amps else -1,
             max(amps) if amps else -1,
             {k: round(v, 3) for k, v in
              (md.get("amp_stats") or {}).items()},
             md.get("counterfactual_faithfulness", -9),
             md.get("posctx_suppression_score", -9)))
    if len(amps) < len(non_seed):
        ok = False
print("AMPLITUDES:", "OK (every non-seed node carries one)" if ok
      else "MISSING on some nodes")
sys.exit(0 if ok else 1)

import json
import os
import sys
from typing import List, Dict, Any
from pipeline.component_index import split_component_idx

def list_discovered_circuits(summary_path: str = "outputs/circuits/summary.json"):
    if not os.path.exists(summary_path):
        print(f"Error: Summary file not found at {summary_path}. Have you run the discovery window yet?")
        return

    with open(summary_path, "r") as f:
        circuits: List[Dict[str, Any]] = json.load(f)

    if not circuits:
        print("No circuits found in the summary file.")
        return

    # Sort by faithfulness descending
    circuits.sort(key=lambda x: x.get("metadata", {}).get("faithfulness", 0.0), reverse=True)

    print(f"\nDiscovered Circuits ({len(circuits)} total):")
    print("-" * 120)
    print(f"{'Name':<25} | {'Faith.':<7} | {'Suff.':<7} | {'Comp.':<7} | {'Nodes':<5} | {'Edges':<5} | {'Seed (L/I)':<15} | {'Method'}")
    print("-" * 120)

    for c in circuits:
        name = c.get("name", "Unknown")
        meta = c.get("metadata", {})
        faith = meta.get("faithfulness", 0.0)
        suff = meta.get("sufficiency", 0.0)
        comp = meta.get("completeness", 0.0)
        nodes = c.get("nodes", 0)
        edges = c.get("edges", 0)
        method = meta.get("discovery_method", "Unknown")
        
        seed_l, _ = split_component_idx(meta.get("seed_comp", 0), 3)
        seed_i = meta.get("seed_latent", 0)
        seed_str = f"L{seed_l:02d} / {seed_i:5d}"

        print(f"{name:<25} | {faith:7.3f} | {suff:7.3f} | {comp:7.3f} | {nodes:<5} | {edges:<5} | {seed_str:<15} | {method}")
    print("-" * 120)

if __name__ == "__main__":
    path = "outputs/circuits/summary.json"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    list_discovered_circuits(path)

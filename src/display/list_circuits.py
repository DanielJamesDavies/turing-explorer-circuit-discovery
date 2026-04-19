import json
import os
import sys
from collections import defaultdict
from typing import List, Dict, Any
from pipeline.component_index import split_component_idx

def _flatten_evals(evals: Dict[str, Any]) -> Dict[str, float]:
    """Flatten an evals dict into a single-level {label: value} mapping."""
    flat: Dict[str, float] = {}
    for key, val in evals.items():
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                if isinstance(sub_val, (int, float)):
                    flat[f"{key}.{sub_key}"] = float(sub_val)
        elif isinstance(val, (int, float)):
            flat[key] = float(val)
    return flat


def _print_eval_stats(circuits: List[Dict[str, Any]]) -> None:
    """Compute and print mean + variance for every eval field across all circuits."""
    from rich.console import Console
    from rich.table import Table
    from rich import box

    buckets: Dict[str, List[float]] = defaultdict(list)

    for c in circuits:
        evals = c.get("metadata", {}).get("evals", {})
        for label, value in _flatten_evals(evals).items():
            buckets[label].append(value)

    if not buckets:
        print("(no eval data found)\n")
        return

    table = Table(
        title="Eval Stats",
        box=box.ROUNDED,
        show_lines=False,
        header_style="bold cyan",
        title_style="bold white",
    )
    table.add_column("Eval",     style="magenta", no_wrap=True)
    table.add_column("Mean",     justify="right", style="green")
    table.add_column("Variance", justify="right", style="yellow")
    table.add_column("N",        justify="right", style="dim")

    for label in sorted(buckets):
        vals = buckets[label]
        n = len(vals)
        mean = sum(vals) / n
        variance = sum((v - mean) ** 2 for v in vals) / n if n > 1 else 0.0
        table.add_row(label, f"{mean:.4f}", f"{variance:.4f}", str(n))

    Console().print(table)


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

    _print_eval_stats(circuits)

if __name__ == "__main__":
    path = "outputs/circuits/summary.json"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    list_discovered_circuits(path)

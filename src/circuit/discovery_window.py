import torch
import os
import time
from typing import List, Dict, Any
from tqdm import tqdm

from model.inference import Inference
from sae.bank import SAEBank
from data.loader import DataLoader
from config import config
from store.circuits import circuit_store
from store.latent_stats import latent_stats
from store.top_coactivation import top_coactivation

from circuit.probe_dataset import ProbeDatasetBuilder
from circuit.discovery.base import DiscoveryMethod
from circuit.discovery.coactivation_statistical import CoactivationStatistical
from circuit.discovery.logit_attribution import LogitAttribution
from circuit.discovery.sfc_attribution_patching import SFCAttributionPatching
from circuit.discovery.neighborhood_expansion import NeighborhoodExpansion
from circuit.discovery.top_coactivation import TopCoactivationDiscovery
from circuit.discovery.top_coact_expansion.mlp_top_coact_sparse_expansion import MlpTopCoactSparseExpansion
from circuit.discovery.top_coact_expansion.attn_top_coact_sparse_expansion import AttnTopCoactSparseExpansion
from circuit.discovery.top_coact_expansion.resid_top_coact_sparse_expansion import ResidTopCoactSparseExpansion
from circuit.discovery.top_coact_expansion.attn_mlp_top_coact_sparse_expansion import AttnMlpTopCoactSparseExpansion
from circuit.discovery.top_coact_expansion.attn_resid_top_coact_sparse_expansion import AttnResidTopCoactSparseExpansion
from circuit.discovery.top_coact_expansion.mlp_resid_top_coact_sparse_expansion import MlpResidTopCoactSparseExpansion
from circuit.discovery.top_coact_expansion.all_top_coact_sparse_expansion import AllTopCoactSparseExpansion
from circuit.discovery.top_coact_expansion.hard_negative_coact_sparse_expansion import HardNegativeCoactSparseExpansion
from circuit.discovery.differential_activation import DifferentialActivation


METHOD_REGISTRY: Dict[str, type[DiscoveryMethod]] = {
    "coactivation_statistical": CoactivationStatistical,
    "logit_attribution": LogitAttribution,
    "sfc_attribution_patching": SFCAttributionPatching,
    "neighborhood_expansion": NeighborhoodExpansion,
    "top_coactivation": TopCoactivationDiscovery,
    "mlp_top_coact_sparse_expansion": MlpTopCoactSparseExpansion,
    "attn_top_coact_sparse_expansion": AttnTopCoactSparseExpansion,
    "resid_top_coact_sparse_expansion": ResidTopCoactSparseExpansion,
    "attn_mlp_top_coact_sparse_expansion": AttnMlpTopCoactSparseExpansion,
    "attn_resid_top_coact_sparse_expansion": AttnResidTopCoactSparseExpansion,
    "mlp_resid_top_coact_sparse_expansion": MlpResidTopCoactSparseExpansion,
    "all_top_coact_sparse_expansion": AllTopCoactSparseExpansion,
    "hard_negative_coact_sparse_expansion": HardNegativeCoactSparseExpansion,
    "differential_activation": DifferentialActivation,
}


def _build_methods(
    inference: Inference,
    bank: SAEBank,
    avg_acts: torch.Tensor,
    probe_builder: ProbeDatasetBuilder,
) -> List[DiscoveryMethod]:
    """
    Instantiates all discovery methods listed in config.discovery.methods.

    Supported method names:
      "coactivation_statistical"  — fast, no gradients, statistical baseline
      "logit_attribution"         — two-pass gradient method (recommended)
      "sfc_attribution_patching"  — SFC-style delta×gradient node attribution + Jacobian edges
      "neighborhood_expansion"    — two-hop co-activation neighbourhood expansion (no gradients)
      "top_coactivation"          — legacy feature-to-feature attribution (broken cross-layer)
      "mlp_top_coact_sparse_expansion"      — MLP-only two-hop expansion + full attn/resid passthrough
      "attn_top_coact_sparse_expansion"     — attn-only two-hop expansion + full MLP/resid passthrough
      "resid_top_coact_sparse_expansion"    — resid-only two-hop expansion + full attn/MLP passthrough
      "attn_mlp_top_coact_sparse_expansion" — attn+mlp expansion + full resid passthrough
      "attn_resid_top_coact_sparse_expansion" — attn+resid expansion + full mlp passthrough
      "mlp_resid_top_coact_sparse_expansion" — mlp+resid expansion + full attn passthrough
      "all_top_coact_sparse_expansion"      — all-kinds expansion (attn/mlp/resid), no passthrough
      "hard_negative_coact_sparse_expansion" — all-kinds expansion + hard-negative inhibitor search
      "differential_activation"              — pos vs neg differential activation + causal attribution
    """
    enabled_raw = config.discovery.methods
    if isinstance(enabled_raw, list):
        enabled: List[str] = enabled_raw
    elif isinstance(enabled_raw, tuple):
        enabled = list(enabled_raw)
    else:
        enabled = []
    if not enabled:
        # Default to both main methods if config key is missing
        enabled = ["coactivation_statistical", "logit_attribution"]

    methods: List[DiscoveryMethod] = []
    for name in enabled:
        method_cls = METHOD_REGISTRY.get(name)
        if method_cls is None:
            print(f"[DiscoveryWindow] Warning: unknown discovery method '{name}' — skipped.")
            continue

        methods.append(method_cls(inference, bank, avg_acts, probe_builder))

    return methods


class DiscoveryWindow:
    """
    Orchestrates circuit discovery for a list of seed candidates.

    Runs all enabled discovery methods for every seed and stores every circuit that
    passes the faithfulness threshold, tagged with its source method in metadata.
    Multiple methods may find different circuits from the same seed.
    """

    def __init__(
        self,
        inference: Inference,
        bank: SAEBank,
        loader: DataLoader,
        output_dir: str = "outputs/circuits",
    ):
        self.inference = inference
        self.bank = bank
        self.loader = loader
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Zero-ablation baseline: set all non-circuit latents to 0 during patching.
        # Using latent_stats.mean_seq (conditional mean over sequences where each
        # latent fires) produced baselines too close to the original on positive
        # contexts — the same sequences where the seed fires — collapsing the
        # faithfulness denominator.  Zero ablation is an unambiguous counterfactual:
        # "what would the model do if these SAE features contributed nothing?"
        n_components = bank.n_layer * len(bank.kinds)
        self.avg_acts = torch.zeros(
            (n_components, bank.d_sae), dtype=torch.float32, device=bank.device
        )
        self.probe_builder = ProbeDatasetBuilder(inference, bank, loader)
        self.methods = _build_methods(inference, bank, self.avg_acts, self.probe_builder)

        method_names = [type(m).__name__ for m in self.methods]
        print(f"[DiscoveryWindow] Active methods: {method_names}")

    def run(self, candidates: List[Dict[str, Any]], save_interval: int = 10):
        """Runs all discovery methods for each seed candidate."""
        print(f"--- Starting Discovery Window: {len(candidates)} candidates × {len(self.methods)} method(s) ---")

        discovered_count = 0

        pbar = tqdm(candidates, desc="Discovering Circuits")
        for cand in pbar:
            comp_idx = cand["comp_idx"]
            latent_idx = cand["latent_idx"]

            for method in self.methods:
                m_t0 = time.perf_counter()
                circuit = method.discover(comp_idx, latent_idx)
                m_dt = time.perf_counter() - m_t0
                
                # If a method is slow (>1s), log its duration to the console for observability
                if m_dt > 1.0:
                    from utils.observability import obs
                    pbar.write(f"  - {method.method_name}: {m_dt:.2f}s ({obs.attempt_forward_passes} forwards)")
                
                if circuit:
                    discovered_count += 1
                    circuit_store.add_circuit(circuit)
                    pbar.set_postfix({"found": discovered_count})

                    if discovered_count % save_interval == 0:
                        self.save_store()

        self.save_store()
        
        from utils.observability import obs
        print(f"Discovery Window complete. Found {discovered_count} faithful circuits.")
        print(f"Total Forward Passes: {obs.forward_passes}")
        print(f"Total Forward Time: {obs.total_forward_time:.2f} s")
        if obs.forward_passes > 0:
            print(f"Average Forward Duration: {obs.total_forward_time / obs.forward_passes * 1000:.1f} ms")
        print("")
        self._print_summary_table()

    def _print_summary_table(self):
        """Prints a Rich-formatted table of all discovered circuits sorted by faithfulness."""
        from rich.console import Console
        from rich.table import Table
        from rich import box

        circuits = list(circuit_store.circuits.values())
        if not circuits:
            return

        rows = []
        for c in circuits:
            m = c.metadata
            rows.append({
                "method":      m.get("discovery_method", "unknown"),
                "seed_comp":   str(m.get("seed_comp",   "?")),
                "seed_latent": str(m.get("seed_latent", "?")),
                "nodes":       len(c.nodes),
                "edges":       len(c.edges),
                "faith":       m.get("faithfulness", float("nan")),
                "suff":        m.get("sufficiency",  float("nan")),
                "comp":        m.get("completeness", float("nan")),
            })
        rows.sort(key=lambda r: r["faith"], reverse=True)

        table = Table(
            title="Discovered Circuits",
            box=box.ROUNDED,
            show_lines=False,
            header_style="bold cyan",
            title_style="bold white",
        )
        table.add_column("Method",  style="magenta",    no_wrap=True)
        table.add_column("Comp",    justify="right",    style="dim")
        table.add_column("Latent",  justify="right",    style="dim")
        table.add_column("Nodes",   justify="right")
        table.add_column("Edges",   justify="right")
        table.add_column("Faith",   justify="right",    style="green")
        table.add_column("Suff",    justify="right",    style="yellow")
        table.add_column("Compl",   justify="right",    style="blue")

        for r in rows:
            faith = r["faith"]
            faith_str = f"{faith:.4f}"
            # Dim rows with low faithfulness so stand-out circuits pop visually
            row_style = "" if faith >= 0.5 else "dim"
            table.add_row(
                r["method"],
                r["seed_comp"],
                r["seed_latent"],
                str(r["nodes"]),
                str(r["edges"]),
                faith_str,
                f"{r['suff']:.4f}",
                f"{r['comp']:.4f}",
                style=row_style,
            )

        Console().print(table)

    def save_store(self):
        """Persists the circuit store to disk."""
        path = os.path.join(self.output_dir, "discovered_circuits.pt")
        circuit_store.save(path)
        self._save_summary()

    def _save_summary(self):
        """Saves a JSON summary of all discovered circuits."""
        import json
        summary = []
        for _, circuit in circuit_store.circuits.items():
            summary.append({
                "name": circuit.name,
                "uuid": circuit.uuid,
                "nodes": len(circuit.nodes),
                "edges": len(circuit.edges),
                "metadata": circuit.metadata,
            })

        path = os.path.join(self.output_dir, "summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)


def run_discovery_window(
    inference: Inference,
    bank: SAEBank,
    loader: DataLoader,
    candidates_path: str = "outputs/candidates.pt",
):
    """Entry point to run a discovery window from saved candidates."""
    if not os.path.exists(candidates_path):
        print(f"Error: Candidates file not found at {candidates_path}. Run candidate selection first.")
        return

    if not latent_stats._allocated:
        latent_stats.load("outputs/latent_stats.pt")
    if not top_coactivation._allocated:
        top_coactivation.load("outputs/top_coactivation.pt")

    from store.context import neg_ctx
    if not neg_ctx._allocated:
        neg_ctx.load("outputs/neg_ctx.pt")

    candidates = torch.load(candidates_path, weights_only=False)

    window = DiscoveryWindow(inference, bank, loader)
    window.run(candidates)

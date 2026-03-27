import os
import torch
import argparse
from typing import Optional, List, Dict, Tuple

from data.loader import DataLoader
from model.inference import Inference
from sae.bank import SAEBank
from store.context import top_ctx, mid_ctx, neg_ctx
from store.logit_context import logit_ctx
from store.top_coactivation import top_coactivation
from store.circuits import Circuit, CircuitNode
from circuit.feature_id import FeatureID
from circuit.discovery.sfc_attribution_patching import SFCAttributionPatching
from circuit.probe_dataset import ProbeDatasetBuilder, ProbeDataset
from circuit.patcher import CircuitPatcher
from pipeline.component_index import component_idx as build_component_idx, split_component_idx
from display.display import display
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from tqdm import tqdm

VALID_KINDS = ["attn", "mlp", "resid"]

def _try_load(ctx, path: str, name: str) -> None:
    """Load a context store if the file exists; print a notice otherwise."""
    if os.path.exists(path):
        ctx.load(path)
        print(f"  loaded {name}")
    else:
        print(f"  {name} not found at {path} — skipping")

def parse_latent_input(raw: str):
    """Parse '<layer> <kind> <latent>' from a string. Returns (layer, kind, latent) or None."""
    parts = raw.strip().split()
    if len(parts) != 3:
        return None
    try:
        layer = int(parts[0])
    except ValueError:
        return None
    kind = parts[1].lower()
    if kind not in VALID_KINDS:
        return None
    try:
        latent = int(parts[2])
    except ValueError:
        return None
    return layer, kind, latent

class AblationSensitivityTool:
    def __init__(self, device: torch.device):
        self.console = Console()
        self.device = device
        
        self.console.print("[bold yellow]Loading stores...[/bold yellow]")
        top_ctx.load("outputs/top_ctx.pt")
        logit_ctx.load("outputs/logit_ctx.pt")
        top_coactivation.load("outputs/top_coactivation.pt")
        _try_load(mid_ctx, "outputs/mid_ctx.pt", "mid_ctx")
        _try_load(neg_ctx, "outputs/neg_ctx.pt", "neg_ctx")

        self.console.print("[bold yellow]Initializing components...[/bold yellow]")
        self.loader = DataLoader(device=device)
        self.model = Inference(device=device)
        self.bank = SAEBank(device=device, load_decoders=True) # Need decoders for ablation
        
        # We need avg_acts for the discovery and patcher
        from store.latent_stats import latent_stats
        latent_stats.load("outputs/latent_stats.pt")
        self.avg_acts = latent_stats.mean # [n_components, d_sae]

        self.probe_builder = ProbeDatasetBuilder(self.model, self.bank, self.loader)
        
    def discover_candidates(self, layer_idx: int, kind: str, latent_idx: int) -> Optional[List[FeatureID]]:
        """Candidate Discovery using Attribution, Co-activation, and Frequency Analysis."""
        self.console.print(f"\n[bold cyan]Discovering neighborhood for {layer_idx} {kind} {latent_idx}...[/bold cyan]")
        
        comp_idx = build_component_idx(layer_idx, VALID_KINDS.index(kind), len(VALID_KINDS))
        seed_fid = FeatureID(layer_idx, kind, latent_idx)
        
        # 1. Get Attribution-based candidates (Downstream / High Impact)
        sfc = SFCAttributionPatching(
            inference=self.model,
            sae_bank=self.bank,
            avg_acts=self.avg_acts,
            probe_builder=self.probe_builder,
            node_threshold=0.05,
            min_faithfulness=-1.0,
        )
        
        circuit = sfc.discover(comp_idx, latent_idx)
        attr_candidates = []
        if circuit:
            for node in circuit.nodes.values():
                fid = node.metadata.get("feature_id")
                if fid and isinstance(fid, FeatureID):
                    score = abs(node.metadata.get("effect_score", 0.0))
                    attr_candidates.append((fid, score))
        
        # 2. Get Co-activation-based candidates (Upstream / Peers)
        # top_coactivation stores top-K for every latent
        co_ids = top_coactivation.top_indices[comp_idx, latent_idx]
        co_vals = top_coactivation.top_values[comp_idx, latent_idx]
        
        co_candidates = []
        d_sae = self.bank.d_sae
        for i in range(co_ids.shape[0]):
            gid = int(co_ids[i].item())
            val = float(co_vals[i].item())
            if gid == 0 and val == 0: continue
            
            c_idx = gid // d_sae
            l_idx = gid % d_sae
            layer, k_idx = split_component_idx(c_idx, len(self.bank.kinds))
            fid = FeatureID(layer, self.bank.kinds[k_idx], l_idx)
            co_candidates.append((fid, val))

        # 3. Combine and filter
        # We want a mix of both. Let's take top 8 from each.
        attr_candidates.sort(key=lambda x: x[1], reverse=True)
        co_candidates.sort(key=lambda x: x[1], reverse=True)
        
        combined_set = {seed_fid}
        final_list = [seed_fid]
        
        def add_from_list(src_list, limit):
            added = 0
            for fid, _ in src_list:
                if fid not in combined_set:
                    combined_set.add(fid)
                    final_list.append(fid)
                    added += 1
                if added >= limit: break

        add_from_list(attr_candidates, 8)
        add_from_list(co_candidates, 8)

        # 4. High-frequency co-firing (Scan all 1.1M latents)
        # Build probe dataset to get tokens for the frequency scan
        probe_data = self.probe_builder.build_for_latent(comp_idx, latent_idx, top_ctx, mid_ctx, neg_ctx, n_pos=16)
        freq_candidates = self.discover_frequent_latents(probe_data.pos_tokens)
        
        # Add the frequent ones (up to 12 per SAE)
        for fid in freq_candidates:
            if fid not in combined_set:
                combined_set.add(fid)
                final_list.append(fid)
                
        # Sort candidates by layer for the matrix layout
        final_list.sort(key=lambda x: (x.layer, VALID_KINDS.index(x.kind), x.index))
        
        self.console.print(f"[green]Identified {len(final_list)} latents (including {len(freq_candidates)} high-frequency neighbors).[/green]")
        return final_list

    def discover_frequent_latents(self, tokens: torch.Tensor) -> List[FeatureID]:
        """Find latents that fire most frequently during the seed's sequences."""
        self.console.print("\n[bold cyan]Scanning all 1.1M latents for high-frequency co-firing...[/bold cyan]")
        
        num_components = self.bank.n_layer * len(self.bank.kinds)
        # [num_components, d_sae] tensor for counting
        counts = torch.zeros((num_components, self.bank.d_sae), dtype=torch.int32, device="cpu")
        
        def count_hook(layer_idx, activations):
            for kind_idx, kind in enumerate(self.bank.kinds):
                act = activations[kind_idx]
                _, top_indices = self.bank.encode(act, kind, layer_idx)
                comp_idx = build_component_idx(layer_idx, kind_idx, len(self.bank.kinds))
                # top_indices: [B, T, K]
                counts[comp_idx].scatter_add_(
                    0, 
                    top_indices.cpu().flatten().long(), 
                    torch.ones(top_indices.numel(), dtype=torch.int32)
                )

        self.model.forward(tokens, num_gen=1, tokenize_final=False, activations_callback=count_hook, return_activations=False)
        
        frequent_fids = []
        for comp_idx in range(num_components):
            # Get top 12 for this SAE
            vals, idxs = torch.topk(counts[comp_idx], k=12)
            layer, k_idx = split_component_idx(comp_idx, len(self.bank.kinds))
            kind = self.bank.kinds[k_idx]
            for i in range(12):
                if vals[i] > 0:
                    frequent_fids.append(FeatureID(layer, kind, int(idxs[i].item())))
        
        return frequent_fids

    def _capture_acts(self, tokens: torch.Tensor, pos_argmax: torch.Tensor, fids: List[FeatureID], patcher=None) -> torch.Tensor:
        """Capture activations for a list of FIDs at specific positions."""
        B = tokens.shape[0]
        # Result: [len(fids), B]
        results = torch.zeros((len(fids), B), device="cpu")
        
        # Group FIDs by layer/kind for efficient capture
        by_lk: Dict[Tuple[int, str], List[Tuple[int, int]]] = {}
        for i, fid in enumerate(fids):
            by_lk.setdefault((fid.layer, fid.kind), []).append((fid.index, i))

        def capture_hook(layer_idx: int, activations: Tuple[torch.Tensor, ...]):
            for kind_idx, kind in enumerate(self.bank.kinds):
                lk = (layer_idx, kind)
                if lk in by_lk:
                    act = activations[kind_idx]
                    top_acts, top_indices = self.bank.encode(act, kind, layer_idx)
                    
                    for latent_idx, result_idx in by_lk[lk]:
                        # Find if latent_idx is in the top-K
                        is_target = (top_indices == latent_idx)
                        target_acts = torch.where(is_target, top_acts, torch.zeros_like(top_acts)).sum(dim=-1)
                        # Extract value at pos_argmax for each batch item
                        vals = target_acts[torch.arange(B), pos_argmax.to(target_acts.device)]
                        results[result_idx] = vals.cpu()

        self.model.forward(
            tokens, 
            num_gen=1, 
            tokenize_final=False, 
            activations_callback=capture_hook,
            patcher=patcher,
            return_activations=False
        )
        return results

    def run_sensitivity_sweep(self, layer_idx: int, kind: str, latent_idx: int, candidates: List[FeatureID]):
        """Measuring Causal Sensitivity across the neighborhood."""
        self.console.print(f"\n[bold cyan]Measuring Causal Sensitivity for {len(candidates)} nodes...[/bold cyan]")
        
        comp_idx = build_component_idx(layer_idx, VALID_KINDS.index(kind), len(VALID_KINDS))
        # 1. Build probe dataset to get tokens and peak positions
        probe_data = self.probe_builder.build_for_latent(comp_idx, latent_idx, top_ctx, mid_ctx, neg_ctx, n_pos=16)
        tokens = probe_data.pos_tokens
        pos_argmax = probe_data.pos_argmax
        B = tokens.shape[0]

        # 2. Capture Baseline
        self.console.print("  [dim]Capturing baseline activations...[/dim]")
        baseline_acts = self._capture_acts(tokens, pos_argmax, candidates)
        # Average across the batch
        mean_baseline = baseline_acts.mean(dim=1) 

        # 3. Sweep with progress bar
        matrix = torch.zeros((len(candidates), len(candidates)))
        
        self.console.print(f"  [dim]Performing causal sweep over {len(candidates)} candidates...[/dim]")
        for i, source_fid in enumerate(tqdm(candidates, desc="Ablating", unit="node")):
            # Create a one-node circuit for the patcher
            temp_node = CircuitNode(metadata={"feature_id": source_fid})
            temp_circuit = Circuit(name="ablation_sweep")
            temp_circuit.add_node(temp_node)
            
            # Inverse=True means ablate only the nodes in the circuit
            patcher = CircuitPatcher(self.bank, temp_circuit, self.avg_acts, inverse=True, pos_argmax=pos_argmax)
            
            # Capture activations while ablated
            ablated_acts = self._capture_acts(tokens, pos_argmax, candidates, patcher=patcher)
            mean_ablated = ablated_acts.mean(dim=1)
            
            # Sensitivity = % drop
            # If baseline is 0, sensitivity is 0.
            drop = (mean_baseline - mean_ablated) / mean_baseline.clamp(min=1e-5)
            matrix[i] = drop

        # 4. Display Matrix (Filtered for Top M Movers)
        # Calculate total absolute sensitivity per node (as source + as target)
        # to find the "most interesting" nodes.
        # matrix is [Source, Target]
        source_impact = matrix.abs().mean(dim=1)
        target_sensitivity = matrix.abs().mean(dim=0)
        total_interestingness = source_impact + target_sensitivity
        
        # Always keep the seed (target_fid) as the first node if possible
        comp_idx_seed = build_component_idx(layer_idx, VALID_KINDS.index(kind), len(VALID_KINDS))
        seed_fid = FeatureID(layer_idx, kind, latent_idx)
        
        # Take Top 20 most interesting nodes
        n_show = 20
        top_indices = torch.topk(total_interestingness, k=min(n_show, len(candidates))).indices.tolist()
        
        # Sort these top indices to maintain layer order
        show_candidates_idxs = sorted(top_indices, key=lambda idx: (candidates[idx].layer, VALID_KINDS.index(candidates[idx].kind), candidates[idx].index))
        show_candidates = [candidates[idx] for idx in show_candidates_idxs]

        table = Table(title=f"Ablation Sensitivity Matrix (% Drop) - Top {len(show_candidates)} movers", box=box.ROUNDED)
        table.add_column("Ablated \\ Affected", justify="left", style="bold yellow")
        for fid in show_candidates:
            table.add_column(f"{fid.layer} {fid.kind}\n{fid.index}", justify="right")
            
        for i, idx_i in enumerate(show_candidates_idxs):
            source_fid = candidates[idx_i]
            row = [f"{source_fid.layer} {source_fid.kind} {source_fid.index}"]
            for j, idx_j in enumerate(show_candidates_idxs):
                if idx_i == idx_j:
                    row.append("[dim]--[/dim]")
                else:
                    val = matrix[idx_i, idx_j].item()
                    color = "red" if val > 0.5 else "orange1" if val > 0.1 else "white"
                    if val < -0.1: color = "blue" # Facilitation
                    row.append(f"[{color}]{val*100:.0f}%[/{color}]")
            table.add_row(*row)
            
        self.console.print(table)
        self.console.print(f"\n[dim]Analysis complete. Swept {len(candidates)} latents in total.[/dim]")
        self.console.print("[dim]Interpretation: Row i, Column j shows how much latent j's activation drops when latent i is ablated.[/dim]")

def main():
    parser = argparse.ArgumentParser(description="Ablation Sensitivity Matrix Tool")
    parser.add_argument("--layer", type=int, help="Layer index")
    parser.add_argument("--kind", type=str, choices=VALID_KINDS, help="Kind of component")
    parser.add_argument("--latent", type=int, help="Latent index")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tool = AblationSensitivityTool(device)

    if args.layer is not None and args.kind is not None and args.latent is not None:
        candidates = tool.discover_candidates(args.layer, args.kind, args.latent)
        if candidates:
            tool.run_sensitivity_sweep(args.layer, args.kind, args.latent, candidates)
    else:
        tool.console.print("\nEntering interactive mode (CTRL+C to exit)")
        tool.console.print("Format: <layer> <kind> <latent>  e.g. 5 resid 12345\n")
        try:
            while True:
                raw = input("Analyze sensitivity for latent: ").strip()
                if not raw:
                    continue
                parsed = parse_latent_input(raw)
                if parsed is None:
                    print(f"Invalid input. Expected: <layer> <kind> <latent>  (kind: {', '.join(VALID_KINDS)})")
                    continue
                layer, kind, latent = parsed
                candidates = tool.discover_candidates(layer, kind, latent)
                if candidates:
                    tool.run_sensitivity_sweep(layer, kind, latent, candidates)
        except KeyboardInterrupt:
            print("\nExiting...")

if __name__ == "__main__":
    main()

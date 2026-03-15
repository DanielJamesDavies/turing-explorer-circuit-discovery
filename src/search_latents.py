import os
import torch
import argparse
import pandas as pd
from display.display import display, _build_mid_neg_seqs
from rich.panel import Panel
from rich.console import Console
from model.inference import Inference
from model.tokenizer import Tokenizer
from model.hooks import multi_patch
from sae.bank import SAEBank
from data.loader import DataLoader
from pipeline.component_index import component_idx, layer_component_bounds, split_component_idx

def get_latent_avg_activations(model, bank, loader, target_latents):
    """
    Calculates the average of the maximum activation values for each target latent
    across its top sequences.
    
    Uses passive observation (activations_callback) to measure baseline intensities.
    
    target_latents: list of (component_idx, latent_idx, sequence_ids)
    returns: list of (component_idx, latent_idx, avg_val)
    """
    results = []
    
    n_kinds = len(bank.kinds)
    for comp_idx, lat_idx, seq_ids in target_latents:
        max_activations = []
        
        def make_callback(c_idx, l_idx):
            def callback(layer_idx, activations):
                start, end = layer_component_bounds(layer_idx, n_kinds)
                if start <= c_idx < end:
                    _, kind_idx = split_component_idx(c_idx, n_kinds)
                    with torch.no_grad():
                        kind = bank.kinds[kind_idx]
                        latents = bank.encode(activations[kind_idx], kind, layer_idx)
                        top_acts = latents[0]    # (batch, seq, k)
                        top_indices = latents[1] # (batch, seq, k)
                        
                        # For each sequence in the batch, find its maximum activation for this latent
                        for b in range(top_indices.shape[0]):
                            mask = (top_indices[b] == l_idx)
                            if mask.any():
                                max_activations.append(top_acts[b][mask].max().item())
            return callback

        for _batch_ids, batch_tokens in loader.get_batches_by_ids(seq_ids):
            model.forward(
                batch_tokens,
                num_gen=1,
                tokenize_final=False,
                activations_callback=make_callback(comp_idx, lat_idx),
                return_activations=False
            )
        
        avg_val = sum(max_activations) / len(max_activations) if max_activations else 0.0
        results.append((comp_idx, lat_idx, avg_val))
        
    return results

def run_search(args, model, bank, loader, top_ctx, df, query_str, device):
    search_terms = [term.strip().lower() for term in query_str.split(",")]
    
    print(f"\nSearching for terms: {', '.join(search_terms)}")
    
    # Vectorized keyword matching using Pandas
    df = df.copy() # Avoid modifying the original cache
    df["relevance"] = 0
    for term in search_terms:
        # Case-insensitive match check
        df["relevance"] += df["text"].str.lower().str.count(term.lower())

    # Filter to matching latents and sort by relevance
    df_matches = df[df["relevance"] > 0]
    # Explicitly cast to DataFrame to help the linter if necessary, and use a list for 'by'
    assert isinstance(df_matches, pd.DataFrame)
    matches = df_matches.sort_values(by=["relevance"], ascending=False).head(args.n_latents)

    if matches.empty:
        print("\nNo matching latents found for the given query.")
        return

    print(f"\nFound {len(df[df['relevance'] > 0])} matching latents")

    # Prepare data for Display.analyze_and_print_latents
    top_results = []
    for _, row in matches.iterrows():
        comp_idx = int(row["component_idx"])
        lat_idx = int(row["latent_idx"])
        
        # Get the top sequences for the display re-run
        seq_ids = top_ctx.ctx_seq_idx[comp_idx, lat_idx, :args.n_sequences].tolist()
        seq_vals = top_ctx.ctx_seq_val[comp_idx, lat_idx, :args.n_sequences].tolist()
        valid_seqs = [(sid, sv) for sid, sv in zip(seq_ids, seq_vals) if sv > 0]
        
        top_results.append({
            "component_idx": comp_idx,
            "latent_idx":    lat_idx,
            "relevance":     row["relevance"],
            "sequences":     valid_seqs,
            "norm_val":      float(row["relevance"]),
            "raw_val":       float(seq_vals[0]) if seq_vals else 0.0,
            **_build_mid_neg_seqs(comp_idx, lat_idx, args.n_sequences),
        })

    # Delegate to the shared display logic for re-run activations and printing
    display.analyze_and_print_latents(model, bank, loader, top_results)

    if args.run_patch_clamp:
        # Identify top latents
        top_n_results = top_results[:args.n_patch]
        
        # Prepare target_latents for average calculation
        target_latents_info = []
        for info in top_n_results:
            comp_idx = info["component_idx"]
            lat_idx = info["latent_idx"]
            seq_ids = [s[0] for s in info["sequences"]]
            target_latents_info.append((comp_idx, lat_idx, seq_ids))
            
        print(f"\nCalculating average activations for top {len(target_latents_info)} latents...")
        latents_with_avg = get_latent_avg_activations(model, bank, loader, target_latents_info)
        
        for c_idx, l_idx, avg in latents_with_avg:
            layer_idx, kind_idx = split_component_idx(c_idx, len(bank.kinds))
            kind = bank.kinds[kind_idx]
            print(f"  Latent {l_idx} ({kind} layer {layer_idx}): avg activation = {avg:.4f}")
            
        # Prepare bank for decoding
        print(f"\nLoading decoders for patch clamp...")
        for kind_name in bank.kinds:
            for l in range(bank.n_layer):
                sae = bank.saes[kind_name][l]
                if sae: sae.move_decoder_to_vram()

        # Create the patcher factory
        def patcher_factory(m):
            def transform(layer_idx, kind, x):
                # Find target latents for this specific component
                relevant = [
                    (l_idx, avg_val) for c_idx, l_idx, avg_val in latents_with_avg
                    if c_idx == component_idx(layer_idx, bank.kinds.index(kind), len(bank.kinds))
                ]
                if not relevant:
                    return x
                
                with torch.no_grad():
                    # Encode to get sparse activations
                    encoded_acts, _, _ = bank.encode(x, kind, layer_idx)
                    
                    # Amplify target latents to their average values
                    for l_idx, avg_val in relevant:
                        encoded_acts[..., l_idx] = avg_val
                        
                    # Decode back to activation space
                    return bank.decode(encoded_acts, kind, layer_idx)
            
            return multi_patch(m, transform)
            
        # Setup input and run generation
        tokenizer = Tokenizer()
        input_tokens = torch.tensor([[tokenizer.get_bos_token()]], device=device)
            
        print(f"\nRunning patch clamp generation for {args.n_gen} tokens...")
        
        # 1. Baseline generation (no clamping)
        print(f"  [1/2] Baseline generation...")
        baseline_tokens, _, _ = model.forward(
            input_tokens,
            num_gen=args.n_gen,
            return_activations=False
        )
        baseline_text = tokenizer.decode(baseline_tokens[0].tolist())
        
        # 2. Clamped generation
        print(f"  [2/2] Clamped generation (top {len(latents_with_avg)} latents)...")
        clamped_tokens, _, _ = model.forward(
            input_tokens,
            num_gen=args.n_gen,
            patcher=patcher_factory,
            return_activations=False
        )
        clamped_text = tokenizer.decode(clamped_tokens[0].tolist())
        
        # Display comparison
        console = Console()
        console.print()
        console.rule("[bold]PATCH CLAMP GENERATION COMPARISON[/bold]")
        console.print()
        
        console.print(Panel(
            baseline_text,
            title="[bold green]Baseline Output[/bold green]",
            title_align="left",
            border_style="green",
            padding=(1, 2)
        ))
        
        console.print()
        
        console.print(Panel(
            clamped_text,
            title=f"[bold cyan]Clamped Output ({len(latents_with_avg)} Latents Amplified)[/bold cyan]",
            title_align="left",
            border_style="cyan",
            padding=(1, 2)
        ))
        
        console.print()
        console.rule()

def main():
    parser = argparse.ArgumentParser(description="Fast search for latents using pre-computed CSV cache.")
    parser.add_argument("--query", type=str, help="Keyword(s) to search for (comma separated). If omitted, enters interactive mode.")
    parser.add_argument("--n_latents", type=int, default=5, help="Number of top matching latents to return")
    parser.add_argument("--n_sequences", type=int, default=3, help="Number of sequences per latent to display")
    parser.add_argument("--run_patch_clamp", action="store_true", help="Run model generation with top latents amplified")
    parser.add_argument("--n_patch", type=int, default=5, help="Number of top latents to amplify in patch clamp")
    parser.add_argument("--n_gen", type=int, default=63, help="Number of tokens to generate for patch clamp")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("")

    print(f"Initializing DataLoader...")
    loader = DataLoader(device=device)

    print(f"Initializing Model...")
    model = Inference(device=device)

    print(f"Initializing SAE Bank...")
    bank = SAEBank(device=device, load_decoders=False)

    print(f"Loading Context Stores...")
    from store.context import top_ctx, mid_ctx, neg_ctx
    from store.logit_context import logit_ctx
    from store.top_coactivation import top_coactivation
    top_ctx.load("outputs/top_ctx.pt")
    logit_ctx.load("outputs/logit_ctx.pt")
    top_coactivation.load("outputs/top_coactivation.pt")
    for ctx, path, name in [
        (mid_ctx, "outputs/mid_ctx.pt", "mid_ctx"),
        (neg_ctx, "outputs/neg_ctx.pt", "neg_ctx"),
    ]:
        if os.path.exists(path):
            ctx.load(path)
            print(f"  loaded {name}")
        else:
            print(f"  {name} not found at {path} — skipping")
    
    print(f"Loading Search Cache...")
    df = pd.read_parquet("outputs/search_cache.parquet")

    if args.query:
        run_search(args, model, bank, loader, top_ctx, df, args.query, device)
    else:
        print("\nEntering interactive search mode (CTRL+C to exit)")
        try:
            first_search = True
            while True:
                prompt = "\nEnter search query (e.g. relativity): " if first_search else "\nEnter search query: "
                query = input(prompt).strip()
                if not query:
                    continue
                run_search(args, model, bank, loader, top_ctx, df, query, device)
                first_search = False
        except KeyboardInterrupt:
            print("\nExiting...")



if __name__ == "__main__":
    main()

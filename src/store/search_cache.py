import torch
import gc
import pandas as pd
from tqdm import tqdm
from model.tokenizer import Tokenizer
from pipeline.component_index import split_component_idx
from config import config

def generate_search_cache(
    top_ctx, 
    bank, 
    loader, 
    output_path="outputs/search_cache.parquet", 
    n_sequences=None,
    component_chunk_size=None
):
    """
    Vastly optimized search cache generation using a sequence-first approach and vectorized processing.
    Memory-safe version: processes components in chunks and minimizes intermediate allocations.
    """
    n_sequences = n_sequences or int(config.persist.search_cache_n_sequences or 16)
    component_chunk_size = component_chunk_size or int(config.persist.search_cache_component_chunk or 4)
    
    tokenizer = Tokenizer()
    n_components = top_ctx.num_components
    
    chunk_dfs = []
    n_chunks = (n_components + component_chunk_size - 1) // component_chunk_size
    
    pbar = tqdm(range(0, n_components, component_chunk_size), desc="  [search_cache]", unit="chunk")
    # Process components in chunks to bound memory usage
    for comp_start in pbar:
        comp_end = min(comp_start + component_chunk_size, n_components)
        pbar.set_postfix({"comps": f"{comp_start}-{comp_end-1}"})
        
        # 1. Get IDs needed for this chunk only
        chunk_idx = top_ctx.ctx_seq_idx[comp_start:comp_end, :, :n_sequences]
        chunk_val = top_ctx.ctx_seq_val[comp_start:comp_end, :, :n_sequences]
        mask = chunk_val > 0
        
        # Use torch.unique for memory efficiency before converting to list
        needed_ids_tensor = torch.unique(chunk_idx[mask])
        needed_ids = set(needed_ids_tensor.cpu().tolist())
        if 0 in needed_ids:
            needed_ids.remove(0)
            
        if not needed_ids:
            continue
            
        # 2. Decode texts for this chunk's IDs
        id_to_text: dict[int, str] = {}
        for shard_idx, (start_id, end_id) in enumerate(loader.shard_id_ranges):
            if start_id == -1: continue
            shard_needed = [idx for idx in needed_ids if start_id <= idx <= end_id]
            if not shard_needed: continue
            
            local_indices = [sid - start_id for sid in shard_needed]
            seq_map = loader.load_shard_sequences(shard_idx, local_indices)
            
            batch_ids: list[int] = []
            batch_tokens: list[list[int]] = []
            for sid in shard_needed:
                local_idx = sid - start_id
                if local_idx in seq_map:
                    batch_ids.append(sid)
                    batch_tokens.append(seq_map[local_idx].tolist())
            
            if batch_tokens:
                decoded_texts = tokenizer.tokenizer.batch_decode(batch_tokens, skip_special_tokens=True)
                for sid, text in zip(batch_ids, decoded_texts):
                    id_to_text[sid] = text
        
        # 3. Build DataFrame for this chunk
        indices = torch.nonzero(mask) # [N, 3] -> [chunk_comp_idx, latent_idx, seq_rank]
        active_sids = chunk_idx[mask]
        
        df_chunk = pd.DataFrame({
            "c_idx": indices[:, 0].cpu().numpy() + comp_start, # Map back to global component index
            "l_idx": indices[:, 1].cpu().numpy(),
            "sid": active_sids.cpu().numpy()
        })
        
        df_chunk["text"] = df_chunk["sid"].map(lambda x: id_to_text.get(x))
        df_chunk = df_chunk[df_chunk["text"].notna() & (df_chunk["text"].str.strip() != "")]
        
        if not df_chunk.empty:
            df_agg = df_chunk.groupby(["c_idx", "l_idx"])["text"].agg(" | ".join).reset_index()
            chunk_dfs.append(df_agg)
            
        # Cleanup chunk-specific data
        del chunk_idx, chunk_val, mask, needed_ids_tensor, needed_ids, id_to_text, df_chunk
        gc.collect()

    if not chunk_dfs:
        print("  [search_cache] Warning: No active latents found for search cache.")
        return

    # Final reduction
    df_final = pd.concat(chunk_dfs, ignore_index=True)
    kinds = bank.kinds
    df_final["layer"] = df_final["c_idx"].apply(lambda x: split_component_idx(int(x), len(kinds))[0])
    df_final["kind"] = df_final["c_idx"].apply(lambda x: kinds[split_component_idx(int(x), len(kinds))[1]])
    df_final.rename(columns={"c_idx": "component_idx", "l_idx": "latent_idx"}, inplace=True)
    
    df_final.to_parquet(output_path, index=False)
    print(f"  ✓ search_cache saved to {output_path}")

"""Exact per-head decomposition of GPT-2 attn-out SAE latent pre-activations, for the Top-K slots at each position.

081 engine.head_drive does it for a fixed list of latents; here the latent differs per (row, position, slot): the k
active slots `ti`. Same identity (hook_attn_out SAEs read the summed post-W_O output, layer-normalised per token):
  pre_i = sum_h ((o_h - mean o_h) / s) . w_i  +  ((b_O - mean b_O) / s) . w_i + b_eff_i,
  o_h = z_h W_O[h rows], s = std(sum_h o_h + b_O) + eps (torch's unbiased std, as the adapter's layer_norm_in).
"""
import torch


def head_drive_slots(z, w_o, b_o, w_enc, ti, n_heads, eps):
    """z [B, T, H*dh] (c_proj input), w_o [H*dh, d] (HF Conv1D weight), b_o [d], w_enc [d_sae, d] (encoder rows),
    ti [B, T, k] slot latent ids -> (drive [B, T, H, k], const [B, T, k]); sum_h drive + const + b_eff = pre."""
    B, T, Dz = z.shape
    dh = Dz // n_heads
    zh = z.reshape(B, T, n_heads, dh).float()
    wh = w_o.float().reshape(n_heads, dh, -1)
    o = torch.einsum("bthk,hkd->bthd", zh, wh)                       # [B, T, H, d]
    a = o.sum(2) + b_o.float()
    s = (a - a.mean(-1, keepdim=True)).std(-1, keepdim=True) + eps  # [B, T, 1]
    oc = (o - o.mean(-1, keepdim=True)) / s[..., None]
    W = w_enc.float()[ti.long()]                                     # [B, T, k, d]
    drive = torch.einsum("bthd,btkd->bthk", oc, W)
    bc = (b_o.float() - b_o.float().mean())
    const = torch.einsum("btd,btkd->btk", bc[None, None, :] / s, W)
    return drive, const

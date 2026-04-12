import torch
from typing import Optional, Tuple

@torch.compile
def fused_sparse_matmul(
    act1: Optional[torch.Tensor],
    act2: Optional[torch.Tensor],
    res1: Optional[torch.Tensor],
    res2: Optional[torch.Tensor],
    resc1: Optional[torch.Tensor],
    resc2: Optional[torch.Tensor]
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Fused kernel for SparseAct matrix multiplication.
    Computes act1 * act2 and residual contraction in one GPU pass.
    """
    out_act = None
    if act1 is not None and act2 is not None:
        out_act = act1 * act2
        
    out_resc = None
    if res1 is not None and res2 is not None:
        # Fused multiply-and-reduce
        out_resc = (res1 * res2).sum(dim=-1, keepdim=True)
    elif resc1 is not None and resc2 is not None:
        out_resc = resc1 * resc2
        
    return out_act, out_resc

@torch.compile
def fused_sparse_add(
    act1: Optional[torch.Tensor],
    act2: Optional[torch.Tensor],
    res1: Optional[torch.Tensor],
    res2: Optional[torch.Tensor],
    resc1: Optional[torch.Tensor],
    resc2: Optional[torch.Tensor]
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Fused kernel for SparseAct addition.
    """
    out_act = None
    if act1 is not None and act2 is not None:
        out_act = act1 + act2
    elif act1 is not None:
        out_act = act1
    elif act2 is not None:
        out_act = act2

    out_res = None
    if res1 is not None and res2 is not None:
        out_res = res1 + res2
    elif res1 is not None:
        out_res = res1
    elif res2 is not None:
        out_res = res2

    out_resc = None
    if resc1 is not None and resc2 is not None:
        out_resc = resc1 + resc2
    elif resc1 is not None:
        out_resc = resc1
    elif resc2 is not None:
        out_resc = resc2

    return out_act, out_res, out_resc

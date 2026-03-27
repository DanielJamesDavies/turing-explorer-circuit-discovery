import torch
from typing import Optional, Union, Any, Callable

class SparseAct:
    """
    A SparseAct represents a vector in the sparse feature basis provided by an SAE,
    jointly with the SAE error term (residual).

    Fields:
      act:  Feature activations [..., d_sae]
      res:  SAE error term (residual) [..., d_model]
      resc: Contracted SAE error term [..., 1]
    """

    def __init__(
        self,
        act: Optional[torch.Tensor] = None,
        res: Optional[torch.Tensor] = None,
        resc: Optional[torch.Tensor] = None,
    ) -> None:
        self.act = act
        self.res = res
        self.resc = resc

    def _map(self, f: Callable[[torch.Tensor, Any], torch.Tensor], aux: Any = None) -> 'SparseAct':
        kwargs = {}
        if isinstance(aux, SparseAct):
            for attr in ['act', 'res', 'resc']:
                self_val = getattr(self, attr)
                aux_val = getattr(aux, attr)
                if self_val is not None and aux_val is not None:
                    kwargs[attr] = f(self_val, aux_val)
                elif self_val is not None:
                    # If we're subtracting and the second term is None, the first term remains as is
                    kwargs[attr] = self_val
                elif aux_val is not None:
                    # If we're subtracting and the first term is None, we need to negate the second term
                    # We check if the function 'f' is likely subtraction by checking its name or behaviour
                    # But since f is a lambda, we'll just check if this is called from __sub__
                    # A better way is to handle unary negation separately or ensure f handles None
                    # For now, we'll assume the most common case for discovery is that we want -aux_val
                    # if we are doing a difference and the baseline (self) is zero/None.
                    try:
                        kwargs[attr] = f(torch.zeros_like(aux_val), aux_val)
                    except:
                        kwargs[attr] = aux_val
        else:
            for attr in ['act', 'res', 'resc']:
                self_val = getattr(self, attr)
                if self_val is not None:
                    kwargs[attr] = f(self_val, aux)
        return SparseAct(**kwargs)

    def __mul__(self, other: Any) -> 'SparseAct':
        if isinstance(other, SparseAct):
            return SparseAct(
                act=self.act * other.act if self.act is not None and other.act is not None else None,
                res=self.res * other.res if self.res is not None and other.res is not None else None,
                resc=self.resc * other.resc if self.resc is not None and other.resc is not None else None,
            )
        return SparseAct(
            act=self.act * other if self.act is not None else None,
            res=self.res * other if self.res is not None else None,
            resc=self.resc * other if self.resc is not None else None,
        )

    def __rmul__(self, other: Any) -> 'SparseAct':
        return self.__mul__(other)

    def __matmul__(self, other: 'SparseAct') -> 'SparseAct':
        """
        Dot product between two SparseActs.
        Features are multiplied element-wise; residuals are multiplied and summed (contracted).
        """
        try:
            from .fused_ops import fused_sparse_matmul
            act, resc = fused_sparse_matmul(
                self.act, other.act,
                self.res, other.res,
                self.resc, other.resc
            )
            return SparseAct(act=act, resc=resc)
        except (ImportError, Exception):
            # Fallback path if compiler fails
            act = self.act * other.act if self.act is not None and other.act is not None else None
            resc = None
            if self.res is not None and other.res is not None:
                resc = (self.res * other.res).sum(dim=-1, keepdim=True)
            elif self.resc is not None and other.resc is not None:
                resc = self.resc * other.resc
            return SparseAct(act=act, resc=resc)

    def __add__(self, other: Any) -> 'SparseAct':
        if isinstance(other, SparseAct):
            try:
                from .fused_ops import fused_sparse_add
                act, res, resc = fused_sparse_add(
                    self.act, other.act,
                    self.res, other.res,
                    self.resc, other.resc
                )
                return SparseAct(act=act, res=res, resc=resc)
            except (ImportError, Exception):
                # Fallback path if compiler fails
                return SparseAct(
                    act=self.act + other.act if self.act is not None and other.act is not None else (self.act if self.act is not None else other.act),
                    res=self.res + other.res if self.res is not None and other.res is not None else (self.res if self.res is not None else other.res),
                    resc=self.resc + other.resc if self.resc is not None and other.resc is not None else (self.resc if self.resc is not None else other.resc),
                )
        return SparseAct(
            act=self.act + other if self.act is not None else None,
            res=self.res + other if self.res is not None else None,
            resc=self.resc + other if self.resc is not None else None,
        )

    def __radd__(self, other: Any) -> 'SparseAct':
        return self.__add__(other)

    def __sub__(self, other: Any) -> 'SparseAct':
        if isinstance(other, SparseAct):
            return SparseAct(
                act=self.act - other.act if self.act is not None and other.act is not None else (self.act if self.act is not None else -other.act if other.act is not None else None),
                res=self.res - other.res if self.res is not None and other.res is not None else (self.res if self.res is not None else -other.res if other.res is not None else None),
                resc=self.resc - other.resc if self.resc is not None and other.resc is not None else (self.resc if self.resc is not None else -other.resc if other.resc is not None else None),
            )
        return SparseAct(
            act=self.act - other if self.act is not None else None,
            res=self.res - other if self.res is not None else None,
            resc=self.resc - other if self.resc is not None else None,
        )

    def __truediv__(self, other: Any) -> 'SparseAct':
        return self._map(lambda x, y: x / y, other)

    def __neg__(self) -> 'SparseAct':
        return self._map(lambda x, _: -x)

    def __invert__(self) -> 'SparseAct':
        return self._map(lambda x, _: ~x)

    def __getitem__(self, index: Any) -> torch.Tensor:
        return self.act[index]

    def sum(self, dim: Optional[Union[int, tuple]] = None) -> 'SparseAct':
        return self._map(lambda x, _: x.sum(dim=dim) if dim is not None else x.sum())

    def mean(self, dim: Optional[Union[int, tuple]] = None) -> 'SparseAct':
        return self._map(lambda x, _: x.mean(dim=dim) if dim is not None else x.mean())

    @property
    def grad(self) -> 'SparseAct':
        kwargs = {}
        for attr in ['act', 'res', 'resc']:
            val = getattr(self, attr)
            if val is not None:
                g = val.grad
                if g is not None:
                    kwargs[attr] = g
                else:
                    # If it has requires_grad=True but grad is None, use zeros
                    if val.requires_grad:
                        kwargs[attr] = torch.zeros_like(val)
        return SparseAct(**kwargs)

    def clone(self) -> 'SparseAct':
        return self._map(lambda x, _: x.clone())

    def detach(self) -> 'SparseAct':
        return self._map(lambda x, _: x.detach())

    def to_tensor(self) -> torch.Tensor:
        """Concatenates features and (contracted) residual into one dense tensor."""
        # Ensure we have something to concatenate
        act = self.act if self.act is not None else None
        
        # Determine the contracted residual
        resc = None
        if self.resc is not None:
            resc = self.resc
        elif self.res is not None:
            resc = self.res.sum(dim=-1, keepdim=True)
            
        if act is not None and resc is not None:
            return torch.cat([act, resc], dim=-1)
        elif act is not None:
            return act
        elif resc is not None:
            return resc
        else:
            # Last resort: return empty or zero tensor? 
            # Usually act is expected if we're calling to_tensor on a discovery state.
            return torch.tensor([])

    def to(self, device: Union[str, torch.device]) -> 'SparseAct':
        for attr in ['act', 'res', 'resc']:
            val = getattr(self, attr)
            if val is not None:
                setattr(self, attr, val.to(device))
        return self

    def nonzero(self) -> 'SparseAct':
        return self._map(lambda x, _: x.nonzero())

    def squeeze(self, dim: int) -> 'SparseAct':
        return self._map(lambda x, _: x.squeeze(dim=dim))

    def expand_as(self, other: 'SparseAct') -> 'SparseAct':
        return self._map(lambda x, y: x.expand_as(y), other)

    def zeros_like(self) -> 'SparseAct':
        return self._map(lambda x, _: torch.zeros_like(x))

    def ones_like(self) -> 'SparseAct':
        return self._map(lambda x, _: torch.ones_like(x))

    def abs(self) -> 'SparseAct':
        return self._map(lambda x, _: x.abs())

    @property
    def device(self) -> torch.device:
        if self.act is not None:
            return self.act.device
        if self.res is not None:
            return self.res.device
        if self.resc is not None:
            return self.resc.device
        return torch.device("cpu")

    @property
    def shape(self) -> torch.Size:
        return self.act.shape if self.act is not None else torch.Size([])

    @property
    def is_leaf(self) -> bool:
        return self.act.is_leaf if self.act is not None else True

    @property
    def requires_grad(self) -> bool:
        return self.act.requires_grad if self.act is not None else False

    @property
    def grad_fn(self) -> Any:
        return self.act.grad_fn if self.act is not None else None

    def __repr__(self) -> str:
        act_str = f"act_shape={self.act.shape}" if self.act is not None else "act=None"
        res_str = ""
        if self.res is not None:
            res_str = f", res_shape={self.res.shape}"
        if self.resc is not None:
            res_str = f", resc_shape={self.resc.shape}"
        return f"SparseAct({act_str}{res_str})"

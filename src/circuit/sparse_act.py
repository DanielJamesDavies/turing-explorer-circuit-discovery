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
                if self_val is not None or aux_val is not None:
                    if self_val is not None and aux_val is not None:
                        kwargs[attr] = f(self_val, aux_val)
                    elif self_val is not None:
                        # If adding/subtracting, we should probably keep the existing value
                        # But since f is generic, we'll only do this if it's a known identity op
                        # For now, let's just pass self_val if aux_val is None
                        kwargs[attr] = self_val
                    else:
                        kwargs[attr] = aux_val
        else:
            for attr in ['act', 'res', 'resc']:
                self_val = getattr(self, attr)
                if self_val is not None:
                    kwargs[attr] = f(self_val, aux)
        return SparseAct(**kwargs)

    def __mul__(self, other: Any) -> 'SparseAct':
        return self._map(lambda x, y: x * y, other)

    def __rmul__(self, other: Any) -> 'SparseAct':
        return self.__mul__(other)

    def __matmul__(self, other: 'SparseAct') -> 'SparseAct':
        """
        Dot product between two SparseActs.
        Features are multiplied element-wise; residuals are multiplied and summed (contracted).
        """
        if self.res is not None and other.res is not None:
            return SparseAct(
                act=self.act * other.act,
                resc=(self.res * other.res).sum(dim=-1, keepdim=True)
            )
        elif self.resc is not None and other.resc is not None:
            return SparseAct(
                act=self.act * other.act,
                resc=self.resc * other.resc
            )
        else:
            # Fallback for when one side doesn't have residuals
            return SparseAct(act=self.act * other.act)

    def __add__(self, other: Any) -> 'SparseAct':
        return self._map(lambda x, y: x + y, other)

    def __radd__(self, other: Any) -> 'SparseAct':
        return self.__add__(other)

    def __sub__(self, other: Any) -> 'SparseAct':
        return self._map(lambda x, y: x - y, other)

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
        return self.act.device

    @property
    def shape(self) -> torch.Size:
        return self.act.shape

    def __repr__(self) -> str:
        res_str = ""
        if self.res is not None:
            res_str = f", res_shape={self.res.shape}"
        if self.resc is not None:
            res_str = f", resc_shape={self.resc.shape}"
        return f"SparseAct(act_shape={self.act.shape}{res_str})"

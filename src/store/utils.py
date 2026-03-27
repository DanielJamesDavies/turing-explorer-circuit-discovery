from __future__ import annotations

from typing import Optional, overload, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from typing import Any


class _AutoAllocTensor:
    """
    Descriptor for lazily-allocated tensor attributes on store classes.

    Any read or write of the attribute will transparently call owner.allocate()
    if the tensor has not yet been initialised (i.e. the backing slot is None).
    This makes it impossible to hit a NoneType crash regardless of whether the
    caller goes through update_component() or writes directly to the tensor.

    Usage
    -----
    class MyStore:
        some_tensor = _AutoAllocTensor()

        def allocate(self):
            self.some_tensor = torch.zeros(...)   # sets the backing slot

    The descriptor stores its value under the private name "_tensor_<attr_name>"
    in the instance __dict__ to avoid conflicting with the class-level descriptor.
    """

    def __set_name__(self, owner: type, name: str) -> None:
        self.public  = name
        self.private = f"_tensor_{name}"

    @overload
    def __get__(self, obj: None, objtype: type) -> "_AutoAllocTensor": ...
    @overload
    def __get__(self, obj: object, objtype: type) -> torch.Tensor: ...

    def __get__(self, obj: Any, objtype: Any = None) -> "torch.Tensor | _AutoAllocTensor":
        if obj is None:
            return self
        val: Optional[torch.Tensor] = obj.__dict__.get(self.private)
        if val is None:
            obj.allocate()
        return obj.__dict__.get(self.private)  # type: ignore[return-value]

    def __set__(self, obj: object, value: Optional[torch.Tensor]) -> None:
        obj.__dict__[self.private] = value

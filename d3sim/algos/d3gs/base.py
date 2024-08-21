import enum
import math
from typing import Annotated, Any, Generic, Self, Type, TypeVar

from d3sim.algos.d3gs.data_base import D3simDataset
from d3sim.constants import D3SIM_DEFAULT_DEVICE
from d3sim.core import dataclass_dispatch as dataclasses
from d3sim.core.pytorch.hmt import HomogeneousTensor
import torch
import abc
import pccm 
import numpy as np 

class GaussianCoreFields(enum.Enum):
    XYZ = "xyz"
    QUATERNION_XYZW = "quaternion_xyzw"
    OPACITY = "opacity"
    SCALE = "scale"
    RGB = "rgb"

def get_qualname_of_type(klass: Type) -> str:
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + '.' + klass.__qualname__

T = TypeVar("T", bound="GaussianModelBase")
class GaussianModelProxyBase(Generic[T], abc.ABC):
    def __init__(self, model: T, code: pccm.FunctionCode, gaussian_idx: str, batch_idx: str, batch_size: int, is_bwd: bool = False):
        self._model = model 
        self._code = code 
        self._gaussian_idx = gaussian_idx
        self._batch_size = batch_size
        self._is_bwd = is_bwd
        self._batch_idx = batch_idx
        self._unique_id = get_qualname_of_type(type(self))

    def get_unique_id(self):
        return self._unique_id

    @abc.abstractmethod
    def read_field(self, field: GaussianCoreFields, 
            out: str, normed_dir: str = ""):
        """Read field from proxy.
        Call order of fields is always `xyz, scale|quat, opacity, rgb`,
        `|` means any order.
        """

    @abc.abstractmethod
    def prepare_field_proxy(self): 
        """Run in start of prep/prep_bwd kernel, 
        if batch_size > 1 and is backward, run inside batch for loop.
        """

    @abc.abstractmethod
    def accumulate_field_grad(self, field: GaussianCoreFields, 
            out: str, grad: str, normed_dir: str = "", normed_dir_grad: str = ""):
        """Accumulate grad. if batch size == 0, you can save grad directly.
        Call order of fields is always `scale|quat|opacity, rgb, xyz`,
        `|` means any order.
        """
    @abc.abstractmethod
    def save_accumulated_grad(self): ...

    def validate_prep_inputs(self, inputs: dict[str, Any]): 
        return 


@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class GaussianModelBase(HomogeneousTensor, abc.ABC):
    @abc.abstractmethod
    def get_unique_kernel_key(self) -> str: ...

    @abc.abstractmethod
    def get_proxy_field_dict(self) -> dict[str, torch.Tensor]: ...
        
    def set_requires_grad(self, requires_grad: bool):
        for k, v in self.get_all_tensor_fields().items():
            if v.dtype == torch.float32 or v.dtype == torch.float16 or v.dtype == torch.bfloat16:
                v.requires_grad_(requires_grad)

    def clear_grad(self):
        for k, v in self.get_all_tensor_fields().items():
            v.grad = None

    @abc.abstractmethod
    def create_proxy(self, code: pccm.FunctionCode, gaussian_idx: str, batch_idx: str, batch_size: int, is_bwd: bool = False) -> GaussianModelProxyBase:
        raise NotImplementedError

    @abc.abstractmethod
    def create_model_with_act(self) -> Self: ...

    @classmethod
    @abc.abstractmethod
    def create_from_dataset(cls,
              ds: D3simDataset,
              device: torch.device = torch.device(D3SIM_DEFAULT_DEVICE)):
        raise NotImplementedError


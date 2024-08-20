from typing import Any, ClassVar, Generic, Literal, get_args, get_origin
import torch 
import d3sim.core.dataclass_dispatch as dataclasses
from typing import Self
from torch.nn import ParameterDict

from d3sim.core.arrcheck.dcbase import DataClassWithArrayCheck, is_annotated, is_optional

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class HomogeneousTensor(DataClassWithArrayCheck):
    __tensor_field_type_metas__: ClassVar[dict[str, Any] | None] = None
    # TODO should we implement this in __init_subclass__?
    def __post_init__(self):
        # analysis annotations, check is Tensor.
        # currently we support torch.Tensor, torch.nn.Parameter and 
        # optional version of them.
        type_self = type(self)
        if type_self.__tensor_field_type_metas__ is None:
            type_self.__tensor_field_type_metas__ = {}
            for field in dataclasses.fields(self):
                field_type = field.type
                if is_annotated(field_type):
                    field_type = get_args(field_type)[0]
                    assert field_type is not None 
                if is_optional(field_type):
                    args = get_args(field_type) # is union, only None or Tensor allowed
                    if torch.Tensor in args or torch.nn.Parameter in args:
                        assert len(args) == 2, "only None or Tensor allowed"
                        type_self.__tensor_field_type_metas__[field.name] = 1
                else:
                    if field_type == torch.Tensor or field_type == torch.nn.Parameter:
                        type_self.__tensor_field_type_metas__[field.name] = 1

    def _get_all_fields(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        res: dict[str, torch.Tensor] = {}
        res_non_tensor: dict[str, Any] = {}
        for field in dataclasses.fields(self):
            field_value = getattr(self, field.name)
            if isinstance(field_value, torch.Tensor):
                res[field.name] = field_value
            else:
                res_non_tensor[field.name] = field_value
        return res, res_non_tensor


    def get_all_tensor_fields(self) -> dict[str, torch.Tensor]:
        return self._get_all_fields()[0]

    def get_all_non_tensor_fields(self) -> dict[str, Any]:
        return self._get_all_fields()[1]

    def get_all_parameter_fields(self) -> dict[str, torch.nn.Parameter]:
        res: dict[str, torch.nn.Parameter] = {}
        for field in dataclasses.fields(self):
            field_value = getattr(self, field.name)
            if isinstance(field_value, torch.nn.Parameter):
                res[field.name] = field_value
        return res 

    def __getitem__(self, indices_or_mask: Any) -> Self:
        tensor_fields = self.get_all_tensor_fields()
        for key, value in tensor_fields.items():
            tensor_fields[key] = value[indices_or_mask]
        return self.__class__(**tensor_fields, **self.get_all_non_tensor_fields())

    def __setitem__(self, indices_or_mask: Any, other: Self) -> None:
        for key, value in self.get_all_tensor_fields().items():
            value[indices_or_mask] = getattr(other, key)

    def __len__(self) -> int:
        return len(next(iter(self.get_all_tensor_fields().values())))

    def to_parameter_dict(self):
        return ParameterDict(self.get_all_parameter_fields()) 

    def concat(self, other: Self) -> Self:
        tensor_fields = self.get_all_tensor_fields()
        for key, value in tensor_fields.items():
            is_parameter = isinstance(value, torch.nn.Parameter)
            tensor_fields[key] = torch.cat([value, getattr(other, key)], dim=0)
            if is_parameter:
                tensor_fields[key] = torch.nn.Parameter(tensor_fields[key])
        return self.__class__(**tensor_fields, **self.get_all_non_tensor_fields())

    def concat_inplace(self, other: Self) -> Self:
        for key, value in self.get_all_tensor_fields().items():
            is_parameter = isinstance(value, torch.nn.Parameter)
            res = torch.cat([value, getattr(other, key)], dim=0)
            if is_parameter:
                res = torch.nn.Parameter(res)
            setattr(self, key, res)
        return self

    def index_inplace_(self, indices_or_mask: Any) -> Self:
        for key, value in self.get_all_tensor_fields().items():
            is_parameter = isinstance(value, torch.nn.Parameter)
            res = value[indices_or_mask]
            if is_parameter:
                res = torch.nn.Parameter(res)
            setattr(self, key, res)
        return self

    def assign_inplace(self, other: Self) -> Self:
        for key, value in self.get_all_tensor_fields().items():
            setattr(self, key, getattr(other, key))
        return self

    def to_device(self, device: Literal["cpu", "cuda"]) -> Self:
        tensor_fields = self.get_all_tensor_fields()
        for key, value in tensor_fields.items():
            is_parameter = isinstance(value, torch.nn.Parameter)
            res = value.to(device)
            if is_parameter:
                res = torch.nn.Parameter(res)
            tensor_fields[key] = res
        return self.__class__(**tensor_fields, **self.get_all_non_tensor_fields())

    def to_device_inplace(self, device: Literal["cpu", "cuda"]) -> Self:
        for key, value in self.get_all_tensor_fields().items():
            is_parameter = isinstance(value, torch.nn.Parameter)
            res = value.to(device)
            if is_parameter:
                res = torch.nn.Parameter(res)
            setattr(self, key, res)
        return self

    def to_autograd_tuple_repr(self):
        """should only be used in pytorch autograd function
        """
        res = []
        res_tensor_field_names = []
        res_non_tensor = {}
        tensor_field_metas = type(self).__tensor_field_type_metas__
        assert tensor_field_metas is not None
        for field in dataclasses.fields(self):
            field_value = getattr(self, field.name)
            if field.name in tensor_field_metas:
                res_tensor_field_names.append(field.name)
                res.append(field_value)
            else:
                res_non_tensor[field.name] = field_value
        return tuple(res), res_tensor_field_names, res_non_tensor

    def to_autograd_tuple_repr_tensor_only(self):
        """should only be used in pytorch autograd function
        """
        tensor_field_metas = type(self).__tensor_field_type_metas__
        assert tensor_field_metas is not None
        return tuple(getattr(self, k) for k in tensor_field_metas)

    @classmethod
    def from_autograd_tuple_repr(cls, tensor_tuple, tensor_names, non_tensor_dict):
        tensor_dict = dict(zip(tensor_names, tensor_tuple))
        return cls(**tensor_dict, **non_tensor_dict)
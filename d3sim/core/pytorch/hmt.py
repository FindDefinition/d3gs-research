from typing import Any, Generic, Literal
import torch 
import d3sim.core.dataclass_dispatch as dataclasses
from typing import Self
from torch.nn import ParameterDict

from d3sim.core.arrcheck.dcbase import DataClassWithArrayCheck

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class HomogeneousTensor(DataClassWithArrayCheck):
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
            tensor_fields[key] = torch.cat([value, getattr(other, key)], dim=0)
        return self.__class__(**tensor_fields, **self.get_all_non_tensor_fields())

    def concat_inplace(self, other: Self) -> Self:
        for key, value in self.get_all_tensor_fields().items():
            setattr(self, key, torch.cat([value, getattr(other, key)], dim=0))
        return self

    def index_inplace_(self, indices_or_mask: Any) -> Self:
        for key, value in self.get_all_tensor_fields().items():
            setattr(self, key, value[indices_or_mask])
        return self

    def assign_inplace(self, other: Self) -> Self:
        for key, value in self.get_all_tensor_fields().items():
            setattr(self, key, getattr(other, key))
        return self
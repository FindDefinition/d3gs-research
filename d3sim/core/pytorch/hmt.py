from typing import Any, Generic, Literal
import torch 
import d3sim.core.dataclass_dispatch as dataclasses
from typing import Self
from torch.nn import ParameterDict

from d3sim.core.arrcheck.dcbase import DataClassWithArrayCheck

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class HomogeneousTensor(DataClassWithArrayCheck):

    def get_all_tensor_fields(self) -> dict[str, torch.Tensor]:
        res: dict[str, torch.Tensor] = {}
        for field in dataclasses.fields(self):
            field_value = getattr(self, field.name)
            if isinstance(field_value, torch.Tensor):
                res[field.name] = field_value
        return res 

    def get_all_parameter_fields(self) -> dict[str, torch.nn.Parameter]:
        res: dict[str, torch.nn.Parameter] = {}
        for field in dataclasses.fields(self):
            field_value = getattr(self, field.name)
            if isinstance(field_value, torch.nn.Parameter):
                res[field.name] = field_value
        return res 

    def __getitem__(self, indices: Any) -> Self:
        tensor_fields = self.get_all_tensor_fields()
        for key, value in tensor_fields.items():
            tensor_fields[key] = value[indices]
        return self.__class__(**tensor_fields)

    def __len__(self) -> int:
        return len(next(iter(self.get_all_tensor_fields().values())))

    def to_parameter_dict(self):
        return ParameterDict(self.get_all_parameter_fields()) 

        
import math
from typing import Annotated

from d3sim.core import dataclass_dispatch as dataclasses
from d3sim.core.pytorch.hmt import HomogeneousTensor
import torch 
from d3sim.core import arrcheck
import abc 

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class GaussianModelBase(HomogeneousTensor, abc.ABC):
    xyz: Annotated[torch.Tensor, arrcheck.ArrayCheck(["N", 3], arrcheck.F32)]
    quaternion_xyzw: Annotated[torch.Tensor, arrcheck.ArrayCheck(["N", 4], arrcheck.F32)]
    scale: Annotated[torch.Tensor, arrcheck.ArrayCheck(["N", 3], arrcheck.F32)]
    opacity: Annotated[torch.Tensor, arrcheck.ArrayCheck(["N"], arrcheck.F32)]
    color_sh: Annotated[torch.Tensor, arrcheck.ArrayCheck(["N", -1, 3], arrcheck.F32)]

    @property 
    def xyz_act(self) -> torch.Tensor:
        return self.xyz

    @property 
    @abc.abstractmethod
    def quaternion_xyzw_act(self) -> torch.Tensor:
        raise NotImplementedError

    @property 
    @abc.abstractmethod
    def scale_act(self) -> torch.Tensor:
        raise NotImplementedError

    @property 
    @abc.abstractmethod
    def opacity_act(self) -> torch.Tensor:
        raise NotImplementedError

    @property 
    def color_sh_act(self) -> torch.Tensor:
        return self.color_sh

    @property 
    def color_sh_degree(self) -> int:
        res = int(math.sqrt(self.color_sh.shape[1])) - 1
        assert (res + 1) * (res + 1) == self.color_sh.shape[1]
        return res 

    @property 
    def custom_features(self) -> torch.Tensor | None:
        return None


@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class GaussianModelOrigin(GaussianModelBase):
    @property 
    def quaternion_xyzw_act(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self.quaternion_xyzw, p=2, dim=-1)

    @property 
    def scale_act(self) -> torch.Tensor:
        return torch.exp(self.scale)

    @property 
    def opacity_act(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity)

    @staticmethod 
    def empty(N: int, num_degree: int):
        return GaussianModelOrigin(
            xyz=torch.empty(N, 3),
            quaternion_xyzw=torch.empty(N, 4),
            scale=torch.empty(N, 3),
            opacity=torch.empty(N),
            color_sh=torch.empty(N, (num_degree + 1) * (num_degree + 1), 3)
        )

    @staticmethod 
    def zeros(N: int, num_degree: int):
        return GaussianModelOrigin(
            xyz=torch.zeros(N, 3),
            quaternion_xyzw=torch.zeros(N, 4),
            scale=torch.zeros(N, 3),
            opacity=torch.zeros(N),
            color_sh=torch.zeros(N, (num_degree + 1) * (num_degree + 1), 3)
        )
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
    quaternion_xyzw: Annotated[torch.Tensor,
                               arrcheck.ArrayCheck(["N", 4], arrcheck.F32)]
    scale: Annotated[torch.Tensor, arrcheck.ArrayCheck(["N", 3], arrcheck.F32)]
    opacity: Annotated[torch.Tensor, arrcheck.ArrayCheck(["N"], arrcheck.F32)]
    color_sh: Annotated[torch.Tensor,
                        arrcheck.ArrayCheck(["N", -1, 3], arrcheck.F32)]
    color_sh_base: Annotated[torch.Tensor | None,
                             arrcheck.ArrayCheck(["N", 3], arrcheck.F32)]
    instance_id: Annotated[torch.Tensor | None, arrcheck.ArrayCheck(["N"], arrcheck.I32)] = None

    act_applied: bool = False
    cur_sh_degree: int = 0

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

    @abc.abstractmethod
    def inverse_scale_act(self, scale_act: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def opacity_act(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def fused_quaternion_xyzw_act_op(self) -> tuple[str, str] | None:
        """there are three component support fused act to save gpu memory.
        when you implement fused op, you should return identity
        in component_act property.
        """
        return None

    @property
    def fused_scale_act_op(self) -> tuple[str, str] | None:
        return None

    @property
    def fused_opacity_act_op(self) -> tuple[str, str] | None:
        return None

    @property
    def color_sh_act(self) -> torch.Tensor:
        return self.color_sh

    @property
    def color_sh_degree(self) -> int:
        dim = self.color_sh.shape[1]
        if self.color_sh_base is not None:
            dim += 1
        res = int(math.sqrt(dim)) - 1
        assert (res + 1) * (res + 1) == dim
        return res

    def set_requires_grad(self, requires_grad: bool):
        self.xyz.requires_grad_(requires_grad)
        self.quaternion_xyzw.requires_grad_(requires_grad)
        self.scale.requires_grad_(requires_grad)
        self.opacity.requires_grad_(requires_grad)
        self.color_sh.requires_grad_(requires_grad)

    def clear_grad(self):
        self.xyz.grad = None
        self.quaternion_xyzw.grad = None
        self.scale.grad = None
        self.opacity.grad = None
        self.color_sh.grad = None

    def create_model_with_act(self):
        if self.act_applied:
            return self
        return self.__class__(xyz=self.xyz_act,
                              quaternion_xyzw=self.quaternion_xyzw_act,
                              scale=self.scale_act,
                              opacity=self.opacity_act,
                              color_sh=self.color_sh_act,
                              cur_sh_degree=self.cur_sh_degree,
                              color_sh_base=self.color_sh_base,
                              instance_id=self.instance_id,
                              act_applied=True)

    @classmethod
    def empty(cls,
              N: int,
              max_num_degree: int,
              split: bool = False,
              dtype: torch.dtype = torch.float32,
              device: torch.device = torch.device("cpu")):
        return cls(
            xyz=torch.empty(N, 3, dtype=dtype, device=device),
            quaternion_xyzw=torch.empty(N, 4, dtype=dtype, device=device),
            scale=torch.empty(N, 3, dtype=dtype, device=device),
            opacity=torch.empty(N, dtype=dtype, device=device),
            color_sh_base=torch.empty(N, 3, dtype=dtype, device=device)
            if split else None,
            color_sh=torch.empty(N, (max_num_degree + 1) *
                                 (max_num_degree + 1) - (1 if split else 0),
                                 3,
                                 dtype=dtype,
                                 device=device),
            cur_sh_degree=0,
        )

    @classmethod
    def empty_parameter(cls,
                        N: int,
                        max_num_degree: int,
                        split: bool = False,
                        dtype: torch.dtype = torch.float32,
                        device: torch.device = torch.device("cpu")):
        return cls(
            xyz=torch.nn.Parameter(
                torch.empty(N, 3, dtype=dtype, device=device)),
            quaternion_xyzw=torch.nn.Parameter(
                torch.empty(N, 4, dtype=dtype, device=device)),
            scale=torch.nn.Parameter(
                torch.empty(N, 3, dtype=dtype, device=device)),
            opacity=torch.nn.Parameter(
                torch.empty(N, dtype=dtype, device=device)),
            color_sh_base=torch.nn.Parameter(
                torch.empty(N, 3, dtype=dtype, device=device))
            if split else None,
            color_sh=torch.nn.Parameter(
                torch.empty(N, (max_num_degree + 1) * (max_num_degree + 1) -
                            (1 if split else 0),
                            3,
                            dtype=dtype,
                            device=device)),
            cur_sh_degree=0,
        )

    def to_parameter(self):
        return self.__class__(
            xyz=torch.nn.Parameter(self.xyz),
            quaternion_xyzw=torch.nn.Parameter(self.quaternion_xyzw),
            scale=torch.nn.Parameter(self.scale),
            opacity=torch.nn.Parameter(self.opacity),
            color_sh=torch.nn.Parameter(self.color_sh),
            color_sh_base=torch.nn.Parameter(self.color_sh_base)
            if self.color_sh_base is not None else None,
            cur_sh_degree=self.cur_sh_degree,
            act_applied=self.act_applied,
            instance_id=self.instance_id,
        )

    @classmethod
    def zeros(cls,
              N: int,
              max_num_degree: int,
              split: bool = False,
              dtype: torch.dtype = torch.float32,
              device: torch.device = torch.device("cpu")):
        return cls(
            xyz=torch.zeros(N, 3, dtype=dtype, device=device),
            quaternion_xyzw=torch.zeros(N, 4, dtype=dtype, device=device),
            scale=torch.zeros(N, 3, dtype=dtype, device=device),
            opacity=torch.zeros(N, dtype=dtype, device=device),
            color_sh_base=torch.zeros(N, 3, dtype=dtype, device=device)
            if split else None,
            color_sh=torch.zeros(N, (max_num_degree + 1) *
                                 (max_num_degree + 1) - (1 if split else 0),
                                 3,
                                 dtype=dtype,
                                 device=device),
            cur_sh_degree=0,
        )


@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class GaussianModelOrigin(GaussianModelBase):
    @property
    def quaternion_xyzw_act(self) -> torch.Tensor:
        if self.act_applied:
            return self.quaternion_xyzw
        return torch.nn.functional.normalize(self.quaternion_xyzw, p=2, dim=-1)

    @property
    def scale_act(self) -> torch.Tensor:
        if self.act_applied:
            return self.scale
        return torch.exp(self.scale)

    def inverse_scale_act(self, scale_act: torch.Tensor) -> torch.Tensor:
        return torch.log(scale_act)

    @property
    def opacity_act(self) -> torch.Tensor:
        if self.act_applied:
            return self.opacity
        return torch.sigmoid(self.opacity)


@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class GaussianModelOriginFused(GaussianModelBase):
    @property
    def quaternion_xyzw_act(self) -> torch.Tensor:
        return self.quaternion_xyzw

    @property
    def scale_act(self) -> torch.Tensor:
        return self.scale

    def inverse_scale_act(self, scale_act: torch.Tensor) -> torch.Tensor:
        return torch.log(scale_act)

    @property
    def opacity_act(self) -> torch.Tensor:
        return self.opacity

    @property
    def fused_quaternion_xyzw_act_op(self) -> tuple[str, str] | None:
        return ("normalize", "normalize_grad_out")

    @property
    def fused_scale_act_op(self) -> tuple[str, str] | None:
        return ("exponential", "exponential_grad_out")

    @property
    def fused_opacity_act_op(self) -> tuple[str, str] | None:
        return ("sigmoid", "sigmoid_grad_out")

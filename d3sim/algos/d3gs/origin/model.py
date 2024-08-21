import enum
import math
from typing import Annotated, Any, Self, Type

from d3sim.algos.d3gs.data_base import D3simDataset
from d3sim.constants import D3SIM_DEFAULT_DEVICE
from d3sim.core import dataclass_dispatch as dataclasses
from d3sim.core.pytorch.hmt import HomogeneousTensor
import torch
from d3sim.core import arrcheck
import abc
import pccm 
import numpy as np 
from d3sim.csrc.gs3d import SHConstants
from d3sim.algos.d3gs.ops import simple_knn

from d3sim.algos.d3gs.base import GaussianModelBase, GaussianModelProxyBase, GaussianCoreFields

def rgb_to_sh(rgb):
    return (rgb - 0.5) / SHConstants.C0

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def points_to_gaussian_init_scale(points: torch.Tensor):
    knn_res = simple_knn(points, 3)
    knn_res_mean2 = torch.square(knn_res).mean(1)
    dist2 = torch.clamp_min(knn_res_mean2, 0.0000001)       
    scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
    return scales



@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject, kw_only=True)
class GaussianModelOriginBase(GaussianModelBase, abc.ABC):
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
    def fused_quaternion_xyzw_act_op(self) -> tuple[str, str]:
        """there are three component support fused act to save gpu memory.
        when you implement fused op, you should return identity
        in component_act property.
        """
        return ("identity", "identity")

    @property
    def fused_scale_act_op(self) -> tuple[str, str]:
        return ("identity", "identity")

    @property
    def fused_opacity_act_op(self) -> tuple[str, str]:
        return ("identity", "identity")

    @property
    def color_sh_act(self) -> torch.Tensor:
        return self.color_sh

    @property
    def color_sh_base_act(self) -> torch.Tensor | None:
        return self.color_sh_base

    @property
    def color_sh_degree(self) -> int:
        dim = self.color_sh.shape[1]
        if self.color_sh_base is not None:
            dim += 1
        res = int(math.sqrt(dim)) - 1
        assert (res + 1) * (res + 1) == dim
        return res

    def get_unique_kernel_key(self):
        return (f"{self.color_sh_degree}_{self.cur_sh_degree}_{self.fused_quaternion_xyzw_act_op[0]}"
                f"_{self.fused_scale_act_op[0]}_{self.fused_opacity_act_op[0]}_{self.color_sh_base is None}")

    def get_proxy_field_dict(self):
        res = {
            "xyz": self.xyz,
            "quaternion_xyzw": self.quaternion_xyzw,
            "scale": self.scale,
            "opacity": self.opacity,
            "color_sh": self.color_sh,
        }
        if self.color_sh_base is not None:
            res["color_sh_base"] = self.color_sh_base
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

    def create_proxy(self, code: pccm.FunctionCode, gaussian_idx: str, batch_idx: str, batch_size: int, is_bwd: bool = False) -> GaussianModelProxyBase:
        raise NotImplementedError

    def create_model_with_act(self):
        if self.act_applied:
            return self
        return dataclasses.replace(
            self,
            xyz=self.xyz_act,
            quaternion_xyzw=self.quaternion_xyzw_act,
            scale=self.scale_act,
            opacity=self.opacity_act,
            color_sh=self.color_sh_act,
            act_applied=True,
        )

    @classmethod
    def empty(cls,
              N: int,
              max_num_degree: int,
              split: bool = False,
              dtype: torch.dtype = torch.float32,
              device: torch.device | str = D3SIM_DEFAULT_DEVICE):
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
    def create_from_dataset(cls,
              ds: D3simDataset,
              device: torch.device | str = D3SIM_DEFAULT_DEVICE):
        model = cls.empty(ds.dataset_point_cloud.xyz.shape[0], 3, True, device=device)
        points = ds.dataset_point_cloud.xyz
        points_rgb = ds.dataset_point_cloud.rgb
        assert points_rgb is not None 
        assert model.xyz.shape[0] == points.shape[0]
        points_th = torch.from_numpy(points.astype(np.float32)).to(device).contiguous()
        points_rgb_th = torch.from_numpy(points_rgb.astype(np.float32)).to(device)
        points_color_sh = rgb_to_sh(points_rgb_th)
        if model.color_sh_base is None:
            model.color_sh[:, 0] = points_color_sh
        else:
            model.color_sh_base[:] = points_color_sh
        scales = points_to_gaussian_init_scale(points_th)
        model.scale[:] = scales
        quat_xyzw = torch.zeros((points_rgb_th.shape[0], 4), device=device)
        quat_xyzw[:, 3] = 1
        model.quaternion_xyzw[:] = quat_xyzw
        opacities = inverse_sigmoid(0.1 * torch.ones((points_th.shape[0], 1), dtype=torch.float32, device=D3SIM_DEFAULT_DEVICE))
        model.opacity[:] = opacities.reshape(-1)
        model.cur_sh_degree = 0
        model.xyz[:] = points_th
        return model


    @classmethod
    def zeros(cls,
              N: int,
              max_num_degree: int,
              split: bool = False,
              dtype: torch.dtype = torch.float32,
              device: torch.device = torch.device("cpu"),
              opacity: torch.Tensor | None = None):
        return cls(
            xyz=torch.zeros(N, 3, dtype=dtype, device=device),
            quaternion_xyzw=torch.zeros(N, 4, dtype=dtype, device=device),
            scale=torch.zeros(N, 3, dtype=dtype, device=device),
            opacity=opacity if opacity is not None else torch.zeros(N, dtype=dtype, device=device),
            color_sh_base=torch.zeros(N, 3, dtype=dtype, device=device)
            if split else None,
            color_sh=torch.zeros(N, (max_num_degree + 1) *
                                 (max_num_degree + 1) - (1 if split else 0),
                                 3,
                                 dtype=dtype,
                                 device=device),
            cur_sh_degree=0,
        )

    @classmethod
    def zeros_like_debug(cls,
              model: "GaussianModelOriginBase",
              opacity: torch.Tensor | None = None):
        """
        Args:
            model: model to copy the shape from
            opacity: when batch size is 1, we can reuse storage of opacity grad
                from rasterize to save some memory.
        """
        N = model.xyz.shape[0]
        dtype = model.xyz.dtype
        device = model.xyz.device
        split = model.color_sh_base is not None
        max_num_degree = model.color_sh_degree
        return cls.zeros(N, max_num_degree, split, dtype, device, opacity)

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class GaussianModelOrigin(GaussianModelOriginBase):
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

class GaussianModelProxyOrigin(GaussianModelProxyBase["GaussianModelOriginFused"]):
    def read_field(self, field: GaussianCoreFields, 
            out: str, normed_dir: str = ""):
        """Write code segment to get field from model ptr.
        """
        has_color_base = self._model.color_sh_base is not None
        cur_degree = self._model.cur_sh_degree
        max_degree = self._model.color_sh_degree
        if field == GaussianCoreFields.XYZ:
            self._code.raw(f"""
            auto {out} = op::reinterpret_cast_array_nd<3>(xyz_ptr)[{self._gaussian_idx}];
            """)
        elif field == GaussianCoreFields.QUATERNION_XYZW:
            self._code.raw(f"""
            auto {out}_raw = op::reinterpret_cast_array_nd<4>(quaternion_xyzw_ptr)[{self._gaussian_idx}];
            auto {out} = {out}_raw.op<op::{self._model.fused_quaternion_xyzw_act_op[0]}>();
            """)
        elif field == GaussianCoreFields.SCALE:
            self._code.raw(f"""
            auto {out}_raw = op::reinterpret_cast_array_nd<3>(scale_ptr)[{self._gaussian_idx}];
            auto {out} = {out}_raw.op<op::{self._model.fused_scale_act_op[0]}>();
            """)
        elif field == GaussianCoreFields.OPACITY:
            self._code.raw(f"""
            auto {out}_raw = op::reinterpret_cast_array_nd<1>(opacity_ptr)[{self._gaussian_idx}];
            auto {out} = {out}_raw.op<op::{self._model.fused_opacity_act_op[0]}>()[0];
            """)

        elif field == GaussianCoreFields.RGB:
            if not self._is_bwd:
                assert normed_dir != ""
                self._code.raw(f"""
                auto sh_ptr = op::reinterpret_cast_array_nd<3>(color_sh_ptr) + {self._gaussian_idx} * {(max_degree + 1) * (max_degree + 1) - has_color_base};
                """)
                if has_color_base:
                    self._code.raw(f"""
                    auto sh_base_ptr = op::reinterpret_cast_array_nd<3>(color_sh_base_ptr) + {self._gaussian_idx};
                    auto {out} = Gaussian3D::sh_dir_to_rgb<{cur_degree}>({normed_dir}, sh_ptr, sh_base_ptr);
                    """)
                else:
                    self._code.raw(f"""
                    auto {out} = Gaussian3D::sh_dir_to_rgb<{cur_degree}>({normed_dir}, sh_ptr);
                    """)
        else:
            raise NotImplementedError(f"field {field} not implemented yet.")
    
    def prepare_field_proxy(self):
        if self._batch_size == 1:
            return 
        else:
            if self._is_bwd:
                self._code.raw(f"""
                float dopacity_val_acc = 0.0f;
                tv::array<float, 3> dxyz_acc{{}};
                tv::array<float, 4> dquat_acc{{}};
                tv::array<float, 3> dscale_acc{{}};
                """) 

    def accumulate_field_grad(self, field: GaussianCoreFields, 
            out: str, grad: str, normed_dir: str = "", normed_dir_grad: str = ""):
        """Write code segment to get field from model ptr.
        """
        has_color_base = self._model.color_sh_base is not None
        cur_degree = self._model.cur_sh_degree
        max_degree = self._model.color_sh_degree
        if field == GaussianCoreFields.XYZ:
            if self._batch_size == 1:
                self._code.raw(f"""
                op::reinterpret_cast_array_nd<3>(xyz_grad_ptr)[{self._gaussian_idx}] = {grad};
                """)
            else:
                self._code.raw(f"""
                dxyz_acc += {grad};
                """)
        elif field == GaussianCoreFields.QUATERNION_XYZW:
            self._code.raw(f"""
            {grad} = {grad}.op<op::{self._model.fused_quaternion_xyzw_act_op[1]}>({out}, {out}_raw);
            """)
            if self._batch_size == 1:
                self._code.raw(f"""
                op::reinterpret_cast_array_nd<4>(quaternion_xyzw_grad_ptr)[{self._gaussian_idx}] = {grad};
                """)
            else:
                self._code.raw(f"""
                dquat_acc += {grad};
                """)
        elif field == GaussianCoreFields.SCALE:
            self._code.raw(f"""
            {grad} = {grad}.op<op::{self._model.fused_scale_act_op[1]}>({out}, {out}_raw);
            """)
            if self._batch_size == 1:
                self._code.raw(f"""
                op::reinterpret_cast_array_nd<3>(scale_grad_ptr)[{self._gaussian_idx}] = {grad};
                """)
            else:
                self._code.raw(f"""
                dscale_acc += {grad};
                """)
        elif field == GaussianCoreFields.OPACITY:
            self._code.raw(f"""
            {grad} = tv::array<float, 1>{{{grad}}}.op<op::{self._model.fused_opacity_act_op[1]}>(tv::array<float, 1>{{{out}}}, {out}_raw)[0];
            """)
            if self._batch_size == 1:
                self._code.raw(f"""
                opacity_grad_ptr[{self._gaussian_idx}] = {grad};
                """)
            else:
                self._code.raw(f"""
                dopacity_val_acc += {grad};
                """)

        elif field == GaussianCoreFields.RGB:
            assert normed_dir != "" and normed_dir_grad != ""
            self._code.raw(f"""
            auto sh_ptr = op::reinterpret_cast_array_nd<3>(color_sh_ptr) + {self._gaussian_idx} * {(max_degree + 1) * (max_degree + 1) - has_color_base};
            auto dsh_ptr = op::reinterpret_cast_array_nd<3>(color_sh_grad_ptr) + i * {(max_degree + 1) * (max_degree + 1) - has_color_base};
            """)
            sh_grad_fn = "sh_dir_to_rgb_grad" if self._batch_size == 1 else "sh_dir_to_rgb_grad_batch"
            if has_color_base:
                self._code.raw(f"""
                auto dsh_base_ptr = op::reinterpret_cast_array_nd<3>(color_sh_base_grad_ptr) + {self._gaussian_idx};
                auto {normed_dir_grad} = Gaussian3D::{sh_grad_fn}<{cur_degree}>(color_grad, dsh_ptr,
                    {normed_dir}, sh_ptr, dsh_base_ptr);
                """)
            else:
                self._code.raw(f"""
                auto {normed_dir_grad} = Gaussian3D::{sh_grad_fn}<{cur_degree}>(color_grad, dsh_ptr,
                    {normed_dir}, sh_ptr);
                """)
        else:
            raise NotImplementedError(f"field {field} not implemented yet.")

    def save_accumulated_grad(self):
        if self._batch_size == 1:
            return 
        else:
            self._code.raw(f"""
            op::reinterpret_cast_array_nd<3>(xyz_grad_ptr)[{self._gaussian_idx}] = dxyz_acc;
            op::reinterpret_cast_array_nd<4>(quaternion_xyzw_grad_ptr)[{self._gaussian_idx}] = dquat_acc;
            op::reinterpret_cast_array_nd<3>(scale_grad_ptr)[{self._gaussian_idx}] = dscale_acc;
            opacity_grad_ptr[{self._gaussian_idx}] = dopacity_val_acc;
            """) 



@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class GaussianModelOriginFused(GaussianModelOriginBase):
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
    def fused_quaternion_xyzw_act_op(self) -> tuple[str, str]:
        return ("normalize", "normalize_grad_out")

    @property
    def fused_scale_act_op(self) -> tuple[str, str]:
        return ("exponential", "exponential_grad_out")

    @property
    def fused_opacity_act_op(self) -> tuple[str, str]:
        return ("sigmoid", "sigmoid_grad_out")

    def create_proxy(self, code: pccm.FunctionCode, gaussian_idx: str, batch_idx: str, batch_size: int, is_bwd: bool = False) -> GaussianModelProxyBase:
        return GaussianModelProxyOrigin(self, code, gaussian_idx, batch_idx, batch_size, is_bwd)

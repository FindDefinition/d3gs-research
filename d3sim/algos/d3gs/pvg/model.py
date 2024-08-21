from d3sim.algos.d3gs.base import GaussianModelProxyBase, GaussianModelOriginFused, GaussianCoreFields
from d3sim.core import dataclass_dispatch as dataclasses
import pccm
from typing import Annotated, Any, Type
from d3sim.core import arrcheck
import torch 

class PVGProxy(GaussianModelProxyBase):
    def validate_prep_inputs(self, inputs: dict[str, Any]): 
        assert "batch_ts" in inputs

    def prepare_field_proxy(self):
        self._code.raw(f"""
        float t_gaussian = t_ptr[{self._gaussian_idx}];
        tv::array<float, 3> velocity = op::reinterpret_cast_array_nd<3>(velocity_ptr)[{self._gaussian_idx}];
        float scale_t = scaling_t_ptr[{self._gaussian_idx}];
        float batch_t = batch_ts[{self._batch_idx}];
        tv::array<float, 3> _proxy_xyz = op::reinterpret_cast_array_nd<3>(xyz_ptr)[{self._gaussian_idx}];
        auto xyz_shm = _proxy_xyz + velocity * math_op_t::sin(batch_t - t_gaussian);
        """) 

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


@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject, kw_only=True)
class PVGGaussianModel(GaussianModelOriginFused):
    velocity: Annotated[torch.Tensor, arrcheck.ArrayCheck(["N", 3], arrcheck.F32)]
    scaling_t: Annotated[torch.Tensor, arrcheck.ArrayCheck(["N", 1], arrcheck.F32)]
    t: Annotated[torch.Tensor, arrcheck.ArrayCheck(["N", 1], arrcheck.F32)]

    def create_proxy(self, code: pccm.FunctionCode, gaussian_idx: str, batch_idx: str, batch_size: int, is_bwd: bool = False) -> GaussianModelProxyBase:
        return PVGProxy(self, code, gaussian_idx, batch_idx, batch_size, is_bwd)

from d3sim.algos.d3gs.base import GaussianModelProxyBase, GaussianCoreFields
from d3sim.core import dataclass_dispatch as dataclasses
import pccm
from d3sim.algos.d3gs.origin.model import GaussianModelOriginFused
from typing import Annotated, Any, Type
from d3sim.core import arrcheck
import torch 
import sympy
from cumm.inliner.sympy_codegen import sigmoid, tvarray_math_expr, VectorSymOperator, Scalar, Vector, VectorExpr



@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class PVGModelConfig:
    cycle: float = 0.2
    velocity_decay: float = 1.0

class PVGInputOperator(VectorSymOperator):
    def __init__(self, cfg: PVGModelConfig):
        super().__init__()
        self._cfg = cfg

    def forward(self, t_gaussian: Annotated[sympy.Symbol, Scalar()], velocity: Annotated[sympy.Symbol, Vector(3)], scale_t: Annotated[sympy.Symbol, Scalar()], batch_t: Annotated[sympy.Symbol, Scalar()], _proxy_xyz: Annotated[sympy.Symbol, Vector(3)], _proxy_opacity_val: Annotated[sympy.Symbol, Scalar()], time_shift: Annotated[sympy.Symbol, Scalar()]) -> dict[str, VectorExpr]:
        a = 1.0 / self._cfg.cycle * sympy.pi * 2.0
        scale_t_act = sympy.exp(scale_t)
        xyz_shm = _proxy_xyz + velocity * sympy.sin((batch_t - t_gaussian) * a) / a
        inst_velocity = velocity * sympy.exp(-scale_t_act / self._cfg.cycle / 2.0 * self._cfg.velocity_decay)
        xyz_shm += inst_velocity * time_shift
        marginal_t = sympy.exp(-0.5 * (t_gaussian - batch_t) * (t_gaussian - batch_t) / scale_t_act / scale_t_act)
        opacity_t = sigmoid(_proxy_opacity_val) * marginal_t
        return {
            "xyz": VectorExpr(xyz_shm, 3),
            "opacity": opacity_t,
            "marginal_t": marginal_t,
        }


class PVGProxy(GaussianModelProxyBase["PVGGaussianModel"]):
    def __init__(self, pvg_op: PVGInputOperator, model: "PVGGaussianModel", code: pccm.FunctionCode, gaussian_idx: str, batch_idx: str, batch_size: int, is_bwd: bool = False):
        super().__init__(model, code, gaussian_idx, batch_idx, batch_size, is_bwd)
        self.cfg = pvg_op._cfg
        self.pvg_op = pvg_op

    def validate_prep_inputs(self, inputs: dict[str, Any]): 
        assert "batch_ts" in inputs
        assert "time_shift" in inputs


    def read_field(self, field: GaussianCoreFields, 
            out: str, normed_dir: str = ""):
        """Write code segment to get field from model ptr.
        """
        has_color_base = self._model.color_sh_base is not None
        cur_degree = self._model.cur_sh_degree
        max_degree = self._model.color_sh_degree
        if field == GaussianCoreFields.XYZ:
            self._code.raw(f"""
            auto {out} = xyz_shm;
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
            auto {out} = opacity_t;
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
        use_sym = True
        self._code.raw(f"""
        float t_gaussian = t_ptr[{self._gaussian_idx}];
        tv::array<float, 3> velocity = op::reinterpret_cast_array_nd<3>(velocity_ptr)[{self._gaussian_idx}];
        float scale_t = (scaling_t_ptr[{self._gaussian_idx}]);
        float batch_t = batch_ts[{self._batch_idx}] - time_shift;
        tv::array<float, 3> _proxy_xyz = op::reinterpret_cast_array_nd<3>(xyz_ptr)[{self._gaussian_idx}];
        auto _proxy_opacity_val = opacity_ptr[{self._gaussian_idx}];

        """) 
        if use_sym:
            self._code.raw(f"""
            auto xyz_shm = {self.pvg_op.generate_result_expr_code("xyz")};
            auto opacity_t = {self.pvg_op.generate_result_expr_code("opacity")};
            auto marginal_t = {self.pvg_op.generate_result_expr_code("marginal_t")};
            invalid = marginal_t <= 0.05f;

            """)
        else:
            self._code.raw(f"""
            float a = 1.0f / {self.cfg.cycle} * M_PI * 2.0f;
            scale_t = math_op_t::exp(scale_t);
            auto xyz_shm = _proxy_xyz + velocity * math_op_t::sin((batch_t - t_gaussian) * a) / a;
            auto inst_velocity = velocity * math_op_t::exp(-scale_t / {self.cfg.cycle} / 2.0f * {self.cfg.velocity_decay});
            xyz_shm += inst_velocity * time_shift;
            auto marginal_t = math_op_t::exp(-0.5f * (t_gaussian - batch_t) * (t_gaussian - batch_t) / scale_t / scale_t);
            auto opacity_t = tv::array<float, 1>{{_proxy_opacity_val}}.op<op::sigmoid>()[0] * marginal_t;

            invalid = marginal_t <= 0.05f;

            """)
        if self._is_bwd:
            self._code.raw(f"""
            tv::array<float, 3> dscale_tmp{{}};
            float dt_tmp = 0;

            """)

        if self._batch_size == 1:
            return 
        else:
            if self._is_bwd:
                self._code.raw(f"""
                float dopacity_acc = 0.0f;
                float dt_acc = 0.0f;
                auto dscale_t_acc = 0.0f;
                tv::array<float, 3> dxyz_acc{{}};
                tv::array<float, 4> dquat_acc{{}};
                tv::array<float, 3> dscale_acc{{}};
                tv::array<float, 3> dvelo_acc{{}};
                """) 

    def accumulate_field_grad(self, field: GaussianCoreFields, 
            out: str, grad: str, normed_dir: str = "", normed_dir_grad: str = ""):
        """Write code segment to get field from model ptr.
        Call order of fields is always `scale|quat|opacity, rgb, xyz`,
        `|` means any order.
        """
        has_color_base = self._model.color_sh_base is not None
        cur_degree = self._model.cur_sh_degree
        max_degree = self._model.color_sh_degree
        if field == GaussianCoreFields.QUATERNION_XYZW:
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
            {grad} = tv::array<float, 1>{{{grad} * marginal_t}}.op<op::{self._model.fused_opacity_act_op[1]}>(tv::array<float, 1>{{{out}}}, _proxy_opacity_val[0])[0];
            """)
            if self._batch_size == 1:
                self._code.raw(f"""
                opacity_grad_ptr[{self._gaussian_idx}] = {grad};
                """)
            else:
                self._code.raw(f"""
                dopacity_acc += {grad};
                """)
        elif field == GaussianCoreFields.XYZ:
            if self._batch_size == 1:
                self._code.raw(f"""
                op::reinterpret_cast_array_nd<3>(xyz_grad_ptr)[{self._gaussian_idx}] = {grad};
                """)
            else:
                self._code.raw(f"""
                dxyz_acc += {grad};
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
    cycle: float = 0.2

    _pvg_op: PVGInputOperator | None = None

    def __post_init__(self):
        super().__post_init__()
        if self._pvg_op is None:
            self._pvg_op = PVGInputOperator(PVGModelConfig(cycle=self.cycle)).build()


    def create_proxy(self, code: pccm.FunctionCode, gaussian_idx: str, batch_idx: str, batch_size: int, is_bwd: bool = False) -> GaussianModelProxyBase:
        assert self._pvg_op is not None
            
        return PVGProxy(self._pvg_op, self, code, gaussian_idx, batch_idx, batch_size, is_bwd)
    
    def get_proxy_field_dict(self):
        res = super().get_proxy_field_dict()
        res["velocity"] = self.velocity
        res["scaling_t"] = self.scaling_t
        res["t"] = self.t
        return res

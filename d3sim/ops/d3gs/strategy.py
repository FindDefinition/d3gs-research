
from d3sim.core import dataclass_dispatch as dataclasses
from d3sim.core.arrcheck.dcbase import DataClassWithArrayCheck
from typing import Annotated, Callable, Literal
from d3sim.core.pytorch.hmt import HomogeneousTensor
from d3sim.ops.d3gs.base import GaussianModelBase, GaussianModelOrigin
import d3sim.core.arrcheck as ac
import torch
from d3sim.csrc.inliner import INLINER
import pccm 
from d3sim.ops.d3gs.render import GaussianSplatOutput 


@torch.no_grad()
def _update_optimizer_only(
    param_fn: Callable[[str, torch.Tensor], torch.nn.Parameter],
    optimizer_fn: Callable[[str, torch.Tensor], torch.Tensor],
    optimizers: dict[str, torch.optim.optimizer.Optimizer],
    names: list[str],
):
    """Update the parameters and the state in the optimizers with defined functions.

    Args:
        param_fn: A function that takes the name of the parameter and the parameter itself,
            and returns the new parameter.
        optimizer_fn: A function that takes the key of the optimizer state and the state value,
            and returns the new state value.
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        names: A list of key names to update. If None, update all. Default: None.
    """
    for name in names:
        optimizer = optimizers[name]
        for i, param_group in enumerate(optimizer.param_groups):
            p = param_group["params"][0]
            p_state = optimizer.state[p]
            del optimizer.state[p]
            for key in p_state.keys():
                if key != "step":
                    v = p_state[key]
                    p_state[key] = optimizer_fn(key, v)
            p_new = param_fn(name, p)
            optimizer.param_groups[i]["params"] = [p_new]
            optimizer.state[p_new] = p_state

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class GaussianTrainState(HomogeneousTensor):
    # duv_ndc = duv * (0.5 * shape_wh)
    # duv_ndc_length = torch.norm(duv_ndc, dim=1)
    # i.e. l2 norm of duv_ndc
    duv_ndc_length: Annotated[torch.Tensor, ac.ArrayCheck(["N"], ac.F32)]
    count: Annotated[torch.Tensor, ac.ArrayCheck(["N"], ac.F32)]
    max_radii: Annotated[torch.Tensor, ac.ArrayCheck(["N"], ac.F32)]
    scene_scale: float = 1.0

    @classmethod 
    def create(cls, num_gaussian: int, scene_scale: float = 1.0):
        return cls(
            duv_ndc_length=torch.zeros(num_gaussian, dtype=torch.float32),
            count=torch.zeros(num_gaussian, dtype=torch.float32),
            max_radii=torch.zeros(num_gaussian, dtype=torch.float32),
            scene_scale=scene_scale,
        )

    def reset(self):
        self.duv_ndc_length.zero_()
        self.count.zero_()
        self.max_radii.zero_()

@dataclasses.dataclass
class GaussianStrategyConfig:
    prune_opacity_thresh: float = 0.005
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    reset_every: int = 3000
    refine_every: int = 100
    absgrad: bool = False
    revised_opacity: bool = False
    verbose: bool = False

class GaussianStrategyBase:
    def __init__(self, cfg: GaussianStrategyConfig) -> None:
        self.cfg = cfg

    @torch.no_grad()
    def update(self, state: GaussianTrainState, out: GaussianSplatOutput, duv: torch.Tensor):
        duv = duv.clone()
        width, height = out.image_shape_wh
        duv[..., 0] = duv[..., 0] * (0.5 * width)
        duv[..., 1] = duv[..., 1] * (0.5 * height)
        # now duv is duv_ndc.
        duv_length = torch.norm(duv, dim=-1)

        assert out.radii is not None 
        radii = out.radii
        valid_filter = radii > 0
        gs_ids = torch.where(valid_filter)[1]  # [nnz]
        duv_length = duv_length[valid_filter]  # [nnz, 2]
        radii = radii[valid_filter]  # [nnz]
        state.duv_ndc_length.index_add_(0, gs_ids, duv_length)
        state.count.index_add_(
            0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
        )
        if self.cfg.refine_scale2d_stop_iter > 0:
            # Should be ideally using scatter max
            state.max_radii[gs_ids] = torch.maximum(
                state.max_radii[gs_ids],
                # normalize radii to [0, 1] screen space
                radii / float(max(width, height)),
            )

    @torch.no_grad()
    def refine_gs(
        self,
        model: GaussianModelBase,
        optimizers: dict[str, torch.optim.optimizer.Optimizer],
        state: GaussianTrainState,
        step: int,
    ) -> tuple[int, int]:
        
        num_gaussian = model.xyz.shape[0]
        duv_ndc_length = state.duv_ndc_length
        cnt = state.count
        scales = model.scale_act
        xyz = model.xyz_act
        opacity = model.opacity_act
        quat_xyzw = model.quaternion_xyzw_act
        refine_by_radii = step < self.cfg.refine_scale2d_stop_iter
        max_radii = state.max_radii
        origin_prune_mask_u8 = torch.empty(num_gaussian, dtype=torch.uint8, device=duv_ndc_length.device)

        should_dupti_u8 = torch.empty(num_gaussian, dtype=torch.uint8, device=duv_ndc_length.device)
        should_split_u8 = torch.empty(num_gaussian, dtype=torch.uint8, device=duv_ndc_length.device)
        scene_scale = state.scene_scale
        code = pccm.code()
        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<float>;

        auto cnt_val = $cnt[i];
        auto duv_ndc_length_val = $duv_ndc_length[i];
        bool is_grad_high = (duv_ndc_length_val / cnt_val) >= $(self.cfg.grow_grad2d);
        auto scale = op::reinterpret_cast_array_nd<3>($scales)[i];
        auto opacity_val = $opacity[i];
        """)
        if model.fused_scale_act_op is not None:
            # scales don't contains act, apply in kernel
            code.raw(f"""
            scale = scale.op<op::{model.fused_scale_act_op[0]}>();
            """)
        if model.fused_opacity_act_op is not None:
            code.raw(f"""
            opacity_val = tv::array<float, 1>{{opacity_val}}.op<op::{model.fused_opacity_act_op[0]}>()[0];
            """)
        code.raw(f"""
        auto scale_max = scale.op<op::reduce_max>();
        bool is_small = scale_max <= $(self.cfg.grow_scale3d) * $scene_scale;
        bool is_dupli = is_grad_high && is_small;
        $should_dupti_u8[i] = is_dupli;
        bool is_large = !is_small;
        float max_radii_val = $max_radii[i];
        bool prune_mask_origin = opacity_val < $(self.cfg.prune_opacity_thresh);
        prune_mask_origin |= scale_max > $(self.cfg.prune_scale3d) * $scene_scale;
        if ($refine_by_radii){{
            is_large |= max_radii_val > $(self.cfg.grow_scale2d);
            prune_mask_origin |= max_radii_val > $(self.cfg.prune_scale2d);
        }}
        bool is_split = is_grad_high && is_large;
        $should_split_u8[i] = is_split;
        $origin_prune_mask_u8[i] = prune_mask_origin;

        """)
        INLINER.kernel_1d(f"grow gs get mask_{model.fused_scale_act_op}", model.xyz.shape[0], 0, code)
        should_split = should_split_u8 > 0
        should_dupti = should_dupti_u8 > 0
        origin_should_prune = origin_prune_mask_u8 > 0
        dupti_should_prune = origin_should_prune[should_dupti]
        n_split = int(should_split.sum().item())
        n_dupti = int(should_dupti.sum().item())
        should_split_inds = torch.nonzero(should_split).view(-1)
        should_dupti_inds = torch.nonzero(should_dupti).view(-1)
        origin_model_inds = torch.arange(num_gaussian, device=duv_ndc_length.device)
        new_model_inds = torch.cat([should_dupti_inds, should_split_inds, should_split_inds], dim=0)

        # create new model (except origin) and assign duplicated and split GSs
        new_model = model.empty(int(n_dupti + n_split * 2), model.color_sh_degree)
        new_model.cur_sh_degree = model.cur_sh_degree
        new_model_should_prune = torch.empty(n_dupti + n_split * 2, dtype=torch.bool, device=duv_ndc_length.device)
        new_model[:n_dupti] = model[should_dupti]
        new_model_should_prune[:n_dupti] = dupti_should_prune
        model_split = model[should_split_inds]
        new_model[n_dupti:n_dupti + n_split] = model_split
        new_model[n_dupti + n_split:] = model_split

        # create new state and assign duplicated and split GS states
        new_state = GaussianTrainState.create(n_dupti + n_split * 2, state.scene_scale)
        new_state[:n_dupti] = state[should_dupti]
        new_state[n_dupti:n_dupti + n_split] = state[should_split_inds]
        new_state[n_dupti + n_split:] = state[should_split_inds]

        # do new location and scale calc on split points
        new_model_split = new_model[n_dupti:]
        unit_normals = torch.randn(n_split * 2, 3, device=model.xyz.device)
        split_prune_mask_u8 = torch.empty(n_split * 2, dtype=torch.uint8, device=duv_ndc_length.device)

        code = pccm.code()
        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<float>;
        auto quat = op::reinterpret_cast_array_nd<4>($(new_model_split.quaternion_xyzw))[i];
        auto scale = op::reinterpret_cast_array_nd<3>($(new_model_split.scale))[i];
        auto xyz = op::reinterpret_cast_array_nd<3>($(new_model_split.xyz))[i];
        auto opacity_val = $(new_model_split.opacity)[i];
        """)
        if model.fused_scale_act_op is not None:
            # scales don't contains act, apply in kernel
            code.raw(f"""
            scale = scale.op<op::{model.fused_scale_act_op[0]}>();
            """)
        if model.fused_quaternion_xyzw_act_op is not None:
            code.raw(f"""
            quat = quat.op<op::{model.fused_quaternion_xyzw_act_op[0]}>();
            """)
        if model.fused_opacity_act_op is not None:
            code.raw(f"""
            opacity_val = tv::array<float, 1>{{opacity_val}}.op<op::{model.fused_opacity_act_op[0]}>()[0];
            """)
        code.raw(f"""
        // calc new xyz and scale
        auto qmat = quat.op<op::uqmat_colmajor>();
        auto normal_vec3 = op::reinterpret_cast_array_nd<3>($unit_normals)[i];
        auto new_xyz = qmat.op<op::mv_colmajor>(normal_vec3 * scale) + xyz;

        auto new_scaling = scale / (0.8f * 2);

        op::reinterpret_cast_array_nd<3>($(new_model_split.xyz))[i] = new_xyz;
        op::reinterpret_cast_array_nd<3>($(new_model_split.scale))[i] = math_op_t::log(new_scaling);
        // calc prune mask
        bool prune_mask_origin = opacity_val < $(self.cfg.prune_opacity_thresh);
        auto new_scaling_max = new_scaling.op<op::reduce_max>();
        prune_mask_origin |= new_scaling_max > $(self.cfg.prune_scale3d) * $scene_scale;
        if ($refine_by_radii){{
            prune_mask_origin |= max_radii_val > $(self.cfg.prune_scale2d);
        }}
        $split_prune_mask_u8[i] = prune_mask_origin;
        """)
        kernel_name = f"grow gs do split_{model.fused_scale_act_op}_{model.fused_quaternion_xyzw_act_op}_{model.fused_opacity_act_op}"
        INLINER.kernel_1d(kernel_name, n_split * 2, 0, code)

        new_model_should_prune[n_dupti:] = split_prune_mask_u8 > 0

        model_origin_prune = model[origin_should_prune]
        new_model_prune = model[new_model_should_prune]
        origin_model_inds_prune = origin_model_inds[origin_should_prune]
        new_model_inds_prune = new_model_inds[new_model_should_prune]

        state_origin_prune = state[origin_should_prune]
        new_state_prune = state[new_model_should_prune]

        res_model = model_origin_prune.concat(new_model_prune)
        res_state = state_origin_prune.concat(new_state_prune)

        model.assign_inplace(res_model)
        state.assign_inplace(res_state)
        state.reset()

        res_inds = torch.cat([origin_model_inds_prune, new_model_inds_prune], dim=0)
        select_gs_optimizer_by_mask(model, optimizers, res_inds)
        return n_dupti, n_split


@torch.no_grad()
def select_gs_optimizer_by_mask(
    model: GaussianModelBase,
    optimizers: dict[str, torch.optim.optimizer.Optimizer],
    inds: torch.Tensor,
):
    """Inplace remove the Gaussian with the given mask.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to remove the Gaussians.
    """

    def param_fn(name: str, p: torch.Tensor) -> torch.nn.Parameter:
        return getattr(model, name)

    def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
        return v[inds]
    # update the parameters and the state in the optimizers
    _update_optimizer_only(param_fn, optimizer_fn, optimizers, [f.name for f in dataclasses.fields(model)])


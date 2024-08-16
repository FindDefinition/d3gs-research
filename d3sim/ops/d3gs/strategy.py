
from d3sim.core import dataclass_dispatch as dataclasses
from d3sim.core.arrcheck.dcbase import DataClassWithArrayCheck
from typing import Annotated, Callable, Literal, Mapping
from d3sim.core.pytorch.hmt import HomogeneousTensor
from d3sim.ops.d3gs.base import GaussianModelBase, GaussianModelOrigin
import d3sim.core.arrcheck as ac
import torch
from d3sim.csrc.inliner import INLINER
import pccm 
from d3sim.ops.d3gs.render import GaussianSplatOutput 
from d3sim.ops.d3gs import config_def
from d3sim.core.debugtools import get_from_store

@torch.no_grad()
def _update_optimizer_only(
    param_fn: Callable[[str, torch.Tensor], torch.nn.Parameter],
    optimizer_fn: Callable[[str, torch.Tensor], torch.Tensor],
    optimizers: Mapping[str, torch.optim.Optimizer] | torch.optim.Optimizer,
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
    if isinstance(optimizers, torch.optim.Optimizer):
        for i, param_group in enumerate(optimizers.param_groups):
            p = param_group["params"][0]
            name = param_group["name"]
            if name not in names:
                continue 
            p_state = optimizers.state[p]
            del optimizers.state[p]
            for key in p_state.keys():
                if key != "step":
                    v = p_state[key]
                    p_state[key] = optimizer_fn(key, v)
            p_new = param_fn(name, p)
            optimizers.param_groups[i]["params"] = [p_new]
            optimizers.state[p_new] = p_state
    else:
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

def _compare_optimizer_params(
    optimizers: Mapping[str, torch.optim.Optimizer],
    ref_optimizers: Mapping[str, torch.optim.Optimizer],
):

    for name in optimizers.keys():
        optimizer = optimizers[name]
        ref_optimizer = ref_optimizers[name]
        for i, param_group in enumerate(optimizer.param_groups):
            p = param_group["params"][0]
            ref_p = ref_optimizer.param_groups[i]["params"][0]
            p_state = optimizer.state[p]
            ref_p_state = ref_optimizer.state[ref_p]
            for key in p_state.keys():
                if key != "step":
                    v = p_state[key]
                    ref_v = ref_p_state[key]
                    print(f'{param_group["name"]} State-{key}', torch.linalg.norm(v - ref_v))
            print(param_group["name"], torch.linalg.norm(p - ref_p))

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
    def create(cls, num_gaussian: int, scene_scale: float = 1.0, device=None):
        return cls(
            duv_ndc_length=torch.zeros(num_gaussian, dtype=torch.float32, device=device),
            count=torch.zeros(num_gaussian, dtype=torch.float32, device=device),
            max_radii=torch.zeros(num_gaussian, dtype=torch.float32, device=device),
            scene_scale=scene_scale,
        )

    def reset(self):
        self.duv_ndc_length.zero_()
        self.count.zero_()
        self.max_radii.zero_()

class GaussianStrategyBase:
    def __init__(self, cfg: config_def.Strategy) -> None:
        self.cfg = cfg

    @torch.no_grad()
    def update(self, state: GaussianTrainState, out: GaussianSplatOutput, duv: torch.Tensor):
        duv = duv.clone()
        width, height = out.image_shape_wh
        max_width_height = max(width, height)
        # print(width, height)
        duv[..., 0] *= width / 2.0
        duv[..., 1] *= height / 2.0
        # now duv is duv_ndc.
        duv_length = torch.norm(duv, dim=-1)
        assert out.radii is not None 
        radii = out.radii
        valid_filter = radii > 0
        gs_ids_res = torch.where(valid_filter)  # [nnz]
        gs_ids = gs_ids_res[-1]
        duv_length = duv_length[valid_filter]  # [nnz, 2]
        radii = radii[valid_filter]  # [nnz]

        need_refine_scale2d = self.cfg.refine_scale2d_stop_iter > 0
        state.duv_ndc_length.index_add_(0, gs_ids, duv_length)
        state.count.index_add_(
            0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
        )
        if need_refine_scale2d > 0:
            # Should be ideally using scatter max
            state.max_radii[gs_ids] = torch.maximum(
                state.max_radii[gs_ids],
                # normalize radii to [0, 1] screen space
                radii / float(max(width, height)),
            )
        
    @torch.no_grad()
    def update_v2(self, state: GaussianTrainState, out: GaussianSplatOutput, duv: torch.Tensor, batch_size: int = 1):
        width, height = out.image_shape_wh
        max_width_height = max(width, height)
        N = duv.reshape(-1, 2).shape[0]
        assert out.radii is not None 
        duv_ndc_length = state.duv_ndc_length
        count = state.count
        radii = out.radii
        max_radii = state.max_radii
        need_refine_scale2d = self.cfg.refine_scale2d_stop_iter > 0
        assert N == radii.reshape(-1).shape[0]
        assert N == state.duv_ndc_length.shape[0]
        assert N == count.shape[0]
        assert N == max_radii.shape[0]

        INLINER.kernel_1d(f"grow gs update_{need_refine_scale2d}", duv.shape[0], 0, f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<float>;
        auto duv_val = op::reinterpret_cast_array_nd<2>($duv)[i];
        auto radii_val = $radii[i];
        duv_val[0] *= 0.5f * $width * $batch_size;
        duv_val[1] *= 0.5f * $height * $batch_size;
        auto duv_length = duv_val.op<op::length>();
        if (radii_val > 0){{
            $duv_ndc_length[i] += duv_length;
            $count[i] += 1;
            if ({pccm.boolean(need_refine_scale2d)}){{
                auto max_radii_val = $max_radii[i];
                max_radii_val = math_op_t::max(max_radii_val, radii_val / float($max_width_height));
                max_radii[i] = max_radii_val;
            }}
        }}
        """)

    @torch.no_grad()
    def refine_gs(
        self,
        model: GaussianModelBase,
        optimizers: Mapping[str, torch.optim.Optimizer],
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
        prune_too_big = step > self.cfg.reset_opacity_every
        max_radii = state.max_radii
        origin_prune_mask_u8 = torch.empty(num_gaussian, dtype=torch.uint8, device=duv_ndc_length.device)
        assert not refine_by_radii
        should_dupti_u8 = torch.empty(num_gaussian, dtype=torch.uint8, device=duv_ndc_length.device)
        should_split_u8 = torch.empty(num_gaussian, dtype=torch.uint8, device=duv_ndc_length.device)
        scene_scale = state.scene_scale
        assert num_gaussian == duv_ndc_length.shape[0]
        assert num_gaussian == cnt.shape[0]
        assert num_gaussian == scales.shape[0]
        assert num_gaussian == opacity.shape[0]
        assert num_gaussian == quat_xyzw.shape[0]
        assert num_gaussian == max_radii.shape[0]
        
        code = pccm.code()
        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<float>;

        auto cnt_val = $cnt[i];
        auto duv_ndc_length_val = $duv_ndc_length[i];
        bool is_grad_high = (duv_ndc_length_val / math_op_t::max(cnt_val, 1.0f)) >= $(self.cfg.grow_grad2d);
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
        # print("self.cfg.grow_scale3d", self.cfg.grow_grad2d, self.cfg.grow_scale3d, scene_scale, model.fused_opacity_act_op)
        code.raw(f"""
        auto scale_max = scale.op<op::reduce_max>();
        bool is_small = scale_max <= $(self.cfg.grow_scale3d) * $scene_scale;
        bool is_dupli = is_grad_high && is_small;
        $should_dupti_u8[i] = is_dupli;
        bool is_large = !is_small;
        float max_radii_val = $max_radii[i];
        bool prune_mask_origin = opacity_val < $(self.cfg.prune_opacity_thresh);
        if ($prune_too_big){{
            prune_mask_origin |= scale_max > $(self.cfg.prune_scale3d) * $scene_scale;
        }}
        if ($refine_by_radii){{
            is_large |= max_radii_val > $(self.cfg.grow_scale2d);
            prune_mask_origin |= max_radii_val > $(self.cfg.prune_scale2d);
        }}
        bool is_split = is_grad_high && is_large;
        $should_split_u8[i] = is_split;
        $origin_prune_mask_u8[i] = prune_mask_origin || is_split;

        """)
        INLINER.kernel_1d(f"grow gs get mask_{model.fused_scale_act_op}", model.xyz.shape[0], 0, code)
        should_split = should_split_u8 > 0
        should_dupti = should_dupti_u8 > 0
        origin_should_prune = origin_prune_mask_u8 > 0
        dupti_should_prune = origin_should_prune[should_dupti]
        n_split = int(should_split.sum().item())
        n_dupti = int(should_dupti.sum().item())
        # print("n_split", n_split, "n_dupti", n_dupti)
        should_split_inds = torch.nonzero(should_split).view(-1)
        should_dupti_inds = torch.nonzero(should_dupti).view(-1)
        origin_model_inds = torch.arange(num_gaussian, device=duv_ndc_length.device)
        new_model_inds = torch.cat([should_dupti_inds, should_split_inds, should_split_inds], dim=0)

        # create new model (except origin) and assign duplicated and split GSs
        new_model = model.empty(int(n_dupti + n_split * 2), model.color_sh_degree, split=True, device=model.xyz.device)
        new_model.cur_sh_degree = model.cur_sh_degree
        new_model_should_prune = torch.empty(n_dupti + n_split * 2, dtype=torch.bool, device=duv_ndc_length.device)
        new_model[:n_dupti] = model[should_dupti]
        new_model_should_prune[:n_dupti] = dupti_should_prune
        model_split = model[should_split_inds]
        new_model[n_dupti:n_dupti + n_split] = model_split
        new_model[n_dupti + n_split:] = model_split

        # create new state and assign duplicated and split GS states
        new_state = GaussianTrainState.create(n_dupti + n_split * 2, state.scene_scale, device=model.xyz.device)
        new_state.scene_scale = state.scene_scale

        new_state[:n_dupti] = state[should_dupti]
        new_state[n_dupti:n_dupti + n_split] = state[should_split_inds]
        new_state[n_dupti + n_split:] = state[should_split_inds]
        new_state_max_radii = new_state.max_radii

        # do new location and scale calc on split points
        new_model_split = new_model[n_dupti:]
        quaternion_xyzw_split = new_model_split.quaternion_xyzw_act
        scale_split = new_model_split.scale_act
        xyz_split = new_model_split.xyz_act
        opacity_split = new_model_split.opacity_act
        # torch.manual_seed(step)
        unit_normals = torch.randn(n_split * 2, 3, device=model.xyz.device)
        split_prune_mask_u8 = torch.empty(n_split * 2, dtype=torch.uint8, device=duv_ndc_length.device)
        code = pccm.code()
        assert n_split * 2 == xyz_split.shape[0]
        assert n_split * 2 == scale_split.shape[0]
        assert n_split * 2 == opacity_split.shape[0]
        assert n_split * 2 == quaternion_xyzw_split.shape[0]

        code.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<float>;
        auto quat = op::reinterpret_cast_array_nd<4>($quaternion_xyzw_split)[i];
        auto scale = op::reinterpret_cast_array_nd<3>($scale_split)[i];
        auto xyz = op::reinterpret_cast_array_nd<3>($xyz_split)[i];
        auto opacity_val = $opacity_split[i];
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
        float max_radii_val = $new_state_max_radii[i];

        op::reinterpret_cast_array_nd<3>($(new_model_split.xyz))[i] = new_xyz;
        op::reinterpret_cast_array_nd<3>($(new_model_split.scale))[i] = new_scaling.op<op::log>();
        // calc prune mask
        bool prune_mask_origin = opacity_val < $(self.cfg.prune_opacity_thresh);
        auto new_scaling_max = new_scaling.op<op::reduce_max>();
        if($prune_too_big){{
            prune_mask_origin |= new_scaling_max > $(self.cfg.prune_scale3d) * $scene_scale;
        }}
        if ($refine_by_radii){{
            prune_mask_origin |= max_radii_val > $(self.cfg.prune_scale2d);
        }}
        $split_prune_mask_u8[i] = prune_mask_origin;
        """)
        kernel_name = f"grow gs do split_{model.fused_scale_act_op}_{model.fused_quaternion_xyzw_act_op}_{model.fused_opacity_act_op}"
        INLINER.kernel_1d(kernel_name, n_split * 2, 0, code)
        # print("SPLIT XYZ", new_model_split.xyz.shape, new_model_split.xyz)
        # model_debug = model[~should_split].concat(new_model)
        # model_debug_dict = model_debug.get_all_tensor_fields()

        new_model_should_prune[n_dupti:] = split_prune_mask_u8 > 0

        model_origin_prune = model[~origin_should_prune]
        new_model_prune = new_model[~new_model_should_prune]
        origin_model_inds_prune = origin_model_inds[~origin_should_prune]
        new_model_inds_prune = new_model_inds[~new_model_should_prune]

        state_origin_prune = state[~origin_should_prune]
        new_state_prune = new_state[~new_model_should_prune]

        res_model = model_origin_prune.concat(new_model_prune)
        res_state = state_origin_prune.concat(new_state_prune)
        res_inds = torch.cat([origin_model_inds_prune, new_model_inds_prune], dim=0)

        # print(torch.linalg.norm(res_model.xyz - ))

        model.assign_inplace(res_model.to_parameter())
        state.assign_inplace(res_state)
        state.reset()
        select_gs_optimizer_by_mask(model, optimizers, res_inds)
        zero_optimizer_by_range(model, optimizers, (origin_model_inds_prune.shape[0], origin_model_inds_prune.shape[0] + new_model_inds_prune.shape[0]))
        # print(model.xyz.shape, n_dupti, n_split)
        # print(res_inds.shape, res_inds)
        # print("------")
        # debug_optim = get_from_store("debug")
        # _compare_optimizer_params(optimizers, debug_optim)
        # print(n_split, n_dupti, res_inds.shape, res_model.xyz.shape)
        # breakpoint()

        return n_dupti, n_split


@torch.no_grad()
def select_gs_optimizer_by_mask(
    model: GaussianModelBase,
    optimizers: Mapping[str, torch.optim.Optimizer],
    inds: torch.Tensor,
):
    """Inplace remove the Gaussian with the given mask.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to remove the Gaussians.
    """

    def param_fn(name: str, p: torch.Tensor) -> torch.nn.Parameter:
        res = getattr(model, name)
        assert isinstance(res, torch.nn.Parameter)
        return res

    def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
        return v[inds]
    # update the parameters and the state in the optimizers
    _update_optimizer_only(param_fn, optimizer_fn, optimizers, list(model.get_all_tensor_fields().keys()))


@torch.no_grad()
def zero_optimizer_by_range(
    model: GaussianModelBase,
    optimizers: Mapping[str, torch.optim.Optimizer],
    interval: tuple[int, int],
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
        v[interval[0]:interval[1]].zero_()
        return v
    # update the parameters and the state in the optimizers
    _update_optimizer_only(param_fn, optimizer_fn, optimizers, list(model.get_all_tensor_fields().keys()))


@torch.no_grad()
def reset_param_in_optim(
    model: GaussianModelBase,
    optimizers: Mapping[str, torch.optim.Optimizer],
    name: str,
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
        return torch.zeros_like(v)
    # update the parameters and the state in the optimizers
    _update_optimizer_only(param_fn, optimizer_fn, optimizers, [name])

from contextlib import contextmanager, nullcontext
import copy
from pydoc import doc
from pyexpat import model
from tabnanny import verbose
import time
from typing import Annotated, Any, Callable, Literal
from d3sim.core import dataclass_dispatch as dataclasses
from d3sim.core.arrcheck.dcbase import DataClassWithArrayCheck
import d3sim.core.arrcheck as ac
from d3sim.data.scene_def.camera import BasicPinholeCamera
from d3sim.algos.d3gs.base import GaussianCoreFields, GaussianModelBase
from d3sim.algos.d3gs.origin.model import GaussianModelOriginBase

from d3sim.csrc.inliner import INLINER, create_default_inliner
import numpy as np
import math
import torch

import pccm
from cumm.gemm.codeops import div_up
from d3sim.constants import D3SIM_DEFAULT_DEVICE, IsAppleSiliconMacOs
import cumm.tensorview as tv
from d3sim.algos.d3gs.origin.data.utils.general_utils import strip_symmetric, build_scaling_rotation
from cumm.inliner import torch_tensor_to_tv, measure_and_print_torch, get_current_stream
from torch.autograd.function import once_differentiable
from d3sim.algos.d3gs.config_def import EarlyFilterAlgo, GaussianSplatConfig



_DEFAULT_ENABLE_32BIT_SORT = False if IsAppleSiliconMacOs else False


@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class BatchCameraParams:
    # 4x3 matrix
    cam2world_T: torch.Tensor
    focal_xy: torch.Tensor
    principal_point: torch.Tensor
    tan_fov: torch.Tensor
    image_shape_wh: tuple[int, int]


@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class GaussianSplatOpContext(DataClassWithArrayCheck):
    conic_opacity: Annotated[torch.Tensor, ac.ArrayCheck([-1, 4], ac.F32)]
    radii: Annotated[torch.Tensor, ac.ArrayCheck([-1], ac.I32)]
    uvs: Annotated[torch.Tensor, ac.ArrayCheck([-1, 2], ac.F32)]
    rgb_gaussian: Annotated[torch.Tensor | None, ac.ArrayCheck([-1, 3], ac.F32)]
    gaussian_idx_sorted: Annotated[torch.Tensor, ac.ArrayCheck([-1], ac.I32)]
    workload_ranges: Annotated[torch.Tensor, ac.ArrayCheck([-1, 2], ac.I32)]
    image_shape_wh: tuple[int, int]
    depths: Annotated[torch.Tensor, ac.ArrayCheck([-1], ac.F32)]
    cov3d_vecs: Annotated[torch.Tensor | None,
                          ac.ArrayCheck([-1, 6], ac.F32)] = None
    cov2d_vecs: Annotated[torch.Tensor | None,
                          ac.ArrayCheck([-1, 3], ac.F32)] = None
    sh_to_rgb_ne_0: Annotated[torch.Tensor | None,
                              ac.ArrayCheck([-1], ac.U8)] = None


@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class GaussianSplatOutput(DataClassWithArrayCheck):
    # no way to support checker for both nchw and nhwc.
    custom_features: Annotated[torch.Tensor | None,
                               ac.ArrayCheck(["B", -1, "E", -1], ac.F32)]
    T: Annotated[torch.Tensor, ac.ArrayCheck(["B", "H", "W"], ac.F32)]
    color: Annotated[torch.Tensor, ac.ArrayCheck(["B", -1, "E", -1], ac.F32)]
    depth: Annotated[torch.Tensor | None,
                     ac.ArrayCheck(["B", "H", "W"], ac.F32)] = None
    n_contrib: Annotated[torch.Tensor | None,
                         ac.ArrayCheck(["B", "H", "W"], ac.I32)] = None
    radii: Annotated[torch.Tensor | None, ac.ArrayCheck(["B", -1], ac.I32)] = None

    @property
    def image_shape_wh(self) -> tuple[int, int]:
        return (self.T.shape[2], self.T.shape[1])

    def select_batch(self, batch_idx: int):
        return GaussianSplatOutput(
            custom_features=None if self.custom_features is None else self.custom_features[batch_idx:batch_idx + 1],
            T=self.T[batch_idx:batch_idx + 1],
            color=self.color[batch_idx:batch_idx + 1],
            depth=None if self.depth is None else self.depth[batch_idx:batch_idx + 1],
            n_contrib=None if self.n_contrib is None else self.n_contrib[batch_idx:batch_idx + 1],
            radii=None if self.radii is None else self.radii[batch_idx:batch_idx + 1],
        )


@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class RasterizeGradientOutput(DataClassWithArrayCheck):
    duv: Annotated[torch.Tensor, ac.ArrayCheck(["N", 2], ac.F32)]
    dconic: Annotated[torch.Tensor, ac.ArrayCheck(["N", 3], ac.F32)]
    dopacity: Annotated[torch.Tensor, ac.ArrayCheck(["N"], ac.F32)]
    dcolor: Annotated[torch.Tensor | None, ac.ArrayCheck(["N", 3], ac.F32)]
    dcustom_features: Annotated[torch.Tensor | None,
                                ac.ArrayCheck(["N", -1], ac.F32)]
    dz: Annotated[torch.Tensor | None, ac.ArrayCheck(["N"], ac.F32)]

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class GaussianSplatGradients(DataClassWithArrayCheck):
    # no way to support checker for both nchw and nhwc.
    drgb: Annotated[torch.Tensor, ac.ArrayCheck(["B", -1, "F", -1], ac.F32)]
    dT: Annotated[torch.Tensor, ac.ArrayCheck(["B", "H", "W"], ac.F32)]
    ddepth: Annotated[torch.Tensor | None, ac.ArrayCheck(["B", "H", "W"], ac.F32)]
    dcustom_features: Annotated[torch.Tensor | None,
                                ac.ArrayCheck(["B", -1, "F", -1], ac.F32)]

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class CameraParamBundle:
    focal_xy_ppoint: torch.Tensor
    image_shape_wh: tuple[int, int]
    # frame id of each camera, used for object instance.
    frame_ids: torch.Tensor | None

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class CameraBundle(CameraParamBundle):
    cam2world_T: torch.Tensor

    @classmethod 
    def from_pinhole_cams(cls, cams: list[BasicPinholeCamera]):
        cam2worlds = [cam.pose.to_world for cam in cams]
        cam2world_Ts = [cam2world[:3].T for cam2world in cam2worlds]
        cam2world_Ts_onearray = np.stack(cam2world_Ts, axis=0).reshape(-1, 4, 3)
        cam2world_Ts_th = torch.from_numpy(np.ascontiguousarray(cam2world_Ts_onearray)).to(D3SIM_DEFAULT_DEVICE)

        focal_xys = np.stack([cam.focal_length for cam in cams], axis=0).reshape(-1, 2)
        ppoints = np.stack([cam.principal_point for cam in cams], axis=0).reshape(-1, 2)
        focal_xy_ppoints = np.concatenate([focal_xys, ppoints], axis=1)
        focal_xy_ppoints_th = torch.from_numpy(np.ascontiguousarray(focal_xy_ppoints)).to(D3SIM_DEFAULT_DEVICE)
        frame_local_ids = None 
        if cams[0].frame_local_id is not None:
            for cam in cams:
                assert cam.frame_local_id is not None
            frame_local_ids = torch.tensor([cam.frame_local_id for cam in cams], dtype=torch.int32, device=D3SIM_DEFAULT_DEVICE)
        cam_bundle = CameraBundle(focal_xy_ppoints_th, cams[0].image_shape_wh, frame_local_ids, cam2world_Ts_th)
        return cam_bundle

class GaussianSplatOp:

    def __init__(self, cfg: GaussianSplatConfig) -> None:
        # user shouldn't modify cfg in op, create new op instead.
        self._cfg = copy.deepcopy(cfg) 
        self._inliner = create_default_inliner()

        self._code_obj_caches: dict[str, pccm.FunctionCode] = {}

    @property 
    def is_nchw(self):
        return self._cfg.use_nchw

    @property 
    def is_rgba(self):
        return self._cfg.render_rgba

    def forward(
        self,
        model: GaussianModelOriginBase,
        cameras: BasicPinholeCamera | CameraBundle,
        training: bool = False,
        instance2world_T: torch.Tensor | None = None,
        custom_features: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        prep_user_inputs: dict[str, Any] | None = None,
    ):
        enable_verbose = self._cfg.verbose
        # enable_verbose = False
        is_cam_bundle = isinstance(cameras, CameraBundle)
        batch_size = 1
        if not isinstance(cameras, BasicPinholeCamera):
            batch_size = cameras.cam2world_T.shape[0]
            # raise NotImplementedError
        with tv.measure_and_print("forward-0", enable=enable_verbose):
            assert model.act_applied, "model must be activated before render"
            num = len(model)
            scale_global = self._cfg.scale_global
            enable_32bit_sort = self._cfg.enable_32bit_sort
            camera = cameras
            image_shape_wh = camera.image_shape_wh

            width = image_shape_wh[0]
            height = image_shape_wh[1]
            resolution_wh_np = np.array([width, height], np.int32)
            enable_instance_render = False
            if isinstance(camera, BasicPinholeCamera):
                intrinsic = camera.intrinsic
                cam2world = camera.pose.to_world

                focal_xy_np = np.array([intrinsic[0, 0], intrinsic[1, 1]], np.float32)
                principal_point_np = np.array([intrinsic[0, 2], intrinsic[1, 2]],
                                            np.float32)
                cam2world_T_np = np.ascontiguousarray(cam2world[:3].T)
                cam2world_center_np = cam2world_T_np[3]
                if instance2world_T is not None:
                    assert camera.frame_local_id is not None 
                    assert model.instance_id is not None, "instance_id must be provided in model when render with instance"
                    enable_instance_render = True
                    frame_local_id = camera.frame_local_id

            else:
                if instance2world_T is not None:
                    assert camera.frame_ids is not None, "frame_ids must be provided when render with instance"
                    assert model.instance_id is not None, "instance_id must be provided in model when render with instance"
                    enable_instance_render = True
                    assert instance2world_T.ndim == 4, "instance2world_T must be 4D tensor [num_frame, num_instance, 4, 3]"
                    # instance2world_T: [num_frame, num_instance, 4, 3]
                    num_instance = instance2world_T.shape[1]
                frame_ids = camera.frame_ids
                cam2world_T_ten = camera.cam2world_T
                focal_xy_ppoint_ten = camera.focal_xy_ppoint

            tile_num_x = div_up(width, self._cfg.tile_size[0])
            tile_num_y = div_up(height, self._cfg.tile_size[1])
            if enable_32bit_sort:
                assert batch_size * tile_num_x * tile_num_y < 2**14, "32bit sort only support 14 bits tile idx"
            block_size = self._cfg.block_size
            tile_num_xy_np = np.array([tile_num_x, tile_num_y], np.int32)
            xyz = model.xyz
            # quat should be normed in user-defined quaternion_xyzw_act.
            instance_ids = model.instance_id

            depths = torch.empty(batch_size * num, dtype=torch.float32, device=xyz.device)
            radii = torch.empty(batch_size * num, dtype=torch.int32, device=xyz.device)
            uvs = torch.empty([batch_size * num, 2], dtype=torch.float32, device=xyz.device)
            conic_opacity = torch.empty([batch_size * num, 4],
                                        dtype=torch.float32,
                                        device=xyz.device)
            cov3d_vecs = None
            cov2d_vecs = None
            sh_to_rgb_ne_0 = None
            if training:
                # for backward only
                if not self._cfg.recalc_cov3d_in_bwd:
                    cov3d_vecs = torch.empty([batch_size * num, 6],
                                             dtype=torch.float32,
                                             device=xyz.device)
                cov2d_vecs = torch.empty([batch_size * num, 3],
                                         dtype=torch.float32,
                                         device=xyz.device)
                sh_to_rgb_ne_0 = torch.empty([batch_size * num],
                                             dtype=torch.uint8,
                                             device=xyz.device)
            # TODO when to use int64 tile_touched?
            tiles_touched = torch.empty(batch_size * num,
                                        dtype=torch.int64 if self._cfg.use_int64_tile_touched else torch.int32,
                                        device=xyz.device)
            rgb_gaussian = None 
            if not self._cfg.disable_builtin_rgb:
                rgb_gaussian = torch.empty([batch_size * num, 3],
                                        dtype=torch.float32,
                                        device=xyz.device)
            else:
                assert not self._cfg.use_nchw, "only support nhwc if disable builtin rgb."
        # debug_cov2d = torch.empty([num, 3], dtype=torch.float32, device=xyz.device)

        # debug_ten = torch.empty(num, dtype=torch.float32, device=xyz.device)
        custom_feat = custom_features
        num_custom_feat = 0
        if custom_feat is not None:
            num_custom_feat = custom_feat.shape[-1]
            assert custom_feat.shape[0] == batch_size * num
            assert not self.is_nchw, "custom feat don't support nchw, set use_nchw=False in config."
        t1 = time.time()
        use_proxy_model = self._cfg.use_proxy_model
        with measure_and_print_torch("gs3d_preprocess", enable=enable_verbose):
            prep_kernel_name = (
                f"gs3d_preprocess_{self._cfg.tile_size}_{training}_"
                f"{model.get_unique_kernel_key()}_"
                f"{is_cam_bundle}_{mask is None}")
            code_prep = pccm.code()
            lowpass_filter = self._cfg.gaussian_lowpass_filter
            eps = self._cfg.eps
            cov2d_radii_eigen_eps = self._cfg.cov2d_radii_eigen_eps
            gaussian_std_sigma = self._cfg.gaussian_std_sigma
            projected_clamp_factor = self._cfg.projected_clamp_factor
            code_prep.raw(f"""
            namespace op = tv::arrayops;
            using math_op_t = tv::arrayops::MathScalarOp<float>;
            auto resolution_wh = $resolution_wh_np;
            """)
            if mask is not None:
                code_prep.raw(f"""
                bool invalid = $mask[i];
                """)
            else:
                code_prep.raw(f"""
                bool invalid = false;
                """)
            if is_cam_bundle:
                # cam2world, focal and ppoint are tensor, load from gpu mem
                code_prep.raw(f"""
                int batch_idx = i / $num;
                auto gaussian_idx = i % $num;
                auto cam2world_T = op::reinterpret_cast_array_nd<4, 3>($cam2world_T_ten)[batch_idx];
                auto focal_xy_ppoint = op::reinterpret_cast_array_nd<4>($focal_xy_ppoint_ten)[batch_idx];
                auto focal_xy = op::slice<0, 2>(focal_xy_ppoint);
                auto principal_point = op::slice<2, 4>(focal_xy_ppoint);
                auto cam2world_R_T = op::slice<0, 3>(cam2world_T);
                auto cam2world_center = cam2world_T[3];

                """)
                if enable_instance_render:
                    code_prep.raw(f"""
                    auto frame_id = $frame_ids[batch_idx];
                    """)
            else:
                # cam2world, focal and ppoint are registers
                code_prep.raw(f"""
                auto gaussian_idx = i;
                auto cam2world_T = $cam2world_T_np;
                auto focal_xy = $focal_xy_np;
                auto principal_point = $principal_point_np;
                auto cam2world_R_T = op::slice<0, 3>(cam2world_T);
                auto cam2world_center = cam2world_T[3];
                """)
                if enable_instance_render:
                    code_prep.raw(f"""
                    auto frame_id = $frame_local_id;
                    """)
            if enable_instance_render:
                code_prep.raw(f"""
                auto instance_id = $instance_ids[gaussian_idx];
                if (instance_id >= 0){{
                    auto instance2world_T_val = op::reinterpret_cast_array_nd<4, 3>($instance2world_T)[frame_id * $num_instance + instance_id];
                    // get cam2instance_T, cam2instance = world2instance @ cam2world
                    auto cam2instance_T = instance2world_T_val.op<op::transform_matrix_colmajor_inverse>().op<op::transform_matrix_mm_nnn>(cam2world_T);
                    cam2world_R_T = op::slice<0, 3>(cam2instance_T);
                    cam2world_center = cam2instance_T[3];
                }}
                """)
            code_prep.raw(f"""
            auto tan_fov = 0.5f * resolution_wh.cast<float>() / focal_xy;
            // tan_fov[0] = -0.5463024898437905;
            // tan_fov[1] = -0.5463024898437905;
            // auto focal_xy_debug_for_proj = resolution_wh.cast<float>() / 2.0f / tan_fov;
            """)

            if not use_proxy_model:
                quat_xyzw = model.quaternion_xyzw
                scales = model.scale
                opacity = model.opacity

                fused_scale_act = model.fused_scale_act_op
                fused_q_act = model.fused_quaternion_xyzw_act_op
                fused_opacity_act = model.fused_opacity_act_op
                code_prep.raw(f"""
                auto point = op::reinterpret_cast_array_nd<3>($xyz)[gaussian_idx];
                auto point_cam = cam2world_R_T.op<op::mv_rowmajor>(point - cam2world_center);

                auto uvz = CameraOps::pos_cam_to_uv_no_distort<2, 0, 1>(point_cam, principal_point, focal_xy);
                auto uv = std::get<0>(uvz) - 0.5f;
                auto z_in_cam = std::get<1>(uvz);
                auto scale = op::reinterpret_cast_array_nd<3>($scales)[gaussian_idx];
                auto quat = op::reinterpret_cast_array_nd<4>($quat_xyzw)[gaussian_idx];
                quat = quat.op<op::{fused_q_act[0]}>();
                scale = scale.op<op::{fused_scale_act[0]}>();
                """)
                if training:
                    code_prep.raw(f"""
                    auto cov3d_vec = Gaussian3D::scale_quat_to_cov3d(scale, quat);
                    """)
                else:
                    # scale modifier should only be used during inference.
                    code_prep.raw(f"""
                    auto cov3d_vec = Gaussian3D::scale_quat_to_cov3d(scale * $scale_global, quat);
                    """)
                code_prep.raw(f"""
                auto cov2d_vec = Gaussian3D::project_gaussian_to_2d<float, 3>(point_cam, focal_xy, tan_fov, cam2world_R_T, cov3d_vec, $projected_clamp_factor);
                cov2d_vec[0] += {lowpass_filter}f;
                cov2d_vec[2] += {lowpass_filter}f;
                """)
                if self._cfg.enable_anti_aliasing:
                    code_prep.raw(f"""
                    auto cov2d_inv_and_det = Gaussian3D::gaussian_2d_inverse_and_det_with_comp(cov2d_vec, {lowpass_filter}f, $eps);
                    auto det_div_lowpass_sqrt_clamp = math_op_t::sqrt(math_op_t::max(0.000025f, std::get<2>(cov2d_inv_and_det))); 
                    """)
                else:
                    code_prep.raw(f"""
                    auto cov2d_inv_and_det = Gaussian3D::gaussian_2d_inverse_and_det(cov2d_vec, {lowpass_filter}f, $eps);
                    """)
                code_prep.raw(f"""
                auto opacity_val = $opacity[gaussian_idx];
                opacity_val = tv::array<float, 1>{{opacity_val}}.op<op::{fused_opacity_act[0]}>()[0];
                """)
                if self._cfg.enable_anti_aliasing:
                    code_prep.raw(f"""
                    opacity_val = opacity_val * det_div_lowpass_sqrt_clamp;
                    """)
                code_prep.raw(f"""
                auto cov2d_inv = std::get<0>(cov2d_inv_and_det);
                auto det = std::get<1>(cov2d_inv_and_det);
                auto radii_fp = math_op_t::ceil(Gaussian3D::get_gaussian_2d_ellipse(cov2d_vec, det, 
                    $cov2d_radii_eigen_eps, $gaussian_std_sigma));
                constexpr auto tile_size_xy = tv::array<int, 2>{{{self._cfg.tile_size[0]}, {self._cfg.tile_size[1]}}};
                auto tile_num_xy = tv::div_up(resolution_wh, tile_size_xy);
                auto tile_size_xy_float = tile_size_xy.cast<float>();
                // TODO use precise ellipse here instead of original coarse 3-sigma ellipse.
                auto gaussian_rect_min = ((uv - radii_fp) / tile_size_xy_float).cast<int>().op<op::clamp>(0, tile_num_xy);
                auto gaussian_rect_max = ((uv + radii_fp + tile_size_xy_float - 1) / tile_size_xy_float).cast<int>().op<op::clamp>(0, tile_num_xy);
                int rect_area = (gaussian_rect_max[0] - gaussian_rect_min[0]) * (gaussian_rect_max[1] - gaussian_rect_min[1]);
                // the maximum value of gaussian 2d in rasterize kernel is 1.0f
                // meanwhile alpha = G * opacity_val must > alpha_eps. 
                // G <= 1.0, if opacity_val <= alpha_eps, 
                // opacity_val * G always <= alpha_eps, this gaussian is always skipped in rasterize kernel.
                bool is_empty = (det == 0 || z_in_cam < 0.2f) || rect_area == 0 || invalid || opacity_val <= {self._cfg.alpha_eps}f;

                $tiles_touched[i] = is_empty ? 0 : rect_area;
                $radii[i] = is_empty ? 0 : int(radii_fp);
                if (is_empty){{
                    // calc color sh requires much IO, so we early exit here.
                    // this will cause warp divergence, but save io.
                    return;
                }}
                $depths[i] = z_in_cam;
                op::reinterpret_cast_array_nd<2>($uvs)[i] = uv;
                tv::array<float, 4> conic_opacity_vec{{cov2d_inv[0], cov2d_inv[1], cov2d_inv[2], opacity_val}};
                op::reinterpret_cast_array_nd<4>($conic_opacity)[i] = conic_opacity_vec;
                int real_tile_touched = 0;
                """)
                def inner_func(code: pccm.FunctionCode):
                    if self._cfg.early_filter_algo != EarlyFilterAlgo.NONE:
                        code.raw(f"""
                        valid_tile &= x >= gaussian_rect_min[0] && x < gaussian_rect_max[0] && y >= gaussian_rect_min[1] && y < gaussian_rect_max[1];
                        real_tile_touched += valid_tile;
                        """)

                self.early_filter_code_context(code_prep, inner_func, is_prep=True)
                if self._cfg.early_filter_algo != EarlyFilterAlgo.NONE:
                    code_prep.raw(f"""
                    $tiles_touched[i] = real_tile_touched;
                    """)
                if not self._cfg.disable_builtin_rgb:
                    color_sh = model.color_sh
                    color_sh_base = model.color_sh_base
                    degree = model.color_sh_degree
                    cur_degree = model.cur_sh_degree
                    has_color_base = color_sh_base is not None
                    code_prep.raw(f"""
                    auto sh_ptr = op::reinterpret_cast_array_nd<3>($color_sh) + gaussian_idx * {(degree + 1) * (degree + 1) - has_color_base};
                    """)
                    if (training):
                        if has_color_base:
                            code_prep.raw(f"""
                            auto sh_base_ptr = op::reinterpret_cast_array_nd<3>($color_sh_base) + gaussian_idx;
                            auto rgb = Gaussian3D::sh_dir_to_rgb<{cur_degree}>((point - cam2world_center).op<op::normalize>(), sh_ptr, sh_base_ptr);
                            """)
                        else:
                            code_prep.raw(f"""
                            auto rgb = Gaussian3D::sh_dir_to_rgb<{cur_degree}>((point - cam2world_center).op<op::normalize>(), sh_ptr);
                            """)
                        code_prep.raw(f"""
                        uint8_t ne_0_packed = (uint8_t(rgb[0] < 0) << 2) | (uint8_t(rgb[1] < 0) << 1) | uint8_t(rgb[2] < 0);
                        op::reinterpret_cast_array_nd<3>($rgb_gaussian)[i] = rgb.op<op::maximum>(0.0f);
                        $sh_to_rgb_ne_0[i] = ne_0_packed;
                        """)
                    else:
                        if has_color_base:
                            code_prep.raw(f"""
                            auto sh_base_ptr = op::reinterpret_cast_array_nd<3>($color_sh_base) + gaussian_idx;
                            op::reinterpret_cast_array_nd<3>($rgb_gaussian)[i] = Gaussian3D::sh_dir_to_rgb<{cur_degree}>((point - cam2world_center).op<op::normalize>(), sh_ptr, sh_base_ptr).op<op::maximum>(0.0f);
                            """)
                        else:
                            code_prep.raw(f"""
                            op::reinterpret_cast_array_nd<3>($rgb_gaussian)[i] = Gaussian3D::sh_dir_to_rgb<{cur_degree}>((point - cam2world_center).op<op::normalize>(), sh_ptr).op<op::maximum>(0.0f);
                            """)
                if (training):
                    code_prep.raw(f"""
                    op::reinterpret_cast_array_nd<3>($cov2d_vecs)[i] = cov2d_vec;
                    """)
                    if not self._cfg.recalc_cov3d_in_bwd:
                        code_prep.raw(f"""
                        op::reinterpret_cast_array_nd<6>($cov3d_vecs)[i] = cov3d_vec;
                        """)

                self._inliner.kernel_1d(prep_kernel_name, batch_size * num, 0, code_prep)
                raise NotImplementedError
            else:
                proxy = model.create_proxy(code_prep, "gaussian_idx", "batch_idx" if batch_size > 1 else "0", batch_size)
                prep_kernel_name += f"_{proxy.get_unique_id()}"
                proxy.prepare_field_proxy()
                proxy.read_field(GaussianCoreFields.XYZ, "point")
                code_prep.raw(f"""
                auto point_cam = cam2world_R_T.op<op::mv_rowmajor>(point - cam2world_center);
                auto uvz = CameraOps::pos_cam_to_uv_no_distort<2, 0, 1>(point_cam, principal_point, focal_xy);
                auto uv = std::get<0>(uvz) - 0.5f;
                auto z_in_cam = std::get<1>(uvz);
                """)
                proxy.read_field(GaussianCoreFields.QUATERNION_XYZW, "quat")
                proxy.read_field(GaussianCoreFields.SCALE, "scale")
                if training:
                    code_prep.raw(f"""
                    auto cov3d_vec = Gaussian3D::scale_quat_to_cov3d(scale, quat);
                    """)
                else:
                    # scale modifier should only be used during inference.
                    code_prep.raw(f"""
                    auto cov3d_vec = Gaussian3D::scale_quat_to_cov3d(scale * $scale_global, quat);
                    """)
                code_prep.raw(f"""
                auto cov2d_vec = Gaussian3D::project_gaussian_to_2d<float, 3>(point_cam, focal_xy, tan_fov, cam2world_R_T, cov3d_vec, $projected_clamp_factor);
                cov2d_vec[0] += {self._cfg.gaussian_lowpass_filter}f;
                cov2d_vec[2] += {self._cfg.gaussian_lowpass_filter}f;
                """)
                if self._cfg.enable_anti_aliasing:
                    code_prep.raw(f"""
                    auto cov2d_inv_and_det = Gaussian3D::gaussian_2d_inverse_and_det_with_comp(cov2d_vec, {lowpass_filter}f, $eps);
                    auto det_div_lowpass_sqrt_clamp = math_op_t::sqrt(math_op_t::max(0.000025f, std::get<2>(cov2d_inv_and_det))); 
                    """)
                else:
                    code_prep.raw(f"""
                    auto cov2d_inv_and_det = Gaussian3D::gaussian_2d_inverse_and_det(cov2d_vec, {lowpass_filter}f, $eps);
                    """)
                proxy.read_field(GaussianCoreFields.OPACITY, "opacity_val")
                if self._cfg.enable_anti_aliasing:
                    code_prep.raw(f"""
                    opacity_val = opacity_val * det_div_lowpass_sqrt_clamp;
                    """)
                code_prep.raw(f"""
                auto cov2d_inv = std::get<0>(cov2d_inv_and_det);
                auto det = std::get<1>(cov2d_inv_and_det);
                auto radii_fp = math_op_t::ceil(Gaussian3D::get_gaussian_2d_ellipse(cov2d_vec, det, 
                    $cov2d_radii_eigen_eps, $gaussian_std_sigma));
                constexpr auto tile_size_xy = tv::array<int, 2>{{{self._cfg.tile_size[0]}, {self._cfg.tile_size[1]}}};
                auto tile_num_xy = tv::div_up(resolution_wh, tile_size_xy);
                auto tile_size_xy_float = tile_size_xy.cast<float>();
                // we don't use real bounding box here to keep it same as origin impl.
                auto gaussian_rect_min = ((uv - radii_fp) / tile_size_xy_float).cast<int>().op<op::clamp>(0, tile_num_xy);
                auto gaussian_rect_max = ((uv + radii_fp + tile_size_xy_float - 1) / tile_size_xy_float).cast<int>().op<op::clamp>(0, tile_num_xy);
                int rect_area = (gaussian_rect_max[0] - gaussian_rect_min[0]) * (gaussian_rect_max[1] - gaussian_rect_min[1]);
                bool is_empty = (det <= 0 || z_in_cam < 0.2f) || rect_area == 0 || invalid || opacity_val <= {self._cfg.alpha_eps}f;

                $tiles_touched[i] = is_empty ? 0 : rect_area;
                $radii[i] = is_empty ? 0 : int(radii_fp);
                if (is_empty){{
                    // calc color sh requires much IO, so we early exit here.
                    // this will cause warp divergence, but save io.
                    return;
                }}
                $depths[i] = {"z_in_cam" if not self._cfg.render_inv_depth else "1.0f / z_in_cam"};

                op::reinterpret_cast_array_nd<2>($uvs)[i] = uv;
                """)
                code_prep.raw(f"""
                tv::array<float, 4> conic_opacity_vec{{cov2d_inv[0], cov2d_inv[1], cov2d_inv[2], opacity_val}};
                op::reinterpret_cast_array_nd<4>($conic_opacity)[i] = conic_opacity_vec;
                auto normed_dir = (point - cam2world_center).op<op::normalize>();
                int real_tile_touched = 0;
                """)
                def inner_func(code: pccm.FunctionCode):
                    if self._cfg.early_filter_algo != EarlyFilterAlgo.NONE:
                        code.raw(f"""
                        // keep results same between our result and coarse 3-sigma filter.
                        valid_tile &= x >= gaussian_rect_min[0] && x < gaussian_rect_max[0] && y >= gaussian_rect_min[1] && y < gaussian_rect_max[1];
                        real_tile_touched += valid_tile;
                        """)

                self.early_filter_code_context(code_prep, inner_func, is_prep=True)
                if self._cfg.early_filter_algo != EarlyFilterAlgo.NONE:
                    code_prep.raw(f"""
                    $tiles_touched[i] = real_tile_touched;
                    """)

                if not self._cfg.disable_builtin_rgb:
                    proxy.read_field(GaussianCoreFields.RGB, "rgb", "normed_dir")
                    if (training):
                        code_prep.raw(f"""
                        uint8_t ne_0_packed = (uint8_t(rgb[0] < 0) << 2) | (uint8_t(rgb[1] < 0) << 1) | uint8_t(rgb[2] < 0);
                        op::reinterpret_cast_array_nd<3>($rgb_gaussian)[i] = rgb.op<op::maximum>(0.0f);
                        $sh_to_rgb_ne_0[i] = ne_0_packed;
                        """)
                    else:
                        code_prep.raw(f"""
                        op::reinterpret_cast_array_nd<3>($rgb_gaussian)[i] = rgb.op<op::maximum>(0.0f);
                        """)
                if training:
                    code_prep.raw(f"""
                    op::reinterpret_cast_array_nd<3>($cov2d_vecs)[i] = cov2d_vec;
                    """)
                    if not self._cfg.recalc_cov3d_in_bwd:
                        code_prep.raw(f"""
                        op::reinterpret_cast_array_nd<6>($cov3d_vecs)[i] = cov3d_vec;
                        """)

                add_vars = {f"{k}_ptr": v for k, v in model.get_proxy_field_dict().items()}
                if prep_user_inputs is not None:
                    add_vars.update(prep_user_inputs)
                    proxy.validate_prep_inputs(prep_user_inputs)
                self._inliner.kernel_1d(prep_kernel_name, batch_size * num, 0, code_prep, 
                    additional_vars=add_vars)

            # with measure_and_print_torch("1", enable=True):
            # if batch_size == 2:
            #     print(torch.linalg.norm(rgb_gaussian.float().reshape(-1, num,3)[0] - rgb_gaussian.float().reshape(-1, num, 3)[1]))
            # print(INLINER.get_nvrtc_module(prep_kernel_name).params.debug_code)
        with measure_and_print_torch("cumsum", enable=enable_verbose):
            # print("tiles_touched mean", tiles_touched[tiles_touched > 0].float().mean(), tiles_touched.max(), tiles_touched[tiles_touched > 64].shape[0])
            # print(tiles_touched)
            # print(tiles_touched[tiles_touched >= 500], torch.nonzero(tiles_touched >= 500))
            # breakpoint()
            tiles_touched.cumsum_(0)
            num_rendered = int(tiles_touched[-1].item())
        print("num_rendered", num_rendered)
        # raise NotImplementedError


        out_img_shape = [
            batch_size, self._cfg.num_channels, height, width
        ] if self._cfg.use_nchw else [batch_size, height, width, self._cfg.num_channels]
        out_custom_feat_shape = [
            batch_size, num_custom_feat, height, width
        ] if self._cfg.use_nchw else [batch_size, height, width, num_custom_feat]
        if (num_rendered == 0):
            final_T = torch.empty([batch_size, height, width],
                                  dtype=torch.float32,
                                  device=xyz.device)
            n_contrib = torch.empty([batch_size, height, width],
                                    dtype=torch.int32,
                                    device=xyz.device)
            out_color = torch.empty(out_img_shape,
                                    dtype=torch.float32,
                                    device=xyz.device)
            out_custom_feat = torch.empty(out_custom_feat_shape,
                                          dtype=torch.float32,
                                          device=xyz.device)
            final_depth = None
            if self._cfg.render_depth:
                final_depth = torch.zeros([batch_size, height, width],
                                          dtype=torch.float32,
                                          device=xyz.device)
            out_empty = GaussianSplatOutput(custom_features=None,
                                            T=final_T,
                                            color=out_color,
                                            depth=final_depth,
                                            n_contrib=n_contrib)
            return out_empty, None
        keys_tile_idx_depth = torch.empty(
            num_rendered,
            dtype=torch.int64 if not enable_32bit_sort else torch.int32,
            device=xyz.device)
        gaussian_idx = torch.empty(num_rendered,
                                   dtype=tiles_touched.dtype,
                                   device=xyz.device)
        shift_bits = 32 if not enable_32bit_sort else 18

        with measure_and_print_torch(f"prepare sort {num_rendered}",
                                     enable=enable_verbose):
            code = pccm.code()
            code.raw(f"""
            namespace op = tv::arrayops;
            using math_op_t = tv::arrayops::MathScalarOp<float>;
            int batch_idx = i / $num;
            auto radii_val = $radii[i];
            auto radii_fp = float(radii_val);
            constexpr auto tile_size_xy = tv::array<int, 2>{{{self._cfg.tile_size[0]}, {self._cfg.tile_size[1]}}};
            auto tile_num_xy = $tile_num_xy_np;
            auto tile_size_xy_float = tile_size_xy.cast<float>();
            """)
            with code.if_("radii_val > 0"):
                if enable_32bit_sort:
                    code.raw(f"""
                    auto depth_uint = uint32_t($depths[i] / $(self._cfg.depth_32bit_prec)) & 0x3FFFFu;
                    """)
                else:
                    code.raw(f"""
                    auto depth_uint = reinterpret_cast<const TV_METAL_DEVICE uint32_t*>($depths)[i];
                    """)
                code.raw(f"""
                auto offset = i == 0 ? 0 : $tiles_touched[i - 1];
                auto uv = op::reinterpret_cast_array_nd<2>($uvs)[i];
                auto gaussian_rect_min = ((uv - radii_fp) / tile_size_xy_float).cast<int>().op<op::clamp>(0, tile_num_xy);
                auto gaussian_rect_max = ((uv + radii_fp + tile_size_xy_float - 1) / tile_size_xy_float).cast<int>().op<op::clamp>(0, tile_num_xy);
                auto conic_opacity_vec = op::reinterpret_cast_array_nd<4>($conic_opacity)[i];

                """)
                def inner_func(code: pccm.FunctionCode):
                    key_real_dtype = "uint64_t" if not enable_32bit_sort else "uint32_t"
                    if enable_32bit_sort:
                        code.raw(f"""
                        uint32_t key = batch_idx * tile_num_xy[0] * tile_num_xy[1] + y * tile_num_xy[0] + x;
                        """)
                        if IsAppleSiliconMacOs:
                            code.raw(f"""
                            int bitcount = metal::popcount(key);
                            """)
                        else:
                            code.raw(f"""
                            int bitcount = __popc(key);
                            """)
                        if not self._cfg.use_cub_sort:
                            # cub sort use unsigned, torch don't support uint sort
                            # so we only need to reverse the depth bits
                            # when use torch sort.
                            code.raw(f"""
                            if (bitcount == 14){{
                                // inverse order of depth because of sign bit is set
                                depth_uint = 0x3FFFFu - depth_uint;
                            }}
                            """)
                    else:
                        code.raw(f"""
                        uint64_t key = batch_idx * tile_num_xy[0] * tile_num_xy[1] + y * tile_num_xy[0] + x;
                        """)

                    if self._cfg.early_filter_algo != EarlyFilterAlgo.NONE:
                        code.raw(f"""
                        valid_tile &= x >= gaussian_rect_min[0] && 
                                    x < gaussian_rect_max[0] && 
                                    y >= gaussian_rect_min[1] &&
                                    y < gaussian_rect_max[1];

                        if (valid_tile){{
                            key <<= {shift_bits};
                            key |= depth_uint; // assume depth always > 0
                            reinterpret_cast<TV_METAL_DEVICE {key_real_dtype}*>($keys_tile_idx_depth)[offset] = key;
                            $gaussian_idx[offset] = i;
                            ++offset;
                        }}

                        """)
                    else:
                        code.raw(f"""
                        key <<= {shift_bits};
                        key |= depth_uint; // assume depth always > 0
                        reinterpret_cast<TV_METAL_DEVICE {key_real_dtype}*>($keys_tile_idx_depth)[offset] = key;
                        $gaussian_idx[offset] = i;
                        ++offset;
                        """)

                self.early_filter_code_context(code, inner_func, is_prep=False)

            self._inliner.kernel_1d(f"prepare_sort_data_{enable_32bit_sort}",
                                    batch_size * num, 0, code)
        # TODO use radix sort with trunated bits (faster) for cuda
        # TODO use 32bit sort, for 1920x1080 tile, the max tile idx is 8100, 13 bits
        # so we can use 18bit for depth. (19bit if torch support uint32, true in torch >= 2.3 and cuda)
        with measure_and_print_torch("do sort", enable=enable_verbose):
            if self._cfg.use_cub_sort:
                from d3sim.d3sim_thtools.device_sort import DeviceSort
                db = True
                sorted_vals = torch.empty_like(keys_tile_idx_depth)
                gaussian_idx_sorted = torch.empty_like(gaussian_idx)
                end_bit = 32 + DeviceSort.get_higher_msb(
                    batch_size * tile_num_x * tile_num_y)
                if enable_32bit_sort:
                    end_bit = 32
                keys_tile_idx_depth_tv = torch_tensor_to_tv(
                    keys_tile_idx_depth, dtype=tv.uint64 if not enable_32bit_sort else tv.uint32)
                sorted_vals_tv = torch_tensor_to_tv(sorted_vals,
                                                    dtype=tv.uint64 if not enable_32bit_sort else tv.uint32)
                gaussian_idx_tv = torch_tensor_to_tv(gaussian_idx)
                gaussian_idx_sorted_tv = torch_tensor_to_tv(
                    gaussian_idx_sorted)
                workspace_size = DeviceSort.get_sort_workspace_size(
                    keys_tile_idx_depth_tv,
                    sorted_vals_tv,
                    gaussian_idx_tv,
                    gaussian_idx_sorted_tv,
                    0,
                    end_bit,
                    double_buffer=db)
                workspace = torch.empty(workspace_size,
                                        dtype=torch.uint8,
                                        device=xyz.device)
                workspace_tv = torch_tensor_to_tv(workspace)
                key_swap, value_swap = DeviceSort.do_radix_sort_with_bit_range(
                    keys_tile_idx_depth_tv,
                    sorted_vals_tv,
                    gaussian_idx_tv,
                    gaussian_idx_sorted_tv,
                    workspace_tv,
                    0,
                    end_bit,
                    double_buffer=db,
                    stream_int=get_current_stream())
                if key_swap:
                    keys_tile_idx_depth, sorted_vals = sorted_vals, keys_tile_idx_depth
                if value_swap:
                    gaussian_idx, gaussian_idx_sorted = gaussian_idx_sorted, gaussian_idx
            else:
                sorted_vals, indices = torch.sort(keys_tile_idx_depth)
                gaussian_idx_sorted = torch.gather(gaussian_idx, 0, indices)

        with measure_and_print_torch("prepare workrange", enable=enable_verbose):
            workload_ranges = torch.zeros([batch_size * tile_num_x * tile_num_y, 2],
                                          dtype=torch.int32,
                                          device=xyz.device)
            shift_bits = 32 if not enable_32bit_sort else 18
            key_real_dtype = "uint64_t" if not enable_32bit_sort else "uint32_t"

            code = pccm.code()
            code.raw(f"""
            auto key = reinterpret_cast<TV_METAL_DEVICE {key_real_dtype}*>($sorted_vals)[i];
            uint32_t tile_idx = key >> {shift_bits};
            """)
            if self._cfg.enable_device_asserts:
                code.raw(f"""
                assert(tile_idx < $batch_size * $tile_num_x * $tile_num_y);
                """)
            with code.if_("i == 0"):
                code.raw(f"""
                $workload_ranges[tile_idx * 2 + 0] = 0;
                """)
            with code.else_():
                code.raw(f"""
                auto last_key = reinterpret_cast<TV_METAL_DEVICE {key_real_dtype}*>($sorted_vals)[i - 1];
                uint32_t last_tile_idx = last_key >> {shift_bits};
                """)
                if self._cfg.enable_device_asserts:
                    code.raw(f"""
                    assert(last_tile_idx < $batch_size * $tile_num_x * $tile_num_y);
                    """)
                code.raw(f"""
                if (tile_idx != last_tile_idx){{
                    $workload_ranges[last_tile_idx * 2 + 1] = i;
                    $workload_ranges[tile_idx * 2 + 0] = i;
                }}
                """)

            code.raw(f"""
            if (i == $num_rendered - 1){{
                $workload_ranges[tile_idx * 2 + 1] = $num_rendered;
            }}
            """)
            self._inliner.kernel_1d(
                f"prepare_workload_range_{enable_32bit_sort}", num_rendered, 0,
                code)
        # workload_ranges_tv = torch_tensor_to_tv(workload_ranges, to_const=True)
        with measure_and_print_torch("5-prep", enable=enable_verbose):

            final_T = torch.empty(
                [batch_size, height, width],
                dtype=torch.float32,
                device=xyz.device)
            n_contrib = torch.empty([batch_size, height, width],
                                    dtype=torch.int32,
                                    device=xyz.device)
            out_color = torch.empty(out_img_shape,
                                    dtype=torch.float32,
                                    device=xyz.device)
            out_custom_feat = torch.empty(out_custom_feat_shape,
                                          dtype=torch.float32,
                                          device=xyz.device)
            final_depth = None
            if self._cfg.render_depth:
                final_depth = torch.zeros([batch_size, height, width],
                                          dtype=torch.float32,
                                          device=xyz.device)
            output_dc = GaussianSplatOutput(custom_features=out_custom_feat,
                                            T=final_T,
                                            color=out_color,
                                            depth=final_depth,
                                            n_contrib=n_contrib,
                                            radii=radii.view(batch_size, num))
            ctx = GaussianSplatOpContext(
                conic_opacity=conic_opacity,
                radii=radii,
                uvs=uvs,
                rgb_gaussian=rgb_gaussian,
                gaussian_idx_sorted=gaussian_idx_sorted,
                workload_ranges=workload_ranges,
                image_shape_wh=image_shape_wh,
                cov3d_vecs=cov3d_vecs,
                cov2d_vecs=cov2d_vecs,
                sh_to_rgb_ne_0=sh_to_rgb_ne_0,
                depths=depths)
        with measure_and_print_torch("rasterize", enable=enable_verbose):
            self.rasterize_forward_backward(
                model,
                ctx,
                output_dc,
                training=training,
                render_depth=self._cfg.render_depth,
                custom_features=custom_feat)
        if not training:
            ctx = None

        return output_dc, ctx

    def early_filter_code_context(self, code: pccm.FunctionCode, inner_code_func: Callable[[pccm.FunctionCode], Any], is_prep: bool):
        if self._cfg.early_filter_algo != EarlyFilterAlgo.NONE:
            code.raw(f"""
            float bound = 2.0f * math_op_t::log({self._cfg.alpha_eps}f / conic_opacity_vec[3]);
            auto bound_rect = Gaussian3D::get_gaussian_2d_ellipse_bound_aabb(op::slice<0, 3>(conic_opacity_vec), bound);
            auto gaussian_rect_min_bound = ((uv - bound_rect) / tile_size_xy_float).cast<int>().op<op::clamp>(0, tile_num_xy);
            auto gaussian_rect_max_bound = ((uv + bound_rect) / tile_size_xy_float).op<op::ceil>().cast<int>().op<op::clamp>(0, tile_num_xy);
            auto mn = gaussian_rect_max_bound - gaussian_rect_min_bound;

            auto uv_offset = uv - gaussian_rect_min_bound.cast<float>() * tile_size_xy_float;
            auto obb_whsc = Gaussian3D::get_gaussian_2d_ellipse_bound_obb(op::slice<0, 3>(conic_opacity_vec), bound);

            """)
            solve_suffix = "_v2"
            # align xy to zero to avoid float precision problem when pixel value is large.
            if self._cfg.early_filter_algo == EarlyFilterAlgo.OBB_DFVT:
                with code.if_("op::slice<2, 4>(obb_whsc).op<op::abs>().op<op::reduce_min>() >= 0.01f"):
                    code.raw(f"""
                    auto ellipse_bound_corners_and_major = Gaussian3D::get_gaussian_2d_ellipse_bound_rect_corners_and_major_vec(op::slice<0, 3>(conic_opacity_vec), bound);
                    auto ellipse_bound_corners = (std::get<0>(ellipse_bound_corners_and_major) + op::reshape<1, 2>(uv_offset)) / tv::array_nd<float, 1, 2>{{{self._cfg.tile_size[0]}, {self._cfg.tile_size[1]}}};
                    auto ellipse_major_vec = std::get<1>(ellipse_bound_corners_and_major);
                    tv::geometry::OBBCornersGridOverlap<float> obb_corners_grid_overlap(ellipse_bound_corners, ellipse_major_vec, mn);
                    """)
                    with code.for_("int _y = 0; _y < mn[1]; ++_y"):
                        code.raw(f"""
                        auto lr = obb_corners_grid_overlap.inc_step(_y);
                        """)
                        with code.for_("int x = lr[0] + gaussian_rect_min_bound[0]; x <= lr[1] + gaussian_rect_min_bound[0]; ++x"):
                            code.raw(f"""
                            int y = _y + gaussian_rect_min_bound[1];
                            bool valid_tile = true;
                            """)
                            inner_code_func(code)

                    # code_prep.raw(f"auto real_tile_touched_debug = 0;")
                    # with code_prep.for_(
                    #         "int y = gaussian_rect_min_bound[1]; y < gaussian_rect_max_bound[1]; ++y"
                    # ):
                    #     code_prep.raw(f"""
                    #     float pixel_idx_y = (y - gaussian_rect_min_bound[1]) * {self._cfg.tile_size[1]};
                    #     float pixel_idx_y_max = pixel_idx_y + {self._cfg.tile_size[1]};
                    #     """)
                    #     with code_prep.for_(
                    #             "int x = gaussian_rect_min_bound[0]; x < gaussian_rect_max_bound[0]; ++x"
                    #     ):
                    #         code_prep.raw(f"""
                    #         float pixel_idx_x = (x - gaussian_rect_min_bound[0]) * {self._cfg.tile_size[0]};
                    #         float pixel_idx_x_max = pixel_idx_x + {self._cfg.tile_size[0]};
                    #         // bool valid_tile = Gaussian3D::aabb_has_overlap_area(uv_offset - bound_rect, uv_offset + bound_rect, 
                    #         //     {{pixel_idx_x, pixel_idx_y}}, {{pixel_idx_x_max, pixel_idx_y_max}});
                    #         bool valid_tile = Gaussian3D::aabb_overlap_obb(
                    #             {{pixel_idx_x, pixel_idx_y}}, {{pixel_idx_x_max, pixel_idx_y_max}}, 
                    #             obb_whsc, uv_offset);

                    #         real_tile_touched_debug += valid_tile;
                    #         """)

                    # code_prep.raw(f"""
                    # auto ellipse_bound_corners_and_major = Gaussian3D::get_gaussian_2d_ellipse_bound_rect_corners_and_major_vec(cov2d_inv, bound);
                    # auto ellipse_bound_corners = (std::get<0>(ellipse_bound_corners_and_major) + op::reshape<1, 2>(uv_offset)) / 16;
                    # auto ellipse_major_vec = std::get<1>(ellipse_bound_corners_and_major);



                    # tv::geometry::OBBCornersGridOverlap<float> obb_corners_grid_overlap(ellipse_bound_corners, ellipse_major_vec, mn);


                    # auto dfvt_corners_ref = Gaussian3D::prepare_dfvt_corners<1, 1>(cov2d_inv, uv_offset, bound);

                    # auto dfvt_corners_res = Gaussian3D::prepare_dfvt_corners_clipped<16, 16>(cov2d_inv, uv_offset, bound);
                    # auto dfvt_corners = std::get<0>(dfvt_corners_res);
                    # auto dfvt_corners_transposed = dfvt_corners.op<op::transpose>();
                    # auto dfvt_corners_bound_min_x = math_op_t::floor(dfvt_corners_transposed[0].op<op::reduce_min>());
                    # auto dfvt_corners_bound_min_y = math_op_t::floor(dfvt_corners_transposed[1].op<op::reduce_min>());
                    # tv::array<float, 2> dfvt_corners_bound_min{{dfvt_corners_bound_min_x, dfvt_corners_bound_min_y}};
                    # auto dfvt_dirs = std::get<1>(dfvt_corners_res);
                    # int step_left = -1;
                    # int step_right = 1;
                    # bool is_top_point_le_zero = dfvt_corners[0][1] <= 0;
                    # tv::array<float, 2> left_ray_dir = dfvt_dirs[0];
                    # tv::array<float, 2> right_ray_dir = dfvt_dirs[1];
                    # tv::array<float, 2> left_ray_dir_stage2 = right_ray_dir;
                    # tv::array<float, 2> right_ray_dir_stage2 = left_ray_dir;
                    # tv::array<float, 2> ray_origin_left = dfvt_corners[0];
                    # tv::array<float, 2> ray_origin_right = dfvt_corners[1];

                    # tv::array<float, 2> tdelta_left = 1.0f / left_ray_dir.op<op::abs>();
                    # tv::array<float, 2> tdelta_right = 1.0f / right_ray_dir.op<op::abs>();
                    # // tv::array<float, 2> tmax_left = (ray_origin_left.op<op::ceil>().op<op::maximum>(1) - ray_origin_left) / left_ray_dir;
                    # tv::array<float, 2> tmax_left;
                    # tmax_left[0] = (math_op_t::max(math_op_t::ceil(ray_origin_left[0] - dfvt_corners_bound_min[0]), 1.0f) + dfvt_corners_bound_min[0] - 1.0f - ray_origin_left[0]) / left_ray_dir[0];
                    # tmax_left[1] = (math_op_t::max(math_op_t::ceil(ray_origin_left[1] - dfvt_corners_bound_min[1]), 1.0f) + dfvt_corners_bound_min[1] - ray_origin_left[1]) / left_ray_dir[1];
                    # tv::array<float, 2> tmax_right = ((ray_origin_right.op<op::ceil>() - dfvt_corners_bound_min).op<op::maximum>(1) + dfvt_corners_bound_min - ray_origin_right) / right_ray_dir;
                    # int cur_X_left = int(math_op_t::floor(ray_origin_left[0]));
                    # int cur_X_right = int(math_op_t::floor(ray_origin_right[0]));
                    # int bound_X_left = int(math_op_t::floor(dfvt_corners[2][0]));
                    # int bound_Y_left = int(math_op_t::floor(dfvt_corners[2][1]));
                    # int bound_X_right = int(math_op_t::floor(dfvt_corners[3][0]));
                    # int bound_Y_right = int(math_op_t::floor(dfvt_corners[3][1]));
                    # int bound_X_end = int(math_op_t::floor(dfvt_corners[4][0]));
                    # if (i == DEBUG_INDEX){{
                    #     tv::printf2("uv", uv[0], uv[1], bound_rect[0], bound_rect[1]);
                    #     tv::printf2("uv_offset", uv_offset[0], uv_offset[1]);

                    #     tv::printf2("whsc", obb_whsc[0], obb_whsc[1], obb_whsc[2], obb_whsc[3], mn[0], mn[1]);
                    #     tv::printf2("corners_clipped", dfvt_corners[0][0], dfvt_corners[0][1], dfvt_corners[1][0], dfvt_corners[1][1], 
                    #         "|", dfvt_corners[2][0], dfvt_corners[2][1], dfvt_corners[3][0], dfvt_corners[3][1],
                    #         "|", dfvt_corners[4][0], dfvt_corners[4][1]);
                    #     tv::printf2<','>("corners", dfvt_corners_ref[0][0], dfvt_corners_ref[0][1], dfvt_corners_ref[1][0], dfvt_corners_ref[1][1], 
                    #         dfvt_corners_ref[2][0], dfvt_corners_ref[2][1], dfvt_corners_ref[3][0], dfvt_corners_ref[3][1]);

                    #     tv::printf2("ray_dirs", left_ray_dir[0], left_ray_dir[1], right_ray_dir[0], right_ray_dir[1]);
                    #     tv::printf2("ray_origins", ray_origin_left[0], ray_origin_left[1], ray_origin_right[0], ray_origin_right[1]);

                    #     tv::printf2("start_cur", cur_X_left, cur_X_right);
                    #     tv::printf2("bounds", bound_X_left, bound_X_right, bound_X_end);
                    #     tv::printf2("tmax", tmax_left[0], tmax_left[1], tmax_right[0], tmax_right[1]);
                    #     tv::printf2("gaussian_rect_bounds", gaussian_rect_min_bound[0], gaussian_rect_min_bound[1], gaussian_rect_max_bound[0], gaussian_rect_max_bound[1]);
                    #     tv::printf2("t_delta_left", tdelta_left[0], tdelta_left[1]);
                    #     auto eigen_val_vec = Gaussian3D::get_gaussian_2d_ellipse_width_height_vec(cov2d_inv, bound);
                    #     auto major_vec = eigen_val_vec[1];
                    #     auto minor_vec = eigen_val_vec[2];
                    #     tv::printf2("eigen_vals", eigen_val_vec[0][0], eigen_val_vec[0][1]);

                    #     tv::printf2("major_vec", major_vec[0], major_vec[1]);
                    #     tv::printf2("minor_vec", minor_vec[0], minor_vec[1]);
                    #     tv::printf2("dfvt_corners_bound_min", dfvt_corners_bound_min[0], dfvt_corners_bound_min[1]);

                    # }}
                    # for (int y = 0; y < mn[1]; ++y){{
                    #     int left = cur_X_left;
                    #     int right = cur_X_right;
                    #     // left fast voxel traversal
                    #     if (i == DEBUG_INDEX){{
                    #         tv::printf2("dfvt start", y, "|", left, right, step_left, step_right);
                    #     }}

                    #     while (true){{
                    #         if (step_left == -1){{
                    #             if (cur_X_left <= bound_X_left && y == bound_Y_left){{
                    #                 cur_X_left = bound_X_left;
                    #                 step_left = 1;
                    #                 tdelta_left = 1.0f / left_ray_dir_stage2.op<op::abs>();
                    #                 tmax_left = ((dfvt_corners[2].op<op::ceil>() - dfvt_corners_bound_min).op<op::maximum>(1.0f) + dfvt_corners_bound_min - dfvt_corners[2]) / left_ray_dir_stage2;
                    #                 bound_X_left = bound_X_end;
                    #             }}
                    #         }}
                    #         if (step_left == 1){{
                    #             if (cur_X_left >= bound_X_end){{
                    #                 break;
                    #             }}
                    #         }}
                    #         if (i == DEBUG_INDEX){{
                    #             tv::printf2("LEFT =", cur_X_left, tmax_left[0], tmax_left[1]);
                    #         }}

                    #         if (tmax_left[0] < tmax_left[1]){{
                    #             // increase X
                    #             tmax_left[0] = tmax_left[0] + tdelta_left[0];
                    #             cur_X_left += step_left;

                    #             left = std::min(left, cur_X_left);
                    #             if (i == DEBUG_INDEX){{

                    #                 tv::printf2("NEW LEFT =", cur_X_left);
                    #             }}

                    #         }}else{{
                    #             tmax_left[1] = tmax_left[1] + tdelta_left[1];
                    #             break;
                    #         }}
                    #     }}
                    #     // right fast voxel traversal
                    #     while (true){{
                    #         if (step_right == 1){{
                    #             if (cur_X_right >= bound_X_right && y == bound_Y_right){{
                    #                 cur_X_right = bound_X_right;
                    #                 step_right = -1;
                    #                 tdelta_right = 1.0f / right_ray_dir_stage2.op<op::abs>();
                    #                 // tmax_right = ((dfvt_corners[3] - dfvt_corners_bound_min).op<op::ceil>().op<op::maximum>(1) + dfvt_corners_bound_min - dfvt_corners[3]) / right_ray_dir_stage2;
                    #                 tmax_right[0] = (math_op_t::max(math_op_t::ceil(dfvt_corners[3][0] - dfvt_corners_bound_min[0]), 1.0f) + dfvt_corners_bound_min[0] - 1.0f - dfvt_corners[3][0]) / right_ray_dir_stage2[0];
                    #                 tmax_right[1] = (math_op_t::max(math_op_t::ceil(dfvt_corners[3][1] - dfvt_corners_bound_min[1]), 1.0f) + dfvt_corners_bound_min[1] - dfvt_corners[3][1]) / right_ray_dir_stage2[1];

                    #                 bound_X_right = bound_X_end;
                    #             }}
                    #         }}
                    #         if (step_right == -1){{
                    #             if (cur_X_right <= bound_X_end){{
                    #                 break;
                    #             }}
                    #         }}
                    #         if (tmax_right[0] < tmax_right[1]){{
                    #             // increase X
                    #             tmax_right[0] = tmax_right[0] + tdelta_right[0];
                    #             cur_X_right += step_right;
                    #             right = std::max(right, cur_X_right);
                    #         }}else{{
                    #             tmax_right[1] = tmax_right[1] + tdelta_right[1];
                    #             break;
                    #         }}
                    #     }}
                    #     left = std::max(0, left);
                    #     right = std::min(mn[0] - 1, right);
                    #     auto test_res = obb_corners_grid_overlap.inc_step(y);
                    #     // if (test_res[0] != left || test_res[1] != right){{
                    #     //     tv::printf2("DEBUG", i);
                    #     // }}
                    #     left = test_res[0];
                    #     right = test_res[1];
                    #     // tv::printf2_once<' ', 1>("WTF dfvt", left, right);
                    #     if (i == DEBUG_INDEX){{
                    #         tv::printf2("DFVT left,right =", left, right, obb_corners_grid_overlap.cur_X_left_, obb_corners_grid_overlap.cur_X_right_);
                    #     }}
                    #     for (int x = left; x <= right; ++x){{
                    #         bool valid_tile = (x + gaussian_rect_min_bound[0]) >= gaussian_rect_min[0] && 
                    #                         (x + gaussian_rect_min_bound[0]) < gaussian_rect_max[0] && 
                    #                         (y + gaussian_rect_min_bound[1]) >= gaussian_rect_min[1] &&
                    #                         (y + gaussian_rect_min_bound[1]) < gaussian_rect_max[1];
                    #         real_tile_touched += valid_tile;
                    #     }}
                    #     if (i == DEBUG_INDEX){{
                    #         tv::printf2("WTF dfvt cur", y, "|", real_tile_touched, "|");
                    #     }}

                    # }}
                    # // if (i == DEBUG_INDEX){{
                    # //     tv::printf2("real_tile_touched", i, real_tile_touched, "REF", real_tile_touched_debug);
                    # // }}
                    # // if (real_tile_touched_debug != real_tile_touched && real_tile_touched_debug == 5){{
                    # //    tv::printf2("MISMATCH", i, real_tile_touched, real_tile_touched_debug);
                    # // }}

                    # """)
                with code.else_():
                    with code.for_(
                            "int y = gaussian_rect_min_bound[1]; y < gaussian_rect_max_bound[1]; ++y"
                    ):
                        code.raw(f"""
                        float pixel_idx_y = (y - gaussian_rect_min_bound[1]) * {self._cfg.tile_size[1]};
                        float pixel_idx_y_max = pixel_idx_y + {self._cfg.tile_size[1]};
                        """)
                        with code.for_(
                                "int x = gaussian_rect_min_bound[0]; x < gaussian_rect_max_bound[0]; ++x"
                        ):
                            code.raw(f"""
                            float pixel_idx_x = (x - gaussian_rect_min_bound[0]) * {self._cfg.tile_size[0]};
                            float pixel_idx_x_max = pixel_idx_x + {self._cfg.tile_size[0]};
                            bool valid_tile = Gaussian3D::aabb_overlap_obb(
                                {{pixel_idx_x, pixel_idx_y}}, {{pixel_idx_x_max, pixel_idx_y_max}}, 
                                obb_whsc, uv_offset);
                            """)
                            inner_code_func(code)
            else:
                with code.for_(
                        "int y = gaussian_rect_min_bound[1]; y < gaussian_rect_max_bound[1]; ++y"
                ):
                    code.raw(f"""
                    float pixel_idx_y = (y - gaussian_rect_min_bound[1]) * {self._cfg.tile_size[1]};
                    float pixel_idx_y_max = pixel_idx_y + {self._cfg.tile_size[1]};
                    """)
                    if self._cfg.early_filter_algo == EarlyFilterAlgo.ELLIPSE:
                        code.raw(f"""
                        auto int_y0 = Gaussian3D::solve_ellipse_y_constant{solve_suffix}(pixel_idx_y, uv_offset, conic_opacity_vec, bound);
                        auto int_y1 = Gaussian3D::solve_ellipse_y_constant{solve_suffix}(pixel_idx_y_max, uv_offset, conic_opacity_vec, bound);
                        """)
                    with code.for_(
                            "int x = gaussian_rect_min_bound[0]; x < gaussian_rect_max_bound[0]; ++x"
                    ):
                        code.raw(f"""
                        float pixel_idx_x = (x - gaussian_rect_min_bound[0]) * {self._cfg.tile_size[0]};
                        float pixel_idx_x_max = pixel_idx_x + {self._cfg.tile_size[0]};
                        
                        bool valid_tile;
                        """)
                        if self._cfg.early_filter_algo == EarlyFilterAlgo.ELLIPSE:
                            code.raw(f"""
                            auto int_x0 = Gaussian3D::solve_ellipse_x_constant{solve_suffix}(pixel_idx_x, uv_offset, conic_opacity_vec, bound);
                            auto int_x1 = Gaussian3D::solve_ellipse_x_constant{solve_suffix}(pixel_idx_x_max, uv_offset, conic_opacity_vec, bound);
                            // this function can filter more unnecessary gaussians, but it's very slow.
                            valid_tile = Gaussian3D::gs2d_ellipse_aabb_is_overlap_external_inter(int_x0, int_x1, int_y0, int_y1, uv_offset,
                                {{pixel_idx_x, pixel_idx_y}}, {{pixel_idx_x_max, pixel_idx_y_max}}, bound);

                            """)
                        elif self._cfg.early_filter_algo == EarlyFilterAlgo.OBB:
                            code.raw(f"""
                            valid_tile = Gaussian3D::aabb_overlap_obb(
                                {{pixel_idx_x, pixel_idx_y}}, {{pixel_idx_x_max, pixel_idx_y_max}}, 
                                obb_whsc, uv_offset);
                            """)
                        else:
                            code.raw(f"""
                            valid_tile = Gaussian3D::aabb_has_overlap_area(uv_offset - bound_rect, uv_offset + bound_rect, 
                                {{pixel_idx_x, pixel_idx_y}}, {{pixel_idx_x_max, pixel_idx_y_max}});
                            """)
                        inner_code_func(code)
        else:
            if is_prep:
                inner_code_func(code)
            else:
                with code.for_(
                        "int y = gaussian_rect_min[1]; y < gaussian_rect_max[1]; ++y"
                ):
                    code.raw(f"""
                    """)
                    with code.for_(
                            "int x = gaussian_rect_min[0]; x < gaussian_rect_max[0]; ++x"
                    ):
                        code.raw(f"""
                        bool valid_tile = true;
                        """)
                        inner_code_func(code)

    def rasterize_forward_backward(self,
                                   model: GaussianModelOriginBase,
                                   ctx: GaussianSplatOpContext,
                                   out: GaussianSplatOutput,
                                   grad: GaussianSplatGradients | None = None,
                                   training: bool = False,
                                   render_depth: bool = False,
                                   custom_features: torch.Tensor | None = None):
        """if grad is not None, run backward mode, otherwise run forward mode.
        rasterize backward share many code with forward, so we use a single function to handle both.
        """
        num = model.xyz.shape[0]
        is_bwd = grad is not None
        image_shape_wh = ctx.image_shape_wh
        block_size = self._cfg.block_size
        width = image_shape_wh[0]
        height = image_shape_wh[1]

        depths = ctx.depths

        tile_num_x = div_up(width, self._cfg.tile_size[0])
        tile_num_y = div_up(height, self._cfg.tile_size[1])
        workload_ranges = ctx.workload_ranges
        conic_opacity_tv = torch_tensor_to_tv(ctx.conic_opacity, to_const=True)
        uvs_tv = torch_tensor_to_tv(ctx.uvs, to_const=True)
        gaussian_idx_sorted_tv = torch_tensor_to_tv(ctx.gaussian_idx_sorted,
                                                    to_const=True)
        rgb_gaussian_tv = None 
        if ctx.rgb_gaussian is not None:
            rgb_gaussian_tv = torch_tensor_to_tv(ctx.rgb_gaussian, to_const=True)
        final_custom_feat = out.custom_features
        # assume nhwc
        num_custom_feat = 0 if final_custom_feat is None else final_custom_feat.shape[
            1 if self._cfg.use_nchw else -1]
        has_custom_feat = num_custom_feat > 0

        final_T = out.T
        final_color = out.color
        final_n_contrib = out.n_contrib
        batch_size = final_color.shape[0]
        # assert final_T.numel() == width * height
        # assert final_color.numel() == width * height * 3

        assert final_n_contrib is not None
        final_depth = out.depth
        if render_depth:
            assert final_depth is not None

        grad_out: RasterizeGradientOutput | None = None
        color_use_smem = is_bwd  # TODO why bwd use smem for color?
        custom_feat_use_smem = num_custom_feat <= 16
        if grad is not None:
            drgb = grad.drgb
            assert drgb.is_contiguous(), "drgb must be contiguous"
            dT = grad.dT
            ddepth_th = grad.ddepth
            # assert drgb.numel() == width * height * 3
            # assert dT.numel() == width * height
        if is_bwd:
            duv = torch.zeros([batch_size * num, 2],
                              dtype=torch.float32,
                              device=final_T.device)
            dconic = torch.zeros([batch_size * num, 3],
                                 dtype=torch.float32,
                                 device=final_T.device)
            dopacity = torch.zeros([batch_size * num],
                                   dtype=torch.float32,
                                   device=final_T.device)
            dcolor = None 
            if not self._cfg.disable_builtin_rgb:
                dcolor = torch.zeros([batch_size * num, 3],
                                    dtype=torch.float32,
                                    device=final_T.device)
            dcustom_features = None 
            if num_custom_feat > 0:
                dcustom_features = torch.zeros([batch_size * num, num_custom_feat],
                                            dtype=torch.float32,
                                            device=final_T.device)
            dz = None
            if render_depth:
                dz = torch.zeros([batch_size * num],
                                 dtype=torch.float32,
                                 device=final_T.device)
            grad_out = RasterizeGradientOutput(
                duv=duv,
                dconic=dconic,
                dopacity=dopacity,
                dcolor=dcolor,
                dcustom_features=dcustom_features,
                dz=dz)
        bwd_reduce_method = self._cfg.backward_reduction
        t_dtype = "double" if self._cfg.transmittance_is_double else "float"
        use_bwd_reduce = bwd_reduce_method != "none"
        kernel_unique_name = (
            f"rasterize_{is_bwd}_{self._cfg.tile_size}_{training}"
            f"_{num_custom_feat}_{self._cfg.render_depth}_{self._cfg.render_inv_depth}"
            f"_{self._cfg.use_nchw}_{self._cfg.render_rgba}_{batch_size == 1}"
            f"_{final_n_contrib is None}_{self._cfg.backward_reduction}")
        num_channels = self._cfg.num_channels
        render_rgba = self._cfg.render_rgba
        # when use raw kernel in metal, we don't use nonuniform grid to
        # keep logic same as cuda.
        if IsAppleSiliconMacOs:
            launch_param = tv.LaunchParam(
                (tile_num_x, tile_num_y, batch_size),
                (self._cfg.tile_size[0], self._cfg.tile_size[1], 1))
        else:
            launch_param = tv.LaunchParam(
                (tile_num_x, tile_num_y, batch_size),
                (self._cfg.tile_size[0] * self._cfg.tile_size[1], 1, 1))
        atomic_add_count = None 
        if is_bwd and self._cfg.measure_atomic_add_count:
            atomic_add_count = torch.zeros([1], dtype=torch.int32, device=final_T.device)
        if kernel_unique_name in self._code_obj_caches:
            code_rasterize = self._code_obj_caches[kernel_unique_name]
            with measure_and_print_torch(
                    f"BWD-rasterize-{gaussian_idx_sorted_tv.shape[0]}",
                    enable=self._cfg.verbose):
                self._inliner.kernel_raw(kernel_unique_name, launch_param,
                                        code_rasterize)
            if atomic_add_count is not None:
                print(f"atomic add count: {atomic_add_count.item()}")

            return grad_out

        code_rasterize = pccm.code()
        code_rasterize.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<float>;
        """)
        if IsAppleSiliconMacOs:
            code_rasterize.raw(f"""
            tv::array<uint32_t, 2> pixel_idx_xy{{threadPositionInGrid.x, threadPositionInGrid.y}};
            tv::array<uint32_t, 2> tile_idx_xy{{threadgroupPositionInGrid.x, threadgroupPositionInGrid.y}};
            // keep in mind that metal only have 32KB shared memory
            threadgroup int num_done_shared[32];
            uint thread_rank = threadPositionInThreadgroup.y * {self._cfg.tile_size[0]} + threadPositionInThreadgroup.x;
            """)
            if batch_size != 1:
                code_rasterize.raw(f"""
                int batch_idx = threadgroupPositionInGrid.z;
                """)
            else:
                code_rasterize.raw(f"""
                constexpr int batch_idx = 0;
                """)
        else:
            # for 16x16 tile, the warp shape is 16x2, we need to convert to 8x4
            # slightly faster when use warp reduce.
            lane_matrix_shape = self._cfg.warp_size
            num_lane_matrix_shape = (self._cfg.tile_size[0] // lane_matrix_shape[0], self._cfg.tile_size[1] // lane_matrix_shape[1])
            assert self._cfg.tile_size[0] % lane_matrix_shape[0] == 0
            assert self._cfg.tile_size[1] % lane_matrix_shape[1] == 0
            code_rasterize.raw(f"""
            int warp_idx_tmp = threadIdx.x / 32;
            int warp_x = warp_idx_tmp % {num_lane_matrix_shape[0]};
            int warp_y = warp_idx_tmp / {num_lane_matrix_shape[0]};
            int lane_idx_tmp = threadIdx.x % 32;
            int lane_x = lane_idx_tmp % {lane_matrix_shape[0]};
            int lane_y = lane_idx_tmp / {lane_matrix_shape[0]};

            tv::array<uint32_t, 2> pixel_idx_xy{{blockIdx.x * {self._cfg.tile_size[0]} + warp_x * {lane_matrix_shape[0]} + lane_x, 
                                                 blockIdx.y * {self._cfg.tile_size[1]} + warp_y * {lane_matrix_shape[1]} + lane_y}};
            tv::array<uint32_t, 2> tile_idx_xy{{blockIdx.x, blockIdx.y}};
            uint32_t thread_rank = threadIdx.x;
            """)

            # code_rasterize.raw(f"""
            # tv::array<uint32_t, 2> pixel_idx_xy{{blockIdx.x * {self._cfg.tile_size[0]} + threadIdx.x, blockIdx.y * {self._cfg.tile_size[1]} + threadIdx.y}};
            # tv::array<uint32_t, 2> tile_idx_xy{{blockIdx.x, blockIdx.y}};
            # uint32_t thread_rank = threadIdx.y * {self._cfg.tile_size[0]} + threadIdx.x;
            # """)
            if batch_size != 1:
                code_rasterize.raw(f"""
                int batch_idx = blockIdx.z;
                """)
            else:
                code_rasterize.raw(f"""
                constexpr int batch_idx = 0;
                """)
        code_rasterize.raw(f"""
        TV_SHARED_MEMORY int collected_id[{block_size}];
        TV_SHARED_MEMORY float2 collected_xy[{block_size}];
        TV_SHARED_MEMORY float4 collected_conic_opacity[{block_size}];
        """)
        if color_use_smem and not self._cfg.disable_builtin_rgb:
            code_rasterize.raw(f"""
            TV_SHARED_MEMORY float collected_rgb_gaussian[{block_size * 3}];
            """)
        if render_depth:
            code_rasterize.raw(f"""
            TV_SHARED_MEMORY float collected_depth[{block_size}];
            """)
        if custom_feat_use_smem and has_custom_feat:
            code_rasterize.raw(f"""
            TV_SHARED_MEMORY tv::array<float, {num_custom_feat}> collected_custom_feat[{block_size}];
            """)
        if bwd_reduce_method == "block":
            code_rasterize.raw(f"""
            TV_SHARED_MEMORY tv::array<float, {block_size // 32}> shared_dL_drgb_0;
            TV_SHARED_MEMORY tv::array<float, {block_size // 32}> shared_dL_drgb_1;
            TV_SHARED_MEMORY tv::array<float, {block_size // 32}> shared_dL_drgb_2;
            TV_SHARED_MEMORY tv::array<float, {block_size // 32}> shared_dL_dopacity;
            TV_SHARED_MEMORY tv::array<float, {block_size // 32}> shared_dL_duv_0;
            TV_SHARED_MEMORY tv::array<float, {block_size // 32}> shared_dL_duv_1;
            TV_SHARED_MEMORY tv::array<float, {block_size // 32}> shared_dL_dconic_0;
            TV_SHARED_MEMORY tv::array<float, {block_size // 32}> shared_dL_dconic_1;
            TV_SHARED_MEMORY tv::array<float, {block_size // 32}> shared_dL_dconic_2;
            // TV_SHARED_MEMORY float shared_dL_dopacity[{block_size // 32}];
            TV_SHARED_MEMORY tv::array<int, {block_size // 32}> shared_num_contributor;
            """)

        code_rasterize.raw(f"""
        bool inside = pixel_idx_xy[0] < $width && pixel_idx_xy[1] < $height;
        uint32_t pixel_id = ({"" if batch_size == 1 else "batch_idx * ($width * $height) + "}
            $width * pixel_idx_xy[1] + pixel_idx_xy[0]);
        auto pixel_idx_xy_fp = pixel_idx_xy.cast<float>();

        // Check if this thread is associated with a valid pixel or outside.
        // Done threads can help with fetching, but don't rasterize
        bool done = !inside;
        {t_dtype} T = 1.0f;
        """)
        if batch_size == 1:
            code_rasterize.raw(f"""
            auto range = op::reinterpret_cast_array_nd<2>($workload_ranges)[tile_idx_xy[1] * $tile_num_x + tile_idx_xy[0]];
            uint32_t pixel_id_no_batch = pixel_id;
            """)
        else:
            code_rasterize.raw(f"""
            uint32_t pixel_id_no_batch = $width * pixel_idx_xy[1] + pixel_idx_xy[0];
            auto range = op::reinterpret_cast_array_nd<2>($workload_ranges)[batch_idx * $tile_num_x * $tile_num_y + tile_idx_xy[1] * $tile_num_x + tile_idx_xy[0]];
            """)
        if not is_bwd:
            code_rasterize.raw(f"""
            tv::array<float, {num_channels}> rgb{{}};
            int toDo = range[1] - range[0];
            int rounds = tv::div_up(range[1] - range[0], {block_size});
            uint32_t contributor = 0;
            """)
            if has_custom_feat:
                code_rasterize.raw(f"""
                tv::array<float, {num_custom_feat}> custom_feat_accum{{}};
                """)
        else:
            code_rasterize.raw(f"""
            int warp_idx = thread_rank / 32;
            int lane_idx = tv::parallel::lane_index();

            int contributor = inside ? $final_n_contrib[pixel_id] + 1 : 0;
            """)
            if use_bwd_reduce:
                code_rasterize.raw(f"""
                int max_num_contributor = tv::parallel::warp_max(contributor);
                max_num_contributor = tv::parallel::warp_broadcast(max_num_contributor, 0);
                """)
                if bwd_reduce_method == "block":
                    code_rasterize.raw(f"""
                    if (lane_idx == 0){{
                        shared_num_contributor[warp_idx] = max_num_contributor;
                    }}
                    tv::parallel::block_sync_shared_io();
                    max_num_contributor = shared_num_contributor.op<op::reduce_max>();
                    """)
            else:
                code_rasterize.raw(f"""
                int max_num_contributor = contributor;
                """)
            code_rasterize.raw(f"""
            int toDo = max_num_contributor;
            int rounds = tv::div_up(range[1] - range[0], {block_size});
            """)
            if self._cfg.enable_device_asserts:
                code_rasterize.raw(f"""
                assert(max_num_contributor >= 0 && max_num_contributor <= range[1] - range[0]);
                """)

        if render_depth:
            code_rasterize.raw(f"""
            float depth_accum = 0.0f;
            """)
        if is_bwd:
            code_rasterize.raw(f"""
            decltype(T) render_T = inside ? $final_T[pixel_id] : 0.0f;
            auto dT_val = inside ? $dT[pixel_id] : 0.0f;
            """)
            if self._cfg.use_nchw:
                # disable_builtin_rgb not implemented here
                # because we don't support this to reduce code size.
                # check is done in forward pass.
                if batch_size == 1:
                    code_rasterize.raw(f"""
                    tv::array<float, {num_channels}> render_rgb, drgb_array;
                    render_rgb[0] = inside ? $final_color[0 * $height * $width + pixel_id] : 0.0f;
                    render_rgb[1] = inside ? $final_color[1 * $height * $width + pixel_id] : 0.0f;
                    render_rgb[2] = inside ? $final_color[2 * $height * $width + pixel_id] : 0.0f;

                    drgb_array[0] = inside ? $drgb[0 * $height * $width + pixel_id] : 0.0f;
                    drgb_array[1] = inside ? $drgb[1 * $height * $width + pixel_id] : 0.0f;
                    drgb_array[2] = inside ? $drgb[2 * $height * $width + pixel_id] : 0.0f;
                    """)
                    if render_rgba:
                        code_rasterize.raw(f"""
                        render_rgb[3] = inside ? $final_color[3 * $height * $width + pixel_id] : 0.0f;
                        drgb_array[3] = inside ? $drgb[3 * $height * $width + pixel_id] : 0.0f;
                        """)
                else:
                    code_rasterize.raw(f"""
                    tv::array<float, {num_channels}> render_rgb, drgb_array;
                    render_rgb[0] = inside ? $final_color[(batch_idx * {num_channels} + 0) * $height * $width + pixel_id_no_batch] : 0.0f;
                    render_rgb[1] = inside ? $final_color[(batch_idx * {num_channels} + 1) * $height * $width + pixel_id_no_batch] : 0.0f;
                    render_rgb[2] = inside ? $final_color[(batch_idx * {num_channels} + 2) * $height * $width + pixel_id_no_batch] : 0.0f;

                    drgb_array[0] = inside ? $drgb[(batch_idx * {num_channels} + 0) * $height * $width + pixel_id_no_batch] : 0.0f;
                    drgb_array[1] = inside ? $drgb[(batch_idx * {num_channels} + 1) * $height * $width + pixel_id_no_batch] : 0.0f;
                    drgb_array[2] = inside ? $drgb[(batch_idx * {num_channels} + 2) * $height * $width + pixel_id_no_batch] : 0.0f;
                    """)
                    if render_rgba:
                        code_rasterize.raw(f"""
                        render_rgb[3] = inside ? $final_color[(batch_idx * {num_channels} + 3) * $height * $width + pixel_id_no_batch] : 0.0f;
                        drgb_array[3] = inside ? $drgb[(batch_idx * {num_channels} + 3) * $height * $width + pixel_id_no_batch] : 0.0f;
                        """)
            else:
                # support disable-builtin-rgb natively.
                if num_channels > 0:
                    code_rasterize.raw(f"""
                    tv::array<float, {num_channels}> render_rgb = inside ? op::reinterpret_cast_array_nd<{num_channels}>($final_color)[pixel_id] : tv::array<float, {num_channels}>{{}};
                    auto drgb_array = inside ? op::reinterpret_cast_array_nd<{num_channels}>($drgb)[pixel_id] : tv::array<float, {num_channels}>{{}};
                    """)
                if has_custom_feat:
                    code_rasterize.raw(f"""
                    auto render_custom_feat = inside ? op::reinterpret_cast_array_nd<{num_custom_feat}>($final_custom_feat)[pixel_id] : tv::array<float, {num_custom_feat}>{{}};
                    auto dcustom_feat = inside ? op::reinterpret_cast_array_nd<{num_custom_feat}>($dcustom_features)[pixel_id] : tv::array<float, {num_custom_feat}>{{}};
                    """)

            if render_depth:
                code_rasterize.raw(f"""
                auto render_depth = inside ? $final_depth[pixel_id] : 0.0f;
                auto ddepth = inside ? $ddepth_th[pixel_id] : 0.0f;
                """)
        with code_rasterize.for_(
                f"int i = 0; i < rounds; ++i, toDo -= {block_size}"):
            if IsAppleSiliconMacOs:
                code_rasterize.raw(f"""
                int num_done_simd = metal::simd_sum(int(done));
                num_done_shared[tv::parallel::warp_index()] = num_done_simd;
                tv::parallel::block_sync_shared_io();
                int num_done = 0;
                for (uint32_t j = 0; j < ({block_size}u / tv::parallel::warp_size()); ++j){{
                    num_done += num_done_shared[j];
                }}
                """)
            else:
                code_rasterize.raw(f"""
                int num_done = __syncthreads_count(done);
                if (num_done == {block_size}){{
                    break;
                }}
                """)
            code_rasterize.raw(f"""
            int progress = i * {block_size} + thread_rank;
            """)
            with code_rasterize.if_("range[0] + progress < range[1]"):

                code_rasterize.raw("""
                int gaussian_id = $gaussian_idx_sorted_tv[range[0] + progress];
                collected_id[thread_rank] = gaussian_id;
                collected_xy[thread_rank] = reinterpret_cast<TV_METAL_DEVICE const float2*>($uvs_tv)[gaussian_id];
                collected_conic_opacity[thread_rank] = reinterpret_cast<TV_METAL_DEVICE const float4*>($conic_opacity_tv)[gaussian_id];
                """)
                if self._cfg.enable_device_asserts:
                    code_rasterize.raw(f"""
                    assert(gaussian_id >= 0 && gaussian_id < $num);
                    """)
                if color_use_smem and not self._cfg.disable_builtin_rgb:
                    code_rasterize.raw(f"""
                    auto rgb_gaussian_val = op::reinterpret_cast_array_nd<3>($rgb_gaussian_tv)[gaussian_id];
                    collected_rgb_gaussian[thread_rank * 3 + 0] = rgb_gaussian_val[0];
                    collected_rgb_gaussian[thread_rank * 3 + 1] = rgb_gaussian_val[1];
                    collected_rgb_gaussian[thread_rank * 3 + 2] = rgb_gaussian_val[2];
                    """)
                if has_custom_feat and custom_feat_use_smem:
                    code_rasterize.raw(f"""
                    auto custom_feat_val = op::reinterpret_cast_array_nd<{num_custom_feat}>($custom_features)[gaussian_id];
                    collected_custom_feat[thread_rank] = custom_feat_val;
                    """)
                if render_depth:
                    code_rasterize.raw(f"""
                    collected_depth[thread_rank] = $depths[gaussian_id];
                    """)
            code_rasterize.raw(f"""
            tv::parallel::block_sync_shared_io();
            """)
            # if not is_bwd:
            for_stmt = f"int j = 0; !done && j < std::min({block_size}, toDo); ++j"
            if is_bwd:
                for_stmt = f"int j = 0; j < std::min({block_size}, toDo); ++j"

            # else:
            #     for_stmt = f"int j = 0; j < contributer; ++j"

            with code_rasterize.for_(for_stmt):
                # if not is_bwd:
                #     code_rasterize.raw(f"""
                #     ++contributor;
                #     """)
                code_rasterize.raw(f"""
                auto uv = collected_xy[j];
                auto conic_opacity_vec = collected_conic_opacity[j];
                tv::array<float, 2> dist{{uv.x - pixel_idx_xy_fp[0], uv.y - pixel_idx_xy_fp[1]}};
                float power = (-0.5f * (conic_opacity_vec.x * dist[0] * dist[0] + 
                                       conic_opacity_vec.z * dist[1] * dist[1]) - 
                                        conic_opacity_vec.y * dist[0] * dist[1]);
                float G = math_op_t::fast_exp(power); // used in backward
                float alpha = math_op_t::min(0.99f, conic_opacity_vec.w * G);
                // alpha >= eps => conic_opacity_vec.w * G >= eps
                // when G is maximum (1), then conic_opacity_vec.w >= eps
                // we can filter this in prep.
                """)
                if is_bwd and use_bwd_reduce:
                    code_rasterize.raw(f"""
                    float next_T = T * (1.0f - alpha);
                    bool valid = power <= 0.0f && alpha >= {self._cfg.alpha_eps}f && (i * {block_size} + j < contributor); 
                    """)
                    if bwd_reduce_method == "warp":
                        code_rasterize.raw(f"""
                        if (!tv::parallel::warp_any(valid)){{
                            continue;
                        }}
                        """)
                    elif bwd_reduce_method == "block":
                        # don't support block reduce for metal
                        # block reduce currently greatly slower
                        # than warp reduce, so we don't need to support it.
                        if not IsAppleSiliconMacOs:
                            code_rasterize.raw(f"""
                            // avoid empty atomic write
                            if (!__syncthreads_or(valid)){{
                                continue;
                            }}
                            """)
                else:
                    if is_bwd:
                        code_rasterize.raw(f"""
                        if (alpha < {self._cfg.alpha_eps}f || power > 0.0f){{
                            continue;
                        }}
                        {t_dtype} next_T = T * {t_dtype}(1.0f - alpha);
                        """)

                    else:
                        code_rasterize.raw(f"""
                        if (alpha < {self._cfg.alpha_eps}f){{
                            continue;
                        }}
                        {t_dtype} next_T = T * {t_dtype}(1.0f - alpha);
                        """)
                if not is_bwd:
                    code_rasterize.raw(f"""
                    bool should_done = next_T < {self._cfg.transmittance_eps}f;
                    if (power > 0.0f || should_done){{
                        done = should_done;
                        continue;
                    }}
                    """)
                ifctx = nullcontext()
                if is_bwd and use_bwd_reduce:
                    if not self._cfg.disable_builtin_rgb:
                        code_rasterize.raw(f"""
                        tv::array<float, 3> dL_drgb{{}};
                        """)
                    code_rasterize.raw(f"""
                    tv::array<float, 2> dL_duv{{}};
                    tv::array<float, 3> dL_dcov2d{{}};
                    float dL_dopacity = 0.0f;
                    """)
                    if has_custom_feat:
                        code_rasterize.raw(f"""
                        tv::array<float, {num_custom_feat}> dL_dcustom_feat{{}};
                        """)
                    if render_depth:
                        code_rasterize.raw(f"""
                        float dL_dz = 0.0f;
                        """)
                    ifctx = code_rasterize.if_("valid")
                with ifctx:
                    code_rasterize.raw(f"""
                    float weight = alpha * T;
                    T = next_T;
                    """)
                    if not is_bwd:
                        code_rasterize.raw(f"""
                        contributor = i * {block_size} + j;
                        """)
                    gaussian_id_inited = False
                    if not self._cfg.disable_builtin_rgb:
                        if color_use_smem:
                            code_rasterize.raw(f"""
                            auto rgb_val = op::reinterpret_cast_array_nd<3>(collected_rgb_gaussian)[j];
                            """)
                        else:
                            gaussian_id_inited = True
                            code_rasterize.raw(f"""
                            auto gaussian_id = collected_id[j];
                            auto rgb_val = op::reinterpret_cast_array_nd<3>($rgb_gaussian_tv)[gaussian_id];
                            """)
                            if self._cfg.enable_device_asserts:
                                code_rasterize.raw(f"""
                                assert(gaussian_id >= 0 && gaussian_id < $num);
                                """)
                    if has_custom_feat:
                        if custom_feat_use_smem:
                            code_rasterize.raw(f"""
                            auto custom_feat_val = collected_custom_feat[j];
                            """)
                        else:
                            if not gaussian_id_inited:
                                code_rasterize.raw(f"""
                                auto gaussian_id = collected_id[j];
                                """)
                                gaussian_id_inited = True
                            code_rasterize.raw(f"""
                            auto custom_feat_val = op::reinterpret_cast_array_nd<{num_custom_feat}>($custom_features)[gaussian_id];
                            """)
                    if is_bwd:
                        if not self._cfg.disable_builtin_rgb:

                            if render_rgba:
                                code_rasterize.raw(f"""
                                render_rgb[0] -= weight * rgb_val[0];
                                render_rgb[1] -= weight * rgb_val[1];
                                render_rgb[2] -= weight * rgb_val[2];
                                render_rgb[3] -= weight;
                                """)
                            else:
                                code_rasterize.raw(f"""
                                render_rgb -= weight * rgb_val;
                                """)
                        else:
                            if render_rgba:
                                code_rasterize.raw(f"""
                                render_rgb -= weight;
                                """)
                        if render_depth:
                            code_rasterize.raw(f"""
                            auto depth_val = collected_depth[j];
                            render_depth -= weight * depth_val;
                            """)
                        if has_custom_feat:
                            code_rasterize.raw(f"""
                            render_custom_feat -= weight * custom_feat_val;
                            """)

                    if not is_bwd:
                        if not self._cfg.disable_builtin_rgb:
                            if render_rgba:
                                code_rasterize.raw(f"""
                                rgb[0] += weight * rgb_val[0];
                                rgb[1] += weight * rgb_val[1];
                                rgb[2] += weight * rgb_val[2];
                                rgb[3] += weight;
                                """)
                            else:
                                code_rasterize.raw(f"""
                                rgb += weight * rgb_val;
                                """)
                        else:
                            if render_rgba:
                                code_rasterize.raw(f"""
                                rgb += weight;
                                """)
                        if render_depth:
                            code_rasterize.raw(f"""
                            auto depth_val = collected_depth[j];
                            depth_accum += weight * depth_val;
                            """)
                        if has_custom_feat:
                            code_rasterize.raw(f"""
                            custom_feat_accum += weight * custom_feat_val;
                            """)
                    if is_bwd:
                        if not self._cfg.disable_builtin_rgb:
                            if render_rgba:
                                code_rasterize.raw(f"""
                                float dL_dalpha_without_div = (drgb_array.op<op::dot>(T * op::concat(rgb_val, tv::array<float, 1>{{1.0f}}) - render_rgb));
                                // grad from T, we don't apply background when training, apply it in torch side.
                                dL_dalpha_without_div += -dT_val * render_T;
                                """)
                            else:
                                code_rasterize.raw(f"""
                                float dL_dalpha_without_div = (drgb_array.op<op::dot>(T * rgb_val - render_rgb));
                                dL_dalpha_without_div += -dT_val * render_T;
                                """)
                        else:
                            if render_rgba:
                                code_rasterize.raw(f"""
                                float dL_dalpha_without_div = (drgb_array.op<op::dot>(T - render_rgb));
                                dL_dalpha_without_div += -dT_val * render_T;
                                """)
                            else:
                                code_rasterize.raw(f"""
                                float dL_dalpha_without_div = -dT_val * render_T;
                                """)

                        if render_depth:
                            code_rasterize.raw(f"""
                            dL_dalpha_without_div += ddepth * (T * depth_val - render_depth);
                            """)
                        if has_custom_feat:
                            code_rasterize.raw(f"""
                            dL_dalpha_without_div += (dcustom_feat.op<op::dot>(T * custom_feat_val - render_custom_feat));
                            """)
                        code_rasterize.raw(f"""
                        auto dL_dalpha = dL_dalpha_without_div / (1.0f - alpha);
                        
                        """)
                        if not self._cfg.disable_builtin_rgb:
                            code_rasterize.raw(f"""
                            {"" if use_bwd_reduce else "auto"} dL_drgb = weight * {"op::slice<0, 3>(drgb_array)" if render_rgba else "drgb_array"};
                            """)
                        if render_depth:
                            # WARNING this is actually dL/dz_inv if render_inv_depth is True.
                            code_rasterize.raw(f"""
                            {"" if use_bwd_reduce else "float"} dL_dz = weight * ddepth;
                            """)
                        if has_custom_feat:
                            code_rasterize.raw(f"""
                            {"" if use_bwd_reduce else f"tv::array<float, {num_custom_feat}>"} dL_dcustom_feat = weight * dcustom_feat;
                            """)
                        if self._cfg.move_opacity_in_grad_to_prep:
                            code_rasterize.raw(f"""
                            const float dL_dG = dL_dalpha;
                            const float gdx = G * dist[0];
                            const float gdy = G * dist[1];
                            const float dG_du = -gdx * conic_opacity_vec.x - gdy * conic_opacity_vec.y;
                            const float dG_dv = -gdy * conic_opacity_vec.z - gdx * conic_opacity_vec.y;
                            {"" if use_bwd_reduce else "tv::array<float, 2>"} dL_duv =  {{
                                dL_dG * dG_du,
                                dL_dG * dG_dv
                            }};
                            {"" if use_bwd_reduce else "tv::array<float, 3>"} dL_dcov2d = {{
                                gdx * dist[0] * dL_dG,
                                gdx * dist[1] * dL_dG,
                                gdy * dist[1] * dL_dG,
                            }};
                            {"" if use_bwd_reduce else "float"} dL_dopacity = dL_dalpha * G;
                            """)

                        else:
                            code_rasterize.raw(f"""

                            const float dL_dG = conic_opacity_vec.w * dL_dalpha;
                            const float gdx = G * dist[0];
                            const float gdy = G * dist[1];
                            // in original code, the proj matrix map point to ndc,
                            // so their gradient contains 0.5 * W/H.
                            // we don't do this, so that grad is removed.
                            const float dG_du = -gdx * conic_opacity_vec.x - gdy * conic_opacity_vec.y;
                            const float dG_dv = -gdy * conic_opacity_vec.z - gdx * conic_opacity_vec.y;
                            {"" if use_bwd_reduce else "tv::array<float, 2>"} dL_duv =  {{
                                dL_dG * dG_du,
                                dL_dG * dG_dv
                            }};
                            // in origin 3dgs code, this is -0.5f * gdx * d.y * dL_dG
                            // this is actually sym grad, means dL_dcov2d is 2x2,
                            // grad[0][1] == grad[1][0] == -0.5f * gdx * d.y * dL_dG
                            // this make the input of grad of inverse (cov2d) is actually 2x2
                            // here we remove the 0.5 and modify cov2d inverse grad code,
                            // remove the 2 multipler in cov2d inverse grad code.
                            {"" if use_bwd_reduce else "tv::array<float, 3>"} dL_dcov2d = {{
                                -0.5f * gdx * dist[0] * dL_dG,
                                -gdx * dist[1] * dL_dG,
                                -0.5f * gdy * dist[1] * dL_dG,
                            }};
                            {"" if use_bwd_reduce else "float"} dL_dopacity = dL_dalpha * G;
                            """)
                if is_bwd:
                    reduce_if_ctx = nullcontext()
                    if bwd_reduce_method == "warp":
                        reduce_if_ctx = code_rasterize.if_("lane_idx == 0")
                    elif bwd_reduce_method == "block":
                        reduce_if_ctx = code_rasterize.if_("thread_rank == 0")
                    if use_bwd_reduce:
                        if not self._cfg.disable_builtin_rgb:
                            code_rasterize.raw(f"""
                            dL_drgb[0] = tv::parallel::warp_sum(dL_drgb[0]);
                            dL_drgb[1] = tv::parallel::warp_sum(dL_drgb[1]);
                            dL_drgb[2] = tv::parallel::warp_sum(dL_drgb[2]);
                            """)
                        code_rasterize.raw(f"""
                        dL_duv[0] = tv::parallel::warp_sum(dL_duv[0]);
                        dL_duv[1] = tv::parallel::warp_sum(dL_duv[1]);

                        dL_dcov2d[0] = tv::parallel::warp_sum(dL_dcov2d[0]);
                        dL_dcov2d[1] = tv::parallel::warp_sum(dL_dcov2d[1]);
                        dL_dcov2d[2] = tv::parallel::warp_sum(dL_dcov2d[2]);

                        dL_dopacity = tv::parallel::warp_sum(dL_dopacity);
                        """)
                        if has_custom_feat:
                            code_rasterize.raw(f"""
                            dL_dcustom_feat = op::apply(tv::parallel::warp_sum<float>, dL_dcustom_feat);
                            """)
                        if render_depth:
                            code_rasterize.raw(f"""
                            dL_dz = tv::parallel::warp_sum(dL_dz);
                            """)
                        if self._cfg.backward_reduction == "block":
                            code_rasterize.raw(f"""
                            tv::parallel::block_sync_shared_io();

                            if (lane_idx == 0){{
                                shared_dL_drgb_0[warp_idx] = dL_drgb[0];
                                shared_dL_drgb_1[warp_idx] = dL_drgb[1];
                                shared_dL_drgb_2[warp_idx] = dL_drgb[2];
                                shared_dL_duv_0[warp_idx] = dL_duv[0];
                                shared_dL_duv_1[warp_idx] = dL_duv[1];
                                shared_dL_dconic_0[warp_idx] = dL_dcov2d[0];
                                shared_dL_dconic_1[warp_idx] = dL_dcov2d[1];
                                shared_dL_dconic_2[warp_idx] = dL_dcov2d[2];
                                shared_dL_dopacity[warp_idx] = dL_dopacity;
                            }}
                            tv::parallel::block_sync_shared_io();
                            """)
                    with reduce_if_ctx:
                        if not gaussian_id_inited:
                            code_rasterize.raw(f"""
                            auto gaussian_id = collected_id[j];
                            """)
                            gaussian_id_inited = True
                            if self._cfg.enable_device_asserts:
                                code_rasterize.raw(f"""
                                assert(gaussian_id >= 0 && gaussian_id < $num);
                                """)
                        if self._cfg.backward_reduction == "block":
                            code_rasterize.raw(f"""
                            dL_drgb[0] = shared_dL_drgb_0.op<op::sum>();
                            dL_drgb[1] = shared_dL_drgb_1.op<op::sum>();
                            dL_drgb[2] = shared_dL_drgb_2.op<op::sum>();
                            dL_duv[0] = shared_dL_duv_0.op<op::sum>();
                            dL_duv[1] = shared_dL_duv_1.op<op::sum>();
                            dL_dcov2d[0] = shared_dL_dconic_0.op<op::sum>();
                            dL_dcov2d[1] = shared_dL_dconic_1.op<op::sum>();
                            dL_dcov2d[2] = shared_dL_dconic_2.op<op::sum>();
                            dL_dopacity = shared_dL_dopacity.op<op::sum>();
                            """)
                            if render_depth:
                                raise NotImplementedError("not implemented yet")
                        if not self._cfg.disable_builtin_rgb:
                            code_rasterize.raw(f"""
                            tv::parallel::atomicAdd($dcolor + gaussian_id * 3 + 0, dL_drgb[0]);
                            tv::parallel::atomicAdd($dcolor + gaussian_id * 3 + 1, dL_drgb[1]);
                            tv::parallel::atomicAdd($dcolor + gaussian_id * 3 + 2, dL_drgb[2]);
                            """)
                        code_rasterize.raw(f"""
                        tv::parallel::atomicAdd($duv + gaussian_id * 2 + 0, dL_duv[0]);
                        tv::parallel::atomicAdd($duv + gaussian_id * 2 + 1, dL_duv[1]);

                        tv::parallel::atomicAdd($dconic + gaussian_id * 3 + 0, dL_dcov2d[0]);
                        tv::parallel::atomicAdd($dconic + gaussian_id * 3 + 1, dL_dcov2d[1]);
                        tv::parallel::atomicAdd($dconic + gaussian_id * 3 + 2, dL_dcov2d[2]);
                        tv::parallel::atomicAdd($dopacity + gaussian_id, dL_dopacity);
                        """)
                        if render_depth:
                            code_rasterize.raw(f"""
                            tv::parallel::atomicAdd($dz + gaussian_id, dL_dz);
                            """)
                        if has_custom_feat:
                            for j in range(num_custom_feat):
                                code_rasterize.raw(f"""
                                tv::parallel::atomicAdd($dcustom_features + gaussian_id * {num_custom_feat} + {j}, dL_dcustom_feat[{j}]);
                                """)
                        if self._cfg.measure_atomic_add_count:
                            code_rasterize.raw(f"""
                            tv::parallel::atomicAdd($atomic_add_count, 1);
                            """)
                    # if has_custom_feat:
                    #     code_rasterize.raw(f"""
                    #     for (int channel_idx = 0; channel_idx < {num_custom_feat}; ++channel_idx){{
                    #         custom_feature[channel_idx] += weight * $custom_feat[gaussian_id * {num_custom_feat} + channel_idx];
                    #     }}
                    #     """)
        if not is_bwd:
            if not training:
                if not self._cfg.disable_builtin_rgb:
                    if render_rgba:
                        code_rasterize.raw(f"""
                        auto bg_color_val_rgb = $(self._cfg.bg_color);
                        auto bg_color_val = op::concat(bg_color_val_rgb, tv::array<float, 1>{{0.0f}});
                        """)
                    else:
                        code_rasterize.raw(f"""
                        auto bg_color_val = $(self._cfg.bg_color);
                        """)
                else:
                    if render_rgba:
                        code_rasterize.raw(f"""
                        auto bg_color_val = tv::array<float, 1>{{0.0f}};
                        """)
            with code_rasterize.if_("inside"):
                code_rasterize.raw(f"""
                $final_T[pixel_id] = T;
                $final_n_contrib[pixel_id] = contributor;
                """)
                if render_depth:
                    if not training:
                        # TODO should we apply (1 - T) here?
                        code_rasterize.raw(f"""
                        $final_depth[pixel_id] = depth_accum / (1e-10f + 1.0f - T);
                        """)
                    else:
                        code_rasterize.raw(f"""
                        $final_depth[pixel_id] = depth_accum;
                        """)
                # use fused background color on inference.
                # TODO how to support envmap?
                if not self._cfg.use_nchw:
                    if num_channels > 0:
                        if not training:
                            code_rasterize.raw(f"""
                            auto rgb_before_clamp = rgb + T * bg_color_val;
                            rgb_before_clamp = rgb_before_clamp.op<op::clamp>(0.0f, 1.0f);
                            op::reinterpret_cast_array_nd<{num_channels}>($final_color)[pixel_id] = rgb_before_clamp;
                            """)
                        else:
                            code_rasterize.raw(f"""
                            op::reinterpret_cast_array_nd<{num_channels}>($final_color)[pixel_id] = rgb;
                            """)
                    if has_custom_feat:
                        code_rasterize.raw(f"""
                        op::reinterpret_cast_array_nd<{num_custom_feat}>($final_custom_feat)[pixel_id] = custom_feat_accum;
                        """)
                else:
                    if not training:
                        code_rasterize.raw(f"""
                        auto rgb_before_clamp = rgb + T * bg_color_val;
                        rgb_before_clamp = rgb_before_clamp.op<op::clamp>(0.0f, 1.0f);
                        $final_color[(batch_idx * {num_channels} + 0) * $height * $width + pixel_id_no_batch] = rgb_before_clamp[0];
                        $final_color[(batch_idx * {num_channels} + 1) * $height * $width + pixel_id_no_batch] = rgb_before_clamp[1];
                        $final_color[(batch_idx * {num_channels} + 2) * $height * $width + pixel_id_no_batch] = rgb_before_clamp[2];
                        """)
                        if render_rgba:
                            code_rasterize.raw(f"""
                            $final_color[(batch_idx * {num_channels} + 3) * $height * $width + pixel_id_no_batch] = rgb_before_clamp[3];
                            """)
                    else:
                        code_rasterize.raw(f"""
                        $final_color[(batch_idx * {num_channels} + 0) * $height * $width + pixel_id_no_batch] = rgb[0];
                        $final_color[(batch_idx * {num_channels} + 1) * $height * $width + pixel_id_no_batch] = rgb[1];
                        $final_color[(batch_idx * {num_channels} + 2) * $height * $width + pixel_id_no_batch] = rgb[2];
                        """)
                        if render_rgba:
                            code_rasterize.raw(f"""
                            $final_color[(batch_idx * {num_channels} + 3) * $height * $width + pixel_id_no_batch] = rgb[3];
                            """)
        # if num_custom_feat:
        #     code_rasterize.raw(f"""
        #     if (inside)
        #     {{
        #         for (int channel_idx = 0; channel_idx < {num_custom_feat}; ++channel_idx){{
        #             $out_custom_feat[channel_idx * $height * $width + pixel_id] = custom_feature[channel_idx];
        #         }}
        #     }}
        #     """)
        self._code_obj_caches[kernel_unique_name] = code_rasterize
        with measure_and_print_torch(
                f"BWD-rasterize-{gaussian_idx_sorted_tv.shape[0]}",
                enable=self._cfg.verbose):
            self._inliner.kernel_raw(kernel_unique_name, launch_param,
                                     code_rasterize)
        if atomic_add_count is not None:
            print(f"atomic add count: {atomic_add_count.item()}")
        # if is_bwd:
        # print(self._inliner.get_nvrtc_module(kernel_unique_name).params.debug_code)

        #     print(INLINER.get_nvrtc_kernel_attrs(kernel_unique_name))
        return grad_out

    def backward(self,
                 model: GaussianModelOriginBase,
                 cameras: BasicPinholeCamera | CameraBundle,
                 ctx: GaussianSplatOpContext,
                 out: GaussianSplatOutput,
                 grad: GaussianSplatGradients,
                 return_uv_grad: bool = False,
                 instance2world_T: torch.Tensor | None = None,
                 custom_features: torch.Tensor | None = None,
                 prep_user_inputs: dict[str, Any] | None = None,
                ):
        # if isinstance(cameras, CameraBundle):
        #     raise NotImplementedError("CameraBundle is not supported yet.")
        num = model.xyz.shape[0]
        camera = cameras
        image_shape_wh = camera.image_shape_wh

        width = image_shape_wh[0]
        height = image_shape_wh[1]

        tile_num_x = div_up(width, self._cfg.tile_size[0])
        tile_num_y = div_up(height, self._cfg.tile_size[1])
        radii = ctx.radii
        grad_out = self.rasterize_forward_backward(model,
                                                   ctx,
                                                   out,
                                                   grad,
                                                   training=True,
                                                   custom_features=custom_features)
        assert grad_out is not None
        duv = grad_out.duv
        dconic = grad_out.dconic
        dopacity = grad_out.dopacity
        dcolor = grad_out.dcolor
        dz = grad_out.dz
        # ddepth = grad_out.ddepth
        resolution_wh_np = np.array([width, height], np.int32)
        is_cam_bundle = False
        batch_size = 1
        enable_instance_render = False
        if isinstance(camera, BasicPinholeCamera):
            intrinsic = camera.intrinsic
            cam2world = camera.pose.to_world
            focal_xy_np = np.array([intrinsic[0, 0], intrinsic[1, 1]], np.float32)
            principal_point_np = np.array([intrinsic[0, 2], intrinsic[1, 2]],
                                        np.float32)
            cam2world_T_np = np.ascontiguousarray(cam2world[:3].T)
            cam2world_R_np = cam2world_T_np[:3]
            cam2world_center_np = cam2world_T_np[3]
            if instance2world_T is not None:
                assert camera.frame_local_id is not None 
                assert model.instance_id is not None, "instance_id must be provided in model when render with instance"
                enable_instance_render = True
                frame_local_id = camera.frame_local_id

        else:
            is_cam_bundle = True
            if instance2world_T is not None:
                assert camera.frame_ids is not None, "frame_ids must be provided when render with instance"
                assert model.instance_id is not None, "instance_id must be provided in model when render with instance"
                enable_instance_render = True
                assert instance2world_T.ndim == 4, "instance2world_T must be 4D tensor [num_frame, num_instance, 4, 3]"
                # instance2world_T: [num_frame, num_instance, 4, 3]
                num_instance = instance2world_T.shape[1]
            frame_ids = camera.frame_ids
            cam2world_T_ten = camera.cam2world_T
            focal_xy_ppoint_ten = camera.focal_xy_ppoint
            batch_size = cam2world_T_ten.shape[0]

        conic_opacity = ctx.conic_opacity
        cov3d_vecs = ctx.cov3d_vecs
        cov2d_vecs = ctx.cov2d_vecs
        sh_to_rgb_ne_0 = ctx.sh_to_rgb_ne_0
        if not self._cfg.recalc_cov3d_in_bwd:
            assert cov3d_vecs is not None
        assert cov2d_vecs is not None
        assert sh_to_rgb_ne_0 is not None
        # grad_model = GaussianModelOrigin(xyz=dxyz_res,
        #                                  color_sh=dcolor_sh_res,
        #                                  scale=dscale_res,
        #                                  quaternion_xyzw=dquat_res,
        #                                  opacity=dopacity_res,
        #                                  color_sh_base=dcolor_base_sh_res)
        grad_model = model.zeros_like(model, not_param=True, external_tensors={"opacity": dopacity} if batch_size == 1 else None)
        # grad_model = model.zeros_like_debug(model, dopacity=dopacity if batch_size == 1 else None)

        instance_ids = model.instance_id
        prep_kernel_name = (
            f"gs3d_preprocess_bwd_{self._cfg.tile_size}_{model.get_unique_kernel_key()}_"
            f"_{dz is None}_{batch_size == 1}_"
            f"{is_cam_bundle}")
        for_ctx = nullcontext()
        with tv.measure_and_print("BWD-prep", enable=self._cfg.verbose):
            code_prep = pccm.code()
            code_prep.raw(f"""
            namespace op = tv::arrayops;
            using math_op_t = tv::arrayops::MathScalarOp<float>;
            auto resolution_wh = $resolution_wh_np;
            """)
            if not self._cfg.use_proxy_model:
                xyz = model.xyz
                scales = model.scale
                quat_xyzw = model.quaternion_xyzw
                color_sh = model.color_sh
                opacity = model.opacity

                degree = model.color_sh_degree
                cur_degree = model.cur_sh_degree
                fused_scale_act = model.fused_scale_act_op
                fused_q_act = model.fused_quaternion_xyzw_act_op
                fused_opacity_act = model.fused_opacity_act_op

                dxyz_res = grad_model.xyz
                dscale_res = grad_model.scale
                dquat_res = grad_model.quaternion_xyzw
                dcolor_sh_res = grad_model.color_sh
                dcolor_base_sh_res = None
                has_color_base = model.color_sh_base is not None
                if model.color_sh_base is not None:
                    dcolor_base_sh_res = grad_model.color_sh_base
                dopacity_res = grad_model.opacity

                if batch_size > 1:
                    code_prep.raw(f"""
                    float dopacity_val_acc = 0.0f;
                    tv::array<float, 3> dxyz_acc{{}};
                    tv::array<float, 4> dquat_acc{{}};
                    tv::array<float, 3> dscale_acc{{}};
                    """)
                    for_ctx = code_prep.for_("int batch_idx = 0; batch_idx < $batch_size; ++batch_idx")
                with for_ctx:
                    if batch_size > 1:
                        code_prep.raw(f"""
                        auto i_with_batch = i + batch_idx * $num;
                        """)
                    else:
                        code_prep.raw(f"""
                        auto i_with_batch = i;
                        """)
                    code_prep.raw(f"""
                    if ($radii[i_with_batch] == 0){{
                        {"return" if batch_size == 1 else "continue"};
                    }}
                    auto uv_grad = op::reinterpret_cast_array_nd<2>($duv)[i_with_batch];
                    auto conic_grad = op::reinterpret_cast_array_nd<3>($dconic)[i_with_batch];
                    // auto cov3d_vec = op::reinterpret_cast_array_nd<6>($cov3d_vecs)[i_with_batch];
                    """)
                    if self._cfg.move_opacity_in_grad_to_prep:
                        code_prep.raw(f"""
                        auto _conic_w_tmp = $conic_opacity[i_with_batch * 4 + 3];
                        uv_grad *= _conic_w_tmp;
                        op::reinterpret_cast_array_nd<2>($duv)[i_with_batch] = uv_grad;
                        conic_grad[0] *= -0.5f * _conic_w_tmp;
                        conic_grad[1] *= -_conic_w_tmp;
                        conic_grad[2] *= -0.5f * _conic_w_tmp;
                        """)
                    if is_cam_bundle:
                        # cam2world, focal and ppoint are tensor, load from gpu mem
                        code_prep.raw(f"""
                        auto cam2world_T = op::reinterpret_cast_array_nd<4, 3>($cam2world_T_ten)[batch_idx];
                        auto focal_xy_ppoint = op::reinterpret_cast_array_nd<4>($focal_xy_ppoint_ten)[batch_idx];
                        auto focal_xy = op::slice<0, 2>(focal_xy_ppoint);
                        auto principal_point = op::slice<2, 4>(focal_xy_ppoint);
                        auto cam2world_R_T = op::slice<0, 3>(cam2world_T);
                        auto cam2world_center = cam2world_T[3];

                        """)
                        if enable_instance_render:
                            code_prep.raw(f"""
                            auto frame_id = $frame_ids[batch_idx];
                            """)
                    else:
                        # cam2world, focal and ppoint are registers
                        code_prep.raw(f"""
                        auto cam2world_R_T = $cam2world_R_np;
                        auto cam2world_center = $cam2world_center_np;
                        auto focal_xy = $focal_xy_np;
                        auto principal_point = $principal_point_np;
                        """)
                        if enable_instance_render:
                            code_prep.raw(f"""
                            auto frame_id = $frame_local_id;
                            """)
                    if enable_instance_render:
                        code_prep.raw(f"""
                        auto instance_id = $instance_ids[i];
                        if (instance_id >= 0){{
                            auto instance2world_T_val = op::reinterpret_cast_array_nd<4, 3>($instance2world_T)[frame_id * $num_instance + instance_id];
                            // get cam2instance_T, cam2instance = world2instance @ cam2world
                            auto cam2instance_T = instance2world_T_val.op<op::transform_matrix_colmajor_inverse>().op<op::transform_matrix_mm_nnn>(cam2world_T);
                            cam2world_R_T = op::slice<0, 3>(cam2instance_T);
                            cam2world_center = cam2instance_T[3];
                        }}
                        """)
                    code_prep.raw(f"""

                    auto tan_fov = 0.5f * resolution_wh.cast<float>() / focal_xy;
                    auto point = op::reinterpret_cast_array_nd<3>($xyz)[i];
                    point = point - cam2world_center;

                    auto point_cam = cam2world_R_T.op<op::mv_rowmajor>(point);
                    auto cov2d_vec = op::reinterpret_cast_array_nd<3>($cov2d_vecs)[i_with_batch];
                    """)
                    if self._cfg.render_inv_depth:
                        code_prep.raw(f"""
                        auto dpoint_cam = CameraOps::pos_cam_to_uv_no_distort_grad(uv_grad, {"-$dz[i_with_batch] / (point_cam[2] * point_cam[2])" if dz is not None else "0.0f"}, point_cam, principal_point, focal_xy);
                        """)
                    else:
                        code_prep.raw(f"""
                        auto dpoint_cam = CameraOps::pos_cam_to_uv_no_distort_grad(uv_grad, {"$dz[i_with_batch]" if dz is not None else "0.0f"}, point_cam, principal_point, focal_xy);
                        """)

                    if self._cfg.enable_anti_aliasing:
                        code_prep.raw(f"""
                        auto cov2d_inv_and_det = Gaussian3D::gaussian_2d_inverse_and_det_with_comp(cov2d_vec, {self._cfg.gaussian_lowpass_filter}f);
                        
                        auto det_div_lowpass = std::get<2>(cov2d_inv_and_det);
                        auto det_div_lowpass_sqrt_clamp = math_op_t::sqrt(math_op_t::max(0.000025f, det_div_lowpass)); 
                        tv::array<float, 1> opacity_val{{$opacity[i]}};
                        auto opacity_val_with_act = opacity_val.op<op::{fused_opacity_act[0]}>();
                        tv::array<float, 1> dopacity_val{{$dopacity[i_with_batch]}};
                        auto ddet_div_det_lowpass_clamp = dopacity_val[0] * opacity_val_with_act[0];

                        auto ddet_div_det_lowpass = det_div_lowpass <= 0.000025f ? 0.0f : ddet_div_det_lowpass_clamp * 0.5f / det_div_lowpass_sqrt_clamp;

                        auto dcov2d = Gaussian3D::gaussian_2d_inverse_and_det_grad_with_comp(conic_grad, ddet_div_det_lowpass, 
                            cov2d_vec, {self._cfg.gaussian_lowpass_filter}f);
                        dopacity_val = (dopacity_val * det_div_lowpass_sqrt_clamp).op<op::{fused_opacity_act[1]}>(opacity_val_with_act, opacity_val);
                        """)
                        if batch_size > 1:
                            code_prep.raw(f"""
                            dopacity_val_acc += dopacity_val[0];
                            """)
                        else:
                            code_prep.raw(f"""
                            $dopacity[i] = dopacity_val[0];
                            """)
                    else:
                        code_prep.raw(f"""
                        auto dcov2d = Gaussian3D::gaussian_2d_inverse_and_det_grad(conic_grad, cov2d_vec, {self._cfg.gaussian_lowpass_filter}f);
                        """)

                    code_prep.raw(f"""
                    auto scale = op::reinterpret_cast_array_nd<3>($scales)[i];
                    auto quat = op::reinterpret_cast_array_nd<4>($quat_xyzw)[i];
                    auto scale_act = scale.op<op::{fused_scale_act[0]}>();
                    auto quat_act = quat.op<op::{fused_q_act[0]}>();

                    """)
                    if self._cfg.recalc_cov3d_in_bwd:
                        code_prep.raw(f"""
                        auto cov3d_vec = Gaussian3D::scale_quat_to_cov3d(scale_act, quat_act);
                        """)
                    else:
                        code_prep.raw(f"""
                        auto cov3d_vec = op::reinterpret_cast_array_nd<6>($cov3d_vecs)[i_with_batch];
                        """)
                    code_prep.raw(f"""
                    auto proj_grad_res = Gaussian3D::project_gaussian_to_2d_grad<float, 3>(dcov2d, point_cam, focal_xy, tan_fov, cam2world_R_T, cov3d_vec);
                    dpoint_cam += std::get<0>(proj_grad_res);

                    auto dxyz = cam2world_R_T.op<op::mv_colmajor>(dpoint_cam);
                    auto cov3d_grad_res = Gaussian3D::scale_quat_to_cov3d_grad(std::get<1>(proj_grad_res), scale_act, quat_act);
                    auto dscale = std::get<0>(cov3d_grad_res);
                    auto dquat = std::get<1>(cov3d_grad_res);
                    dscale = dscale.op<op::{fused_scale_act[1]}>(scale_act, scale);
                    dquat = dquat.op<op::{fused_q_act[1]}>(quat_act, quat);
                    """)
                    if not self._cfg.enable_anti_aliasing:
                        code_prep.raw(f"""
                        tv::array<float, 1> opacity_val{{$opacity[i]}};
                        tv::array<float, 1> dopacity_val{{$dopacity[i_with_batch]}};
                        dopacity_val = dopacity_val.op<op::{fused_opacity_act[1]}>(opacity_val.op<op::{fused_opacity_act[0]}>(), opacity_val);
                        """)
                        if batch_size > 1:
                            code_prep.raw(f"""
                            dopacity_val_acc += dopacity_val[0];
                            """)
                        else:
                            code_prep.raw(f"""
                            $dopacity[i] = dopacity_val[0];
                            """)
                    if batch_size > 1:
                        code_prep.raw(f"""
                        dscale_acc += dscale;
                        dquat_acc += dquat;
                        """)
                    else:
                        code_prep.raw(f"""
                        op::reinterpret_cast_array_nd<3>($dscale_res)[i] = dscale;
                        op::reinterpret_cast_array_nd<4>($dquat_res)[i] = dquat;
                        """)
                    code_prep.raw(f"""

                    auto normed_dir = (point).op<op::normalize>();
                    auto sh_ptr = op::reinterpret_cast_array_nd<3>($color_sh) + i * {(degree + 1) * (degree + 1) - has_color_base};
                    auto dsh_ptr = op::reinterpret_cast_array_nd<3>($dcolor_sh_res) + i * {(degree + 1) * (degree + 1) - has_color_base};
                    auto color_grad = op::reinterpret_cast_array_nd<3>($dcolor)[i_with_batch];

                    uint8_t nn_packed = $sh_to_rgb_ne_0[i_with_batch];
                    bool rgb0_ne_0 = nn_packed & 0x1;
                    bool rgb1_ne_0 = nn_packed & 0x2;
                    bool rgb2_ne_0 = nn_packed & 0x4;
                    color_grad[0] *= rgb0_ne_0 ? 0.0f : 1.0f;
                    color_grad[1] *= rgb1_ne_0 ? 0.0f : 1.0f;
                    color_grad[2] *= rgb2_ne_0 ? 0.0f : 1.0f;
                    """)
                    sh_grad_fn = "sh_dir_to_rgb_grad" if batch_size == 1 else "sh_dir_to_rgb_grad_batch"
                    if has_color_base:
                        code_prep.raw(f"""
                        auto dsh_base_ptr = op::reinterpret_cast_array_nd<3>($dcolor_base_sh_res) + i;
                        auto dnormed_dir = Gaussian3D::{sh_grad_fn}<{cur_degree}>(color_grad, dsh_ptr,
                            normed_dir, sh_ptr, dsh_base_ptr);
                        """)
                    else:
                        code_prep.raw(f"""
                        auto dnormed_dir = Gaussian3D::{sh_grad_fn}<{cur_degree}>(color_grad, dsh_ptr,
                            normed_dir, sh_ptr);
                        """)
                    code_prep.raw(f"""
                    dxyz += dnormed_dir.op<op::normalize_grad>(point);
                    """)
                    if batch_size == 1:
                        code_prep.raw(f"""
                        op::reinterpret_cast_array_nd<3>($dxyz_res)[i] = dxyz;
                        """)
                    else:
                        code_prep.raw(f"""
                        dxyz_acc += dxyz;
                        """)
                if batch_size > 1:
                    code_prep.raw(f"""
                    op::reinterpret_cast_array_nd<3>($dxyz_res)[i] = dxyz_acc;
                    op::reinterpret_cast_array_nd<4>($dquat_res)[i] = dquat_acc;
                    op::reinterpret_cast_array_nd<3>($dscale_res)[i] = dscale_acc;
                    $dopacity_res[i] = dopacity_val_acc;
                    """)
                self._inliner.kernel_1d(prep_kernel_name, num, 0, code_prep)
            else:
                proxy = model.create_proxy(code_prep, "i", "batch_idx" if batch_size > 1 else "0", batch_size, is_bwd=True)

                proxy.prepare_field_proxy()
                if batch_size > 1:
                    # code_prep.raw(f"""
                    # float dopacity_val_acc = 0.0f;
                    # tv::array<float, 3> dxyz_acc{{}};
                    # tv::array<float, 4> dquat_acc{{}};
                    # tv::array<float, 3> dscale_acc{{}};
                    # """)
                    for_ctx = code_prep.for_("int batch_idx = 0; batch_idx < $batch_size; ++batch_idx")
                with for_ctx:
                    if batch_size > 1:
                        code_prep.raw(f"""
                        auto i_with_batch = i + batch_idx * $num;
                        """)
                    else:
                        code_prep.raw(f"""
                        auto i_with_batch = i;
                        """)
                    code_prep.raw(f"""
                    if ($radii[i_with_batch] == 0){{
                        {"return" if batch_size == 1 else "continue"};
                    }}
                    auto uv_grad = op::reinterpret_cast_array_nd<2>($duv)[i_with_batch];
                    auto conic_grad = op::reinterpret_cast_array_nd<3>($dconic)[i_with_batch];
                    // auto cov3d_vec = op::reinterpret_cast_array_nd<6>($cov3d_vecs)[i_with_batch];
                    """)
                    if self._cfg.move_opacity_in_grad_to_prep:
                        code_prep.raw(f"""
                        auto _conic_w_tmp = $conic_opacity[i_with_batch * 4 + 3];
                        uv_grad *= _conic_w_tmp;
                        op::reinterpret_cast_array_nd<2>($duv)[i_with_batch] = uv_grad;
                        conic_grad[0] *= -0.5f * _conic_w_tmp;
                        conic_grad[1] *= -_conic_w_tmp;
                        conic_grad[2] *= -0.5f * _conic_w_tmp;
                        """)
                    if is_cam_bundle:
                        # cam2world, focal and ppoint are tensor, load from gpu mem
                        code_prep.raw(f"""
                        auto cam2world_T = op::reinterpret_cast_array_nd<4, 3>($cam2world_T_ten)[batch_idx];
                        auto focal_xy_ppoint = op::reinterpret_cast_array_nd<4>($focal_xy_ppoint_ten)[batch_idx];
                        auto focal_xy = op::slice<0, 2>(focal_xy_ppoint);
                        auto principal_point = op::slice<2, 4>(focal_xy_ppoint);
                        auto cam2world_R_T = op::slice<0, 3>(cam2world_T);
                        auto cam2world_center = cam2world_T[3];

                        """)
                        if enable_instance_render:
                            code_prep.raw(f"""
                            auto frame_id = $frame_ids[batch_idx];
                            """)
                    else:
                        # cam2world, focal and ppoint are registers
                        code_prep.raw(f"""
                        auto cam2world_R_T = $cam2world_R_np;
                        auto cam2world_center = $cam2world_center_np;
                        auto focal_xy = $focal_xy_np;
                        auto principal_point = $principal_point_np;
                        """)
                        if enable_instance_render:
                            code_prep.raw(f"""
                            auto frame_id = $frame_local_id;
                            """)
                    if enable_instance_render:
                        code_prep.raw(f"""
                        auto instance_id = $instance_ids[i];
                        if (instance_id >= 0){{
                            auto instance2world_T_val = op::reinterpret_cast_array_nd<4, 3>($instance2world_T)[frame_id * $num_instance + instance_id];
                            // get cam2instance_T, cam2instance = world2instance @ cam2world
                            auto cam2instance_T = instance2world_T_val.op<op::transform_matrix_colmajor_inverse>().op<op::transform_matrix_mm_nnn>(cam2world_T);
                            cam2world_R_T = op::slice<0, 3>(cam2instance_T);
                            cam2world_center = cam2instance_T[3];
                        }}
                        """)
                    code_prep.raw(f"""

                    auto tan_fov = 0.5f * resolution_wh.cast<float>() / focal_xy;
                    """)
                    proxy.read_field(GaussianCoreFields.XYZ, "point")
                    code_prep.raw(f"""
                    point = point - cam2world_center;

                    auto point_cam = cam2world_R_T.op<op::mv_rowmajor>(point);
                    auto cov2d_vec = op::reinterpret_cast_array_nd<3>($cov2d_vecs)[i_with_batch];
                    """)
                    if self._cfg.render_inv_depth:
                        code_prep.raw(f"""
                        auto dpoint_cam = CameraOps::pos_cam_to_uv_no_distort_grad(uv_grad, 
                            {"-$dz[i_with_batch] / (point_cam[2] * point_cam[2])" if dz is not None else "0.0f"}, 
                            point_cam, principal_point, focal_xy);
                        """)
                    else:
                        code_prep.raw(f"""
                        auto dpoint_cam = CameraOps::pos_cam_to_uv_no_distort_grad(uv_grad, 
                            {"$dz[i_with_batch]" if dz is not None else "0.0f"}, 
                            point_cam, principal_point, focal_xy);
                        """)
                    if self._cfg.enable_anti_aliasing:
                        proxy.read_field(GaussianCoreFields.OPACITY, "opacity")

                        code_prep.raw(f"""
                        auto cov2d_inv_and_det = Gaussian3D::gaussian_2d_inverse_and_det_with_comp(cov2d_vec, {self._cfg.gaussian_lowpass_filter}f);
                        
                        auto det_div_lowpass = std::get<2>(cov2d_inv_and_det);
                        auto det_div_lowpass_sqrt_clamp = math_op_t::sqrt(math_op_t::max(0.000025f, det_div_lowpass)); 
                        tv::array<float, 1> dopacity_val{{$dopacity[i_with_batch]}};
                        auto ddet_div_det_lowpass_clamp = dopacity_val[0] * opacity;

                        auto ddet_div_det_lowpass = det_div_lowpass <= 0.000025f ? 0.0f : ddet_div_det_lowpass_clamp * 0.5f / det_div_lowpass_sqrt_clamp;
                        auto dcov2d = Gaussian3D::gaussian_2d_inverse_and_det_grad_with_comp(conic_grad, ddet_div_det_lowpass, 
                            cov2d_vec, {self._cfg.gaussian_lowpass_filter}f);
                        dopacity_val = (dopacity_val * det_div_lowpass_sqrt_clamp);

                        """)
                        proxy.accumulate_field_grad(GaussianCoreFields.OPACITY, "opacity", "dopacity_val[0]")
                    else:
                        code_prep.raw(f"""
                        auto dcov2d = Gaussian3D::gaussian_2d_inverse_and_det_grad(conic_grad, cov2d_vec, {self._cfg.gaussian_lowpass_filter}f);
                        """)
                    proxy.read_field(GaussianCoreFields.SCALE, "scale")
                    proxy.read_field(GaussianCoreFields.QUATERNION_XYZW, "quat")

                    if self._cfg.recalc_cov3d_in_bwd:
                        code_prep.raw(f"""
                        auto cov3d_vec = Gaussian3D::scale_quat_to_cov3d(scale, quat);
                        """)
                    else:
                        code_prep.raw(f"""
                        auto cov3d_vec = op::reinterpret_cast_array_nd<6>($cov3d_vecs)[i_with_batch];
                        """)
                    code_prep.raw(f"""
                    auto proj_grad_res = Gaussian3D::project_gaussian_to_2d_grad<float, 3>(dcov2d, point_cam, focal_xy, tan_fov, cam2world_R_T, cov3d_vec);
                    dpoint_cam += std::get<0>(proj_grad_res);

                    auto dxyz = cam2world_R_T.op<op::mv_colmajor>(dpoint_cam);
                    auto cov3d_grad_res = Gaussian3D::scale_quat_to_cov3d_grad(std::get<1>(proj_grad_res), scale, quat);
                    auto dscale = std::get<0>(cov3d_grad_res);
                    auto dquat = std::get<1>(cov3d_grad_res);

                    """)
                    proxy.accumulate_field_grad(GaussianCoreFields.SCALE, "scale", "dscale")
                    proxy.accumulate_field_grad(GaussianCoreFields.QUATERNION_XYZW, "quat", "dquat")
                    if not self._cfg.enable_anti_aliasing:
                        code_prep.raw(f"""
                        tv::array<float, 1> dopacity_val{{$dopacity[i_with_batch]}};
                        """)
                        proxy.read_field(GaussianCoreFields.OPACITY, "opacity")
                        proxy.accumulate_field_grad(GaussianCoreFields.OPACITY, "opacity", "dopacity_val[0]")

                    code_prep.raw(f"""
                    auto normed_dir = (point).op<op::normalize>();
                    auto color_grad = op::reinterpret_cast_array_nd<3>($dcolor)[i_with_batch];

                    uint8_t nn_packed = $sh_to_rgb_ne_0[i_with_batch];
                    bool rgb0_ne_0 = nn_packed & 0x1;
                    bool rgb1_ne_0 = nn_packed & 0x2;
                    bool rgb2_ne_0 = nn_packed & 0x4;
                    color_grad[0] *= rgb0_ne_0 ? 0.0f : 1.0f;
                    color_grad[1] *= rgb1_ne_0 ? 0.0f : 1.0f;
                    color_grad[2] *= rgb2_ne_0 ? 0.0f : 1.0f;
                    """)
                    proxy.accumulate_field_grad(GaussianCoreFields.RGB, "", "color_grad", "normed_dir", "dnormed_dir")
                    code_prep.raw(f"""
                    dxyz += dnormed_dir.op<op::normalize_grad>(point);
                    """)
                    proxy.accumulate_field_grad(GaussianCoreFields.XYZ, "point", "dxyz")

                if batch_size > 1:
                    proxy.save_accumulated_grad()
                add_vars = {f"{k}_ptr": v for k, v in model.get_proxy_field_dict().items()}
                add_vars.update({f"{k}_grad_ptr": v for k, v in grad_model.get_proxy_field_dict().items()})
                if prep_user_inputs is not None:
                    add_vars.update(prep_user_inputs)
                self._inliner.kernel_1d(prep_kernel_name, num, 0, code_prep,
                    additional_vars=add_vars)

                # raise NotImplementedError
        if return_uv_grad:
            return grad_model, duv.view(batch_size, -1, 2), grad_out.dcustom_features
        return grad_model, None, grad_out.dcustom_features

class _RasterizeGaussians(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        # model
        xyz,
        color_sh,
        color_sh_base,
        scale,
        quaternion_xyzw,
        opacity,
        cur_sh_degree,
        instance_id,
        custom_features,
        mask,
        model_cls: type[GaussianModelOriginBase],
        # cameras
        camera_params: CameraParamBundle | BasicPinholeCamera,
        cam2world_T: torch.Tensor | None,
        frame_ids: torch.Tensor | None,
        instance2world_T,
        # options
        training,
        op: GaussianSplatOp,
        uv_grad_holder=None,
    ):
        enable_verbose = False
        if isinstance(camera_params, CameraParamBundle):
            assert cam2world_T is not None
            cam_bundle = CameraBundle(
                focal_xy_ppoint=camera_params.focal_xy_ppoint,
                image_shape_wh=camera_params.image_shape_wh,
                cam2world_T=cam2world_T,
                frame_ids=frame_ids,
            )
            focal_xy_ppoint = camera_params.focal_xy_ppoint
        else:
            cam_bundle = camera_params
            focal_xy_ppoint = None 
        with measure_and_print_torch("FWD-all-torch-1", enable=enable_verbose):

            model = model_cls(
                xyz=xyz,
                color_sh=color_sh,
                color_sh_base=color_sh_base,
                scale=scale,
                quaternion_xyzw=quaternion_xyzw,
                opacity=opacity,
                cur_sh_degree=cur_sh_degree,
                instance_id=instance_id,
                act_applied=True,
            )
        with measure_and_print_torch("FWD-all-torch", enable=enable_verbose):
            out, fwd_ctx = op.forward(model, cam_bundle, training, instance2world_T, custom_features, mask)
        with measure_and_print_torch("FWD-all-torch-2", enable=enable_verbose):

            if fwd_ctx is not None:
                ctx.save_for_backward(
                    # ctx tensors
                    fwd_ctx.conic_opacity,
                    fwd_ctx.radii,
                    fwd_ctx.uvs,
                    fwd_ctx.rgb_gaussian,
                    fwd_ctx.gaussian_idx_sorted,
                    fwd_ctx.workload_ranges,
                    fwd_ctx.cov3d_vecs,
                    fwd_ctx.cov2d_vecs,
                    fwd_ctx.sh_to_rgb_ne_0,
                    fwd_ctx.depths,
                    # model tensors
                    xyz,
                    color_sh,
                    color_sh_base,
                    scale,
                    quaternion_xyzw,
                    opacity,
                    instance_id,
                    custom_features,
                    # cam bundle
                    cam2world_T,
                    frame_ids,
                    focal_xy_ppoint,
                    # instance pose
                    instance2world_T,
                    # outputs
                    out.n_contrib,
                    out.color,
                    out.depth,
                    out.custom_features,
                    out.T,
                )
                ctx.image_shape_wh = fwd_ctx.image_shape_wh

        ctx.op = op
        if isinstance(camera_params, BasicPinholeCamera):
            ctx.cams = camera_params
        ctx.cur_sh_degree = cur_sh_degree
        ctx.return_uv_grad = uv_grad_holder is not None
        ctx.model_cls = model_cls
        # torch autograd func only support tuple output
        out_items = (out.color, out.depth, out.custom_features, out.T,
                     out.n_contrib, out.radii)
        return out_items

    @staticmethod
    @once_differentiable
    def backward(ctx, drgb, ddepth, dcustom_features, dT, dn_contrib, dradii):
        if len(ctx.saved_tensors) == 0:
            return (None, ) * 18
        op: GaussianSplatOp = ctx.op
        if not drgb.is_contiguous():
            drgb = drgb.contiguous()
        if dcustom_features is not None and not dcustom_features.is_contiguous():
            dcustom_features = dcustom_features.contiguous()

        with tv.measure_and_print("BWD-torch-prep", enable=False):
            out = GaussianSplatOutput(
                n_contrib=ctx.saved_tensors[-5],
                color=ctx.saved_tensors[-4],
                depth=ctx.saved_tensors[-3],
                custom_features=ctx.saved_tensors[-2],
                T=ctx.saved_tensors[-1],
            )
            fwd_ctx = GaussianSplatOpContext(
                conic_opacity=ctx.saved_tensors[0],
                radii=ctx.saved_tensors[1],
                uvs=ctx.saved_tensors[2],
                rgb_gaussian=ctx.saved_tensors[3],
                gaussian_idx_sorted=ctx.saved_tensors[4],
                workload_ranges=ctx.saved_tensors[5],
                image_shape_wh=ctx.image_shape_wh,
                cov3d_vecs=ctx.saved_tensors[6],
                cov2d_vecs=ctx.saved_tensors[7],
                sh_to_rgb_ne_0=ctx.saved_tensors[8],
                depths=ctx.saved_tensors[9],
            )
            model = ctx.model_cls(
                xyz=ctx.saved_tensors[10],
                color_sh=ctx.saved_tensors[11],
                color_sh_base=ctx.saved_tensors[12],
                scale=ctx.saved_tensors[13],
                quaternion_xyzw=ctx.saved_tensors[14],
                opacity=ctx.saved_tensors[15],
                instance_id=ctx.saved_tensors[16],
                act_applied=True,
                cur_sh_degree=ctx.cur_sh_degree,
            )
            custom_features = ctx.saved_tensors[17]
            cam2world_T = ctx.saved_tensors[18]
            frame_ids = ctx.saved_tensors[19]
            focal_xy_ppoint = ctx.saved_tensors[20]
            instance2world_T = ctx.saved_tensors[21]
            if cam2world_T is not None:
                cam_bundle = CameraBundle(
                    focal_xy_ppoint=focal_xy_ppoint,
                    image_shape_wh=fwd_ctx.image_shape_wh,
                    cam2world_T=cam2world_T,
                    frame_ids=frame_ids,
                )
            else:
                cam_bundle = ctx.cams
            # if op._cfg.transmittance_is_double:
            #     dT = dT.float()
            gradient = GaussianSplatGradients(
                drgb=drgb,
                ddepth=ddepth,
                dcustom_features=dcustom_features,
                dT=dT  # .float(),
            )
            # if op.cfg.verbose:
            # print("Backward Used Memory", out.get_all_torch_tensor_byte_size() +
            #     model.get_all_torch_tensor_byte_size() +
            #         fwd_ctx.get_all_torch_tensor_byte_size() +
            #         gradient.get_all_torch_tensor_byte_size())
            # print("Out Size", out.get_all_torch_tensor_byte_size())
            # print("Model Size", model.get_all_torch_tensor_byte_size())
            # print("FwdCtx Size", fwd_ctx.get_all_torch_tensor_byte_size())
            # print("Gradient Size", gradient.get_all_torch_tensor_byte_size())
        grad_out, duv, dcf = op.backward(model, cam_bundle, fwd_ctx, out, gradient,
                                    ctx.return_uv_grad, instance2world_T, 
                                    custom_features)
        return grad_out.xyz, grad_out.color_sh, grad_out.color_sh_base, grad_out.scale, grad_out.quaternion_xyzw, grad_out.opacity, None, None, dcf, None, None, None, None, None, None, None, None, duv

class _RasterizeGaussiansWithDynamicInput(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        # model cls and autograd repr data
        model_cls: type[GaussianModelOriginBase],
        model_tensor_names: list[str],
        model_nontensor_dict: dict[str, Any],
        # cameras
        camera_params: CameraParamBundle | BasicPinholeCamera,
        cam2world_T: torch.Tensor | None,
        frame_ids: torch.Tensor | None,
        instance2world_T,
        # options
        training,
        op: GaussianSplatOp,
        user_input_tensors: dict[str, torch.Tensor] | None,
        user_inputs: dict[str, Any] | None,
        uv_grad_holder,
        custom_features,
        mask,
        # model dynamic fields
        *model_fields,
    ):
        enable_verbose = False
        if isinstance(camera_params, CameraParamBundle):
            assert cam2world_T is not None
            cam_bundle = CameraBundle(
                focal_xy_ppoint=camera_params.focal_xy_ppoint,
                image_shape_wh=camera_params.image_shape_wh,
                cam2world_T=cam2world_T,
                frame_ids=frame_ids,
            )
            focal_xy_ppoint = camera_params.focal_xy_ppoint
        else:
            cam_bundle = camera_params
            focal_xy_ppoint = None 
        if user_input_tensors is None:
            user_input_tensors = {}
        if user_inputs is None:
            user_inputs = {}

        with measure_and_print_torch("FWD-all-torch-1", enable=enable_verbose):
            model = model_cls.from_autograd_tuple_repr(model_fields, model_tensor_names, model_nontensor_dict)
        with measure_and_print_torch("FWD-all-torch", enable=enable_verbose):
            out, fwd_ctx = op.forward(model, cam_bundle, training, instance2world_T, custom_features,
                mask, {**user_input_tensors, **user_inputs})
        with measure_and_print_torch("FWD-all-torch-2", enable=enable_verbose):

            if fwd_ctx is not None:
                ctx.save_for_backward(
                    # ctx tensors
                    fwd_ctx.conic_opacity,
                    fwd_ctx.radii,
                    fwd_ctx.uvs,
                    fwd_ctx.rgb_gaussian,
                    fwd_ctx.gaussian_idx_sorted,
                    fwd_ctx.workload_ranges,
                    fwd_ctx.cov3d_vecs,
                    fwd_ctx.cov2d_vecs,
                    fwd_ctx.sh_to_rgb_ne_0,
                    fwd_ctx.depths,
                    # model tensors
                    custom_features,
                    # cam bundle
                    cam2world_T,
                    frame_ids,
                    focal_xy_ppoint,
                    # instance pose
                    instance2world_T,
                    # outputs
                    out.n_contrib,
                    out.color,
                    out.depth,
                    out.custom_features,
                    out.T,
                    # dynamic model fields
                    *model_fields,
                    # user dynamic tensors
                    *user_input_tensors.values(),
                )
                ctx.image_shape_wh = fwd_ctx.image_shape_wh
                ctx.user_inputs = user_inputs

        ctx.op = op
        if isinstance(camera_params, BasicPinholeCamera):
            ctx.cams = camera_params
        ctx.model_tensor_names = model_tensor_names
        ctx.model_nontensor_dict = model_nontensor_dict
        ctx.user_inputs = user_inputs
        ctx.user_input_tensor_names = list(user_input_tensors.keys())
        ctx.return_uv_grad = uv_grad_holder is not None
        ctx.model_cls = model_cls
        # torch autograd func only support tuple output
        out_items = (out.color, out.depth, out.custom_features, out.T,
                     out.n_contrib, out.radii)
        return out_items

    @staticmethod
    @once_differentiable
    def backward(ctx, drgb, ddepth, dcustom_features, dT, dn_contrib, dradii):
        model_tensor_names = ctx.model_tensor_names

        if len(ctx.saved_tensors) == 0:
            return (None, ) * (14 + len(model_tensor_names))
        op: GaussianSplatOp = ctx.op
        if not drgb.is_contiguous():
            drgb = drgb.contiguous()
        if dcustom_features is not None and not dcustom_features.is_contiguous():
            dcustom_features = dcustom_features.contiguous()
        with tv.measure_and_print("BWD-torch-prep", enable=False):
            fwd_ctx = GaussianSplatOpContext(
                conic_opacity=ctx.saved_tensors[0],
                radii=ctx.saved_tensors[1],
                uvs=ctx.saved_tensors[2],
                rgb_gaussian=ctx.saved_tensors[3],
                gaussian_idx_sorted=ctx.saved_tensors[4],
                workload_ranges=ctx.saved_tensors[5],
                image_shape_wh=ctx.image_shape_wh,
                cov3d_vecs=ctx.saved_tensors[6],
                cov2d_vecs=ctx.saved_tensors[7],
                sh_to_rgb_ne_0=ctx.saved_tensors[8],
                depths=ctx.saved_tensors[9],
            )
            custom_features = ctx.saved_tensors[10]
            cam2world_T = ctx.saved_tensors[11]
            frame_ids = ctx.saved_tensors[12]
            focal_xy_ppoint = ctx.saved_tensors[13]
            instance2world_T = ctx.saved_tensors[14]

            out = GaussianSplatOutput(
                n_contrib=ctx.saved_tensors[15],
                color=ctx.saved_tensors[16],
                depth=ctx.saved_tensors[17],
                custom_features=ctx.saved_tensors[18],
                T=ctx.saved_tensors[19],
            )
            model = ctx.model_cls.from_autograd_tuple_repr(ctx.saved_tensors[20:20+len(model_tensor_names)], model_tensor_names, ctx.model_nontensor_dict)
            user_input_tensors = {k: v for k, v in zip(ctx.user_input_tensor_names, ctx.saved_tensors[20+len(model_tensor_names):])}
            if cam2world_T is not None:
                cam_bundle = CameraBundle(
                    focal_xy_ppoint=focal_xy_ppoint,
                    image_shape_wh=fwd_ctx.image_shape_wh,
                    cam2world_T=cam2world_T,
                    frame_ids=frame_ids,
                )
            else:
                cam_bundle = ctx.cams
            if op._cfg.transmittance_is_double:
                dT = dT.float()
            gradient = GaussianSplatGradients(
                drgb=drgb,
                ddepth=ddepth,
                dcustom_features=dcustom_features,
                dT=dT  # .float(),
            )
        grad_out, duv, dcf = op.backward(model, cam_bundle, fwd_ctx, out, gradient,
                                    ctx.return_uv_grad, instance2world_T, 
                                    custom_features, {**user_input_tensors, **ctx.user_inputs})
        return (None, ) * 11 + (duv, dcf, None) + grad_out.to_autograd_tuple_repr_tensor_only()
        

def rasterize_gaussians(
    model: GaussianModelOriginBase,
    cameras: BasicPinholeCamera | CameraBundle,
    op: GaussianSplatOp,
    training=False,
    # background_tensor: torch.Tensor | None = None,
    uv_grad_holder: torch.Tensor | None = None,
    instance2world_T: torch.Tensor | None = None,
    custom_features: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
):
    # if not training:
    #     assert background_tensor is None, "background_tensor is only used in training mode"
    model = model.create_model_with_act()  # apply activation
    c2w_T = None 
    frame_ids = None 
    if isinstance(cameras, CameraBundle):
        c2w_T = cameras.cam2world_T
        frame_ids = cameras.frame_ids
    if instance2world_T is not None:
        assert frame_ids is not None
    res = _RasterizeGaussians.apply(
        model.xyz,
        model.color_sh,
        model.color_sh_base,
        model.scale,
        model.quaternion_xyzw,
        model.opacity,
        model.cur_sh_degree,
        model.instance_id,
        custom_features,
        mask,
        type(model),
        cameras,
        c2w_T,
        frame_ids,
        instance2world_T,
        training,
        op,
        uv_grad_holder,
    )
    out = GaussianSplatOutput(
        color=res[0],
        depth=res[1],
        custom_features=res[2],
        T=res[3],
        n_contrib=res[4],
        radii=res[5],
    )
    # if background_tensor is not None:
    #     out.color = out.color + out.T * background_tensor
    # else:
    #     if training:
    #         bg_in_cfg_device = op._cfg.get_bg_color_device(model.xyz.device)
    #         out.color = out.color + out.T.reshape(out.color.shape[0], 1, *out.color.shape[2:]) * bg_in_cfg_device.view(1, 3, 1, 1)
    return out

def rasterize_gaussians_dynamic(
    model: GaussianModelOriginBase,
    cameras: BasicPinholeCamera | CameraBundle,
    op: GaussianSplatOp,
    training=False,
    uv_grad_holder: torch.Tensor | None = None,
    instance2world_T: torch.Tensor | None = None,
    custom_features: torch.Tensor | None = None,
    prep_input_tensors: dict[str, torch.Tensor] | None = None,
    prep_inputs: dict[str, Any] | None = None,
    mask: torch.Tensor | None = None,
):
    assert op._cfg.use_proxy_model, "only v2 (proxy) is supported"
    model = model.create_model_with_act()  # apply activation
    c2w_T = None 
    frame_ids = None 
    if isinstance(cameras, CameraBundle):
        c2w_T = cameras.cam2world_T
        frame_ids = cameras.frame_ids
    if instance2world_T is not None:
        assert frame_ids is not None
    mten, mten_names, mnondict = model.to_autograd_tuple_repr()
    res = _RasterizeGaussiansWithDynamicInput.apply(
        type(model),
        mten_names,
        mnondict,
        # cam params
        cameras,
        c2w_T,
        frame_ids,
        instance2world_T,
        # options
        training,
        op,
        prep_input_tensors,
        prep_inputs,
        # special tensors
        uv_grad_holder,
        custom_features,
        mask,
        # model dynamic fields
        *mten,
    )
    out = GaussianSplatOutput(
        color=res[0],
        depth=res[1],
        custom_features=res[2],
        T=res[3],
        n_contrib=res[4],
        radii=res[5],
    )
    return out

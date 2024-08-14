from contextlib import nullcontext
from pyexpat import model
from tabnanny import verbose
import time
from typing import Annotated, Literal
from d3sim.core import dataclass_dispatch as dataclasses
from d3sim.core.arrcheck.dcbase import DataClassWithArrayCheck
import d3sim.core.arrcheck as ac
from d3sim.data.scene_def.camera import BasicPinholeCamera
from d3sim.ops.d3gs.base import GaussianModelBase, GaussianModelOrigin
from d3sim.csrc.inliner import INLINER
import numpy as np
import math
import torch
import pccm
from cumm.gemm.codeops import div_up
from d3sim.constants import IsAppleSiliconMacOs
import cumm.tensorview as tv
from d3sim.ops.d3gs.data.utils.general_utils import strip_symmetric, build_scaling_rotation
from cumm.inliner import torch_tensor_to_tv, measure_and_print_torch
from torch.autograd.function import once_differentiable
from d3sim.ops.d3gs.config_def import GaussianSplatConfig
def fov2focal(fov: float, length: int):
    return float(length) / (2 * math.tan(fov / 2))


def focal2fov(focal: float, length: int):
    return 2 * math.atan(float(length) / (2 * focal))


_DEFAULT_ENABLE_32BIT_SORT = False if IsAppleSiliconMacOs else False

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class BatchCameraParams:
    # 4x3 matrix
    cam2world_T: torch.Tensor
    focal_xy: torch.Tensor
    principal_point: torch.Tensor
    tan_fov: torch.Tensor
    shape_wh: tuple[int, int]


@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class GaussianSplatForwardContext(DataClassWithArrayCheck):
    conic_opacity: Annotated[torch.Tensor, ac.ArrayCheck([-1, 4], ac.F32)]
    radii: Annotated[torch.Tensor, ac.ArrayCheck([-1], ac.I32)]
    uvs: Annotated[torch.Tensor, ac.ArrayCheck([-1, 2], ac.F32)]
    rgb_gaussian: Annotated[torch.Tensor, ac.ArrayCheck([-1, 3], ac.F32)]
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
    custom_features: Annotated[torch.Tensor | None,
                                     ac.ArrayCheck([-1, "H", "W"], ac.F32)]
    T: Annotated[torch.Tensor, ac.ArrayCheck(["H", "W"], ac.F32)]
    color: Annotated[torch.Tensor, ac.ArrayCheck([3, "H", "W"], ac.F32)]
    depth: Annotated[torch.Tensor | None,
                           ac.ArrayCheck(["H", "W"], ac.F32)] = None
    n_contrib: Annotated[torch.Tensor | None,
                               ac.ArrayCheck(["H", "W"], ac.I32)] = None
    radii: Annotated[torch.Tensor | None,
                            ac.ArrayCheck([-1], ac.I32)] = None

    @property 
    def image_shape_wh(self) -> tuple[int, int]:
        return (self.T.shape[1], self.T.shape[0])

    

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class RasterizeGradientOutput(DataClassWithArrayCheck):
    duv: Annotated[torch.Tensor, ac.ArrayCheck(["N", 2], ac.F32)]
    dconic: Annotated[torch.Tensor, ac.ArrayCheck(["N", 3], ac.F32)]
    dopacity: Annotated[torch.Tensor, ac.ArrayCheck(["N"], ac.F32)]
    dcolor: Annotated[torch.Tensor, ac.ArrayCheck(["N", 3], ac.F32)]
    dcustom_features: Annotated[torch.Tensor | None,
                                ac.ArrayCheck(["N", -1], ac.F32)]


@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class GaussianSplatGradients(DataClassWithArrayCheck):
    drgb: Annotated[torch.Tensor, ac.ArrayCheck([-1, "H", "W"], ac.F32)]
    dT: Annotated[torch.Tensor, ac.ArrayCheck(["H", "W"], ac.F32)]
    ddepth: Annotated[torch.Tensor | None, ac.ArrayCheck(["H", "W"], ac.F32)]
    dcustom_features: Annotated[torch.Tensor | None,
                                ac.ArrayCheck([-1, "H", "W"], ac.F32)]


class GaussianSplatForward:
    def __init__(self, cfg: GaussianSplatConfig) -> None:
        self.cfg = cfg

    def forward(
        self,
        model: GaussianModelBase,
        cameras: list[BasicPinholeCamera],
        training: bool = False,
    ):
        enable_verbose = self.cfg.verbose
        # enable_verbose = False

        with tv.measure_and_print("forward-0", enable=enable_verbose):
            assert model.act_applied, "model must be activated before render"
            num = len(model)
            scale_global = self.cfg.scale_global
            enable_32bit_sort = self.cfg.enable_32bit_sort
            if len(cameras) > 1:
                raise NotImplementedError("multi camera not supported yet")
            camera = cameras[0]
            intrinsic = camera.intrinsic
            cam2world = camera.pose.to_world
            image_shape_wh = camera.image_shape_wh
            axis_front_u_v = camera.axes_front_u_v

            width = image_shape_wh[0]
            height = image_shape_wh[1]

            focal_x = intrinsic[0, 0]
            focal_y = intrinsic[1, 1]

            fov_x = focal2fov(focal_x, width)
            fov_y = focal2fov(focal_y, height)
            tanfov_x = math.tan(fov_x / 2)
            tanfov_y = math.tan(fov_y / 2)
            resolution_wh_np = np.array([width, height], np.int32)

            focal_xy_np = np.array([focal_x, focal_y], np.float32)
            principal_point_np = np.array([intrinsic[0, 2], intrinsic[1, 2]],
                                        np.float32)
            tanfov_xy_np = np.array([tanfov_x, tanfov_y], np.float32)
            cam2world_T_np = np.ascontiguousarray(cam2world[:3].T)
            cam2world_center_np = cam2world_T_np[3]
            tile_num_x = div_up(width, self.cfg.tile_size[0])
            tile_num_y = div_up(height, self.cfg.tile_size[1])
            if enable_32bit_sort:
                assert tile_num_x * tile_num_y < 2**13, "32bit sort only support 13 bits tile idx"
            block_size = self.cfg.block_size
            tile_num_xy_np = np.array([tile_num_x, tile_num_y], np.int32)
            xyz = model.xyz
            # quat should be normed in user-defined quaternion_xyzw_act.
            quat_xyzw = model.quaternion_xyzw
            scales = model.scale
            opacity = model.opacity
            color_sh = model.color_sh
            color_sh_base = model.color_sh_base
            degree = model.color_sh_degree
            cur_degree = model.cur_sh_degree

            depths = torch.empty(num, dtype=torch.float32, device=xyz.device)
            radii = torch.empty(num, dtype=torch.int32, device=xyz.device)
            uvs = torch.empty([num, 2], dtype=torch.float32, device=xyz.device)
            conic_opacity = torch.empty([num, 4],
                                        dtype=torch.float32,
                                        device=xyz.device)
            cov3d_vecs = None
            cov2d_vecs = None
            sh_to_rgb_ne_0 = None 
            if training:
                # for backward only
                if not self.cfg.recalc_cov3d_in_bwd:
                    cov3d_vecs = torch.empty([num, 6],
                                            dtype=torch.float32,
                                            device=xyz.device)
                cov2d_vecs = torch.empty([num, 3],
                                            dtype=torch.float32,
                                            device=xyz.device)
                sh_to_rgb_ne_0 = torch.empty([num], dtype=torch.uint8, device=xyz.device)
            tiles_touched = torch.empty(num, dtype=torch.int32, device=xyz.device)
            rgb_gaussian = torch.empty([num, 3],
                                    dtype=torch.float32,
                                    device=xyz.device)
        # debug_cov2d = torch.empty([num, 3], dtype=torch.float32, device=xyz.device)

        # debug_ten = torch.empty(num, dtype=torch.float32, device=xyz.device)
        custom_feat = model.custom_features
        num_custom_feat = 0 if custom_feat is None else custom_feat.shape[1]
        t1 = time.time()
        fused_scale_act = model.fused_scale_act_op
        fused_q_act = model.fused_quaternion_xyzw_act_op
        fused_opacity_act = model.fused_opacity_act_op
        has_color_base = color_sh_base is not None
        with measure_and_print_torch("1", enable=enable_verbose):
            prep_kernel_name = (f"gs3d_preprocess_{axis_front_u_v}_{self.cfg.tile_size}_{degree}_{training}_"
                                f"{fused_scale_act}_{fused_q_act}_{fused_opacity_act}_"
                                f"{cur_degree}_{has_color_base}")
            code_prep = pccm.code()
            lowpass_filter = self.cfg.gaussian_lowpass_filter
            eps = self.cfg.eps
            cov2d_radii_eigen_eps = self.cfg.cov2d_radii_eigen_eps
            gaussian_std_sigma = self.cfg.gaussian_std_sigma
            projected_clamp_factor = self.cfg.projected_clamp_factor
            code_prep.raw(f"""
            namespace op = tv::arrayops;
            using math_op_t = tv::arrayops::MathScalarOp<float>;
            auto cam2world_T = $cam2world_T_np;
            auto focal_xy = $focal_xy_np;
            auto principal_point = $principal_point_np;
            auto tan_fov = $tanfov_xy_np;
            auto resolution_wh = $resolution_wh_np;
            auto point = op::reinterpret_cast_array_nd<3>($xyz)[i];
            auto point_cam = op::slice<0, 3>(cam2world_T).op<op::mv_rowmajor>(point - cam2world_T[3]);

            auto uvz = CameraOps::pos_cam_to_uv_no_distort<2, 0, 1>(point_cam, principal_point, focal_xy);
            auto uv = std::get<0>(uvz) - 0.5f;
            auto z_in_cam = std::get<1>(uvz);
            auto scale = op::reinterpret_cast_array_nd<3>($scales)[i];
            auto quat = op::reinterpret_cast_array_nd<4>($quat_xyzw)[i];
            """)
            if fused_q_act is not None:
                code_prep.raw(f"""
                quat = quat.op<op::{fused_q_act[0]}>();
                """)
            if fused_scale_act is not None:
                code_prep.raw(f"""
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
            auto cov2d_vec = Gaussian3D::project_gaussian_to_2d<float, 4>(point_cam, focal_xy, tan_fov, cam2world_T, cov3d_vec, $projected_clamp_factor);
            
            cov2d_vec[0] += $lowpass_filter;
            cov2d_vec[2] += $lowpass_filter;

            auto cov2d_inv_and_det = Gaussian3D::gaussian_2d_inverse_and_det(cov2d_vec, $eps);
            auto cov2d_inv = std::get<0>(cov2d_inv_and_det);
            auto det = std::get<1>(cov2d_inv_and_det);
            auto radii_fp = math_op_t::ceil(Gaussian3D::get_gaussian_2d_ellipse(cov2d_vec, det, 
                $cov2d_radii_eigen_eps, $gaussian_std_sigma));
            constexpr auto tile_size_xy = tv::array<int, 2>{{{self.cfg.tile_size[0]}, {self.cfg.tile_size[1]}}};
            auto tile_num_xy = tv::div_up(resolution_wh, tile_size_xy);
            auto tile_size_xy_float = tile_size_xy.cast<float>();
            auto gaussian_rect_min = ((uv - radii_fp) / tile_size_xy_float).cast<int>().op<op::clamp>(0, tile_num_xy);
            auto gaussian_rect_max = ((uv + radii_fp + tile_size_xy_float - 1) / tile_size_xy_float).cast<int>().op<op::clamp>(0, tile_num_xy);
            int rect_area = (gaussian_rect_max[0] - gaussian_rect_min[0]) * (gaussian_rect_max[1] - gaussian_rect_min[1]);
            bool is_empty = (det == 0 || z_in_cam < 0.2f) || rect_area == 0;

            $tiles_touched[i] = is_empty ? 0 : rect_area;
            $radii[i] = is_empty ? 0 : int(radii_fp);
            if (is_empty){{
                // calc color sh requires much IO, so we early exit here.
                // this will cause warp divergence, but save io.
                return;
            }}
            $depths[i] = z_in_cam;

            op::reinterpret_cast_array_nd<2>($uvs)[i] = uv;
            auto opacity_val = $opacity[i];
            """)
            if fused_opacity_act is not None:
                code_prep.raw(f"""
                opacity_val = tv::array<float, 1>{{opacity_val}}.op<op::{fused_opacity_act[0]}>()[0];
                """)
            code_prep.raw(f"""
            tv::array<float, 4> conic_opacity_vec{{cov2d_inv[0], cov2d_inv[1], cov2d_inv[2], opacity_val}};
            op::reinterpret_cast_array_nd<4>($conic_opacity)[i] = conic_opacity_vec;
            auto sh_ptr = op::reinterpret_cast_array_nd<3>($color_sh) + i * {(degree + 1) * (degree + 1) - has_color_base};
            
            """)
            if (training):
                if has_color_base:
                    code_prep.raw(f"""
                    auto sh_base_ptr = op::reinterpret_cast_array_nd<3>($color_sh_base) + i;
                    auto rgb = Gaussian3D::sh_dir_to_rgb<{cur_degree}>((point - cam2world_T[3]).op<op::normalize>(), sh_ptr, sh_base_ptr);
                    """)
                else:
                    code_prep.raw(f"""
                    auto rgb = Gaussian3D::sh_dir_to_rgb<{cur_degree}>((point - cam2world_T[3]).op<op::normalize>(), sh_ptr);
                    """)
                code_prep.raw(f"""
                uint8_t ne_0_packed = (uint8_t(rgb[0] < 0) << 2) | (uint8_t(rgb[1] < 0) << 1) | uint8_t(rgb[2] < 0);
                op::reinterpret_cast_array_nd<3>($rgb_gaussian)[i] = rgb.op<op::maximum>(0.0f);
                $sh_to_rgb_ne_0[i] = ne_0_packed;

                op::reinterpret_cast_array_nd<3>($cov2d_vecs)[i] = cov2d_vec;
                """)
                if not self.cfg.recalc_cov3d_in_bwd:
                    code_prep.raw(f"""
                    op::reinterpret_cast_array_nd<6>($cov3d_vecs)[i] = cov3d_vec;
                    """)
            else:
                if has_color_base:
                    code_prep.raw(f"""
                    auto sh_base_ptr = op::reinterpret_cast_array_nd<3>($color_sh_base) + i;
                    op::reinterpret_cast_array_nd<3>($rgb_gaussian)[i] = Gaussian3D::sh_dir_to_rgb<{cur_degree}>((point - cam2world_T[3]).op<op::normalize>(), sh_ptr, sh_base_ptr).op<op::maximum>(0.0f);
                    """)
                else:
                    code_prep.raw(f"""
                    op::reinterpret_cast_array_nd<3>($rgb_gaussian)[i] = Gaussian3D::sh_dir_to_rgb<{cur_degree}>((point - cam2world_T[3]).op<op::normalize>(), sh_ptr).op<op::maximum>(0.0f);
                    """)
            # with measure_and_print_torch("1", enable=True):

            INLINER.kernel_1d(prep_kernel_name, num, 0, code_prep)
            # print(INLINER.get_nvrtc_module(prep_kernel_name).params.debug_code)
        with measure_and_print_torch("2", enable=enable_verbose):
            tiles_touched.cumsum_(0)
            num_rendered = int(tiles_touched[-1].item())
        out_img_shape = [self.cfg.num_channels, height, width] if self.cfg.use_nchw else [height, width, self.cfg.num_channels]
        out_custom_feat_shape = [num_custom_feat, height, width] if self.cfg.use_nchw else [height, width, num_custom_feat]
        if (num_rendered == 0):
            final_T = torch.empty([height, width],
                                  dtype=torch.float32,
                                  device=xyz.device)
            n_contrib = torch.empty([height, width],
                                    dtype=torch.int32,
                                    device=xyz.device)
            out_color = torch.empty(out_img_shape,
                                    dtype=torch.float32,
                                    device=xyz.device)
            out_custom_feat = torch.empty(out_custom_feat_shape,
                                          dtype=torch.float32,
                                          device=xyz.device)
            final_depth = None
            if self.cfg.render_depth:
                final_depth = torch.zeros([height, width],
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
                                   dtype=torch.int32,
                                   device=xyz.device)
        with measure_and_print_torch(f"3 {num_rendered}",
                                     enable=enable_verbose):
            code = pccm.code()
            code.raw(f"""
            namespace op = tv::arrayops;
            using math_op_t = tv::arrayops::MathScalarOp<float>;

            auto radii_val = $radii[i];
            auto radii_fp = float(radii_val);
            constexpr auto tile_size_xy = tv::array<int, 2>{{{self.cfg.tile_size[0]}, {self.cfg.tile_size[1]}}};
            auto tile_num_xy = $tile_num_xy_np;
            auto tile_size_xy_float = tile_size_xy.cast<float>();
            """)
            with code.if_("radii_val > 0"):
                if enable_32bit_sort:
                    code.raw(f"""
                    auto depth_uint = uint32_t($depths[i] / $(self.cfg.depth_32bit_prec)) & 0x3FFFF;
                    """)
                else:
                    code.raw(f"""
                    auto depth_uint = reinterpret_cast<const TV_METAL_DEVICE uint32_t*>($depths)[i];
                    """)
                code.raw(f"""
                auto offset = i == 0 ? 0 : $tiles_touched[i - 1];
                auto uv = op::reinterpret_cast_alignedarray<2>($uvs)[i];
                auto gaussian_rect_min = ((uv - radii_fp) / tile_size_xy_float).cast<int>().op<op::clamp>(0, tile_num_xy);
                auto gaussian_rect_max = ((uv + radii_fp + tile_size_xy_float - 1) / tile_size_xy_float).cast<int>().op<op::clamp>(0, tile_num_xy);
                """)
                with code.for_(
                        "int y = gaussian_rect_min[1]; y < gaussian_rect_max[1]; ++y"
                ):
                    with code.for_(
                            "int x = gaussian_rect_min[0]; x < gaussian_rect_max[0]; ++x"
                    ):
                        shift_bits = 32 if not enable_32bit_sort else 18
                        if enable_32bit_sort:
                            code.raw(f"""
                            uint32_t key = y * tile_num_xy[0] + x;
                            """)
                        else:
                            code.raw(f"""
                            uint64_t key = y * tile_num_xy[0] + x;
                            """)
                        code.raw(f"""
                        key <<= {shift_bits};
                        key |= depth_uint; // assume depth always > 0
                        $keys_tile_idx_depth[offset] = key;
                        $gaussian_idx[offset] = i;
                        ++offset;
                        """)

            INLINER.kernel_1d(f"prepare_sort_data_{enable_32bit_sort}", num, 0,
                              code)
        # TODO use radix sort with trunated bits (faster) for cuda
        # TODO use 32bit sort, for 1920x1080 tile, the max tile idx is 8100, 13 bits
        # so we can use 18bit for depth. (19bit if torch support uint32, true in torch >= 2.3 and cuda)
        with measure_and_print_torch("4", enable=enable_verbose):
            sorted_vals, indices = torch.sort(keys_tile_idx_depth)
            gaussian_idx_sorted = torch.gather(gaussian_idx, 0, indices)

        with measure_and_print_torch("4.2", enable=enable_verbose):
            workload_ranges = torch.empty([tile_num_x * tile_num_y, 2],
                                          dtype=torch.int32,
                                          device=xyz.device)
            shift_bits = 32 if not enable_32bit_sort else 18
            INLINER.kernel_1d(
                f"prepare_workload_range_{enable_32bit_sort}", num_rendered, 0,
                f"""
            auto key = $sorted_vals[i];
            uint32_t tile_idx = key >> {shift_bits};
            if (i == 0){{
                $workload_ranges[tile_idx * 2 + 0] = 0;
            }}else{{
                auto last_key = $sorted_vals[i - 1];
                uint32_t last_tile_idx = last_key >> {shift_bits};
                if (tile_idx != last_tile_idx){{
                    $workload_ranges[last_tile_idx * 2 + 1] = i;
                    $workload_ranges[tile_idx * 2 + 0] = i;
                }}
            }}
            if (i == $num_rendered - 1){{
                $workload_ranges[tile_idx * 2 + 1] = $num_rendered;
            }}
            """)
        # workload_ranges_tv = torch_tensor_to_tv(workload_ranges, to_const=True)
        with measure_and_print_torch("5-prep", enable=enable_verbose):

            final_T = torch.empty([height, width],
                                dtype=torch.float64 if self.cfg.transmittance_is_double else torch.float32,
                                device=xyz.device)
            n_contrib = torch.empty([height, width],
                                    dtype=torch.int32,
                                    device=xyz.device)
            out_color = torch.empty(out_img_shape,
                                    dtype=torch.float32,
                                    device=xyz.device)
            out_custom_feat = torch.empty(out_custom_feat_shape,
                                        dtype=torch.float32,
                                        device=xyz.device)
            final_depth = None
            if self.cfg.render_depth:
                final_depth = torch.zeros([height, width],
                                        dtype=torch.float32,
                                        device=xyz.device)
            output_dc = GaussianSplatOutput(custom_features=out_custom_feat,
                                            T=final_T,
                                            color=out_color,
                                            depth=final_depth,
                                            n_contrib=n_contrib,
                                            radii=radii)
            ctx = GaussianSplatForwardContext(
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
        with measure_and_print_torch("5", enable=enable_verbose):
            self.rasterize_forward_backward(model,
                                            ctx,
                                            output_dc,
                                            training=training,
                                            render_depth=self.cfg.render_depth)
        if not training:
            ctx = None

        return output_dc, ctx

    def rasterize_forward_backward(self,
                                   model: GaussianModelBase,
                                   ctx: GaussianSplatForwardContext,
                                   out: GaussianSplatOutput,
                                   grad: GaussianSplatGradients | None = None,
                                   training: bool = False,
                                   render_depth: bool = False):
        """if grad is not None, run backward mode, otherwise run forward mode.
        rasterize backward share many code with forward, so we use a single function to handle both.
        """
        num = model.xyz.shape[0]
        is_bwd = grad is not None
        image_shape_wh = ctx.image_shape_wh
        block_size = self.cfg.block_size
        width = image_shape_wh[0]
        height = image_shape_wh[1]

        depths = ctx.depths

        tile_num_x = div_up(width, self.cfg.tile_size[0])
        tile_num_y = div_up(height, self.cfg.tile_size[1])
        workload_ranges = ctx.workload_ranges
        conic_opacity_tv = torch_tensor_to_tv(ctx.conic_opacity, to_const=True)
        uvs_tv = torch_tensor_to_tv(ctx.uvs, to_const=True)
        gaussian_idx_sorted_tv = torch_tensor_to_tv(ctx.gaussian_idx_sorted,
                                                    to_const=True)
        rgb_gaussian_tv = torch_tensor_to_tv(ctx.rgb_gaussian, to_const=True)
        final_custom_feat = out.custom_features
        num_custom_feat = 0 if final_custom_feat is None else final_custom_feat.shape[
            0]

        final_T = out.T
        final_color = out.color
        final_n_contrib = out.n_contrib
        assert final_n_contrib is not None 
        final_depth = out.depth
        if render_depth:
            assert final_depth is not None

        grad_out: RasterizeGradientOutput | None = None
        color_use_smem = is_bwd  # TODO why bwd use smem for color?
        if grad is not None:
            drgb = grad.drgb 
            dT = grad.dT
        if is_bwd:
            duv = torch.zeros([num, 2],
                              dtype=torch.float32,
                              device=final_T.device)
            dconic = torch.zeros([num, 3],
                                 dtype=torch.float32,
                                 device=final_T.device)
            dopacity = torch.zeros([num],
                                   dtype=torch.float32,
                                   device=final_T.device)
            dcolor = torch.zeros([num, 3],
                                 dtype=torch.float32,
                                 device=final_T.device)
            dcustom_features = torch.zeros([num, num_custom_feat],
                                           dtype=torch.float32,
                                           device=final_T.device)
            grad_out = RasterizeGradientOutput(
                duv=duv,
                dconic=dconic,
                dopacity=dopacity,
                dcolor=dcolor,
                dcustom_features=dcustom_features)
        bwd_reduce_method = self.cfg.backward_reduction
        t_dtype = "double" if self.cfg.transmittance_is_double else "float"
        use_bwd_reduce = bwd_reduce_method != "none"
        kernel_unique_name = (
            f"rasterize_{is_bwd}_{self.cfg.tile_size}"
            f"_{num_custom_feat}_{self.cfg.render_depth}"
            f"_{self.cfg.use_nchw}_{self.cfg.render_rgba}"
            f"_{final_n_contrib is None}_{t_dtype}_{self.cfg.backward_reduction}")

        code_rasterize = pccm.code()
        code_rasterize.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<float>;
        """)
        # when use raw kernel, we don't use nonuniform grid to
        # keep logic same as cuda.
        launch_param = tv.LaunchParam(
            (tile_num_x, tile_num_y, 1),
            (self.cfg.tile_size[0], self.cfg.tile_size[1], 1))
        if IsAppleSiliconMacOs:
            code_rasterize.raw(f"""
            tv::array<uint32_t, 2> pixel_idx_xy{{threadPositionInGrid.x, threadPositionInGrid.y}};
            tv::array<uint32_t, 2> tile_idx_xy{{threadgroupPositionInGrid.x, threadgroupPositionInGrid.y}};
            // keep in mind that metal only have 32KB shared memory
            threadgroup int num_done_shared[32];
            uint thread_rank = threadPositionInThreadgroup.y * {self.cfg.tile_size[0]} + threadPositionInThreadgroup.x;
            """)
        else:
            launch_param = tv.LaunchParam(
                (tile_num_x, tile_num_y, 1),
                (self.cfg.tile_size[0], self.cfg.tile_size[1], 1))
            code_rasterize.raw(f"""
            tv::array<uint32_t, 2> pixel_idx_xy{{blockIdx.x * {self.cfg.tile_size[0]} + threadIdx.x, blockIdx.y * {self.cfg.tile_size[1]} + threadIdx.y}};
            tv::array<uint32_t, 2> tile_idx_xy{{blockIdx.x, blockIdx.y}};
            uint32_t thread_rank = threadIdx.y * {self.cfg.tile_size[0]} + threadIdx.x;
            """)
        code_rasterize.raw(f"""
        TV_SHARED_MEMORY int collected_id[{block_size}];
        TV_SHARED_MEMORY float2 collected_xy[{block_size}];
        TV_SHARED_MEMORY float4 collected_conic_opacity[{block_size}];
        """)
        if color_use_smem:
            code_rasterize.raw(f"""
            TV_SHARED_MEMORY float collected_rgb_gaussian[{block_size * 3}];
            """)
        if render_depth:
            code_rasterize.raw(f"""
            TV_SHARED_MEMORY float collected_depth[{block_size}];
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

        uint32_t pixel_id = $width * pixel_idx_xy[1] + pixel_idx_xy[0];
        auto pixel_idx_xy_fp = pixel_idx_xy.cast<float>();

        // Check if this thread is associated with a valid pixel or outside.
        // Done threads can help with fetching, but don't rasterize
        bool done = !inside;

        auto range = op::reinterpret_cast_array_nd<2>($workload_ranges)[tile_idx_xy[1] * $tile_num_x + tile_idx_xy[0]];
        
        {t_dtype} T = 1.0f;
        """)
        if not is_bwd:
            code_rasterize.raw(f"""
            tv::array<float, 3> rgb{{}};
            int toDo = range[1] - range[0];
            int rounds = tv::div_up(range[1] - range[0], {block_size});
            float weight_sum = 0.0f;    
            uint32_t contributor = 0;
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

        if render_depth:
            code_rasterize.raw(f"""
            float depth_accum = 0.0f;
            """)
        if is_bwd:
            code_rasterize.raw(f"""
            decltype(T) render_T = inside ? $final_T[pixel_id] : 0.0f;
            auto dT_val = inside ? $dT[pixel_id] : 0.0f;
            """)
            if self.cfg.use_nchw:
                code_rasterize.raw(f"""
                tv::array<float, 3> render_rgb, drgb_array;
                render_rgb[0] = inside ? $final_color[0 * $height * $width + pixel_id] : 0.0f;
                render_rgb[1] = inside ? $final_color[1 * $height * $width + pixel_id] : 0.0f;
                render_rgb[2] = inside ? $final_color[2 * $height * $width + pixel_id] : 0.0f;

                drgb_array[0] = inside ? $drgb[0 * $height * $width + pixel_id] : 0.0f;
                drgb_array[1] = inside ? $drgb[1 * $height * $width + pixel_id] : 0.0f;
                drgb_array[2] = inside ? $drgb[2 * $height * $width + pixel_id] : 0.0f;
                """)
            else:
                code_rasterize.raw(f"""
                tv::array<float, 3> render_rgb = inside ? op::reinterpret_cast_array_nd<3>($final_color)[pixel_id] : tv::array<float, 3>{{}};
                auto drgb_array = inside ? op::reinterpret_cast_array_nd<3>($drgb)[pixel_id] : tv::array<float, 3>{{}};
                """)

            if render_depth:
                code_rasterize.raw(f"""
                auto render_depth = inside ? $final_depth[pixel_id] : 0.0f;
                auto ddepth = inside ? $(grad.ddepth)[pixel_id] : 0.0f;
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
                if color_use_smem:
                    code_rasterize.raw(f"""
                    auto rgb_gaussian_val = op::reinterpret_cast_array_nd<3>($rgb_gaussian_tv)[gaussian_id];
                    collected_rgb_gaussian[thread_rank * 3 + 0] = rgb_gaussian_val[0];
                    collected_rgb_gaussian[thread_rank * 3 + 1] = rgb_gaussian_val[1];
                    collected_rgb_gaussian[thread_rank * 3 + 2] = rgb_gaussian_val[2];
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
                
                """)
                if is_bwd and use_bwd_reduce:
                    code_rasterize.raw(f"""
                    float next_T = T * (1.0f - alpha);
                    bool valid = power <= 0.0f && alpha >= 1.0f / 255.0f && (i * {block_size} + j < contributor); 
                    """)
                    if bwd_reduce_method == "warp":
                        code_rasterize.raw(f"""
                        // avoid empty atomic write
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
                        // TODO if remove the 'f', this kernel runs 4x slower.
                        if (alpha < 1.0f / 255.0f || power > 0.0f){{
                            continue;
                        }}
                        float next_T = T * (1.0f - alpha);
                        """)

                    else:
                        code_rasterize.raw(f"""
                        // TODO if remove the 'f', this kernel runs 4x slower.
                        if (alpha < 1.0f / 255.0f){{
                            continue;
                        }}
                        float next_T = T * (1.0f - alpha);
                        """)
                if not is_bwd:
                    code_rasterize.raw(f"""
                    bool should_done = next_T < {self.cfg.transmittance_eps}f;
                    if (power > 0.0f || should_done){{
                        done = should_done;
                        continue;
                    }}
                    """)
                ifctx = nullcontext()
                if is_bwd and use_bwd_reduce:
                    code_rasterize.raw(f"""
                    tv::array<float, 3> dL_drgb{{}};
                    tv::array<float, 2> dL_duv{{}};
                    tv::array<float, 3> dL_dcov2d{{}};
                    float dL_dopacity = 0.0f;
                    """)
                    ifctx = code_rasterize.if_("valid")
                with ifctx:
                    code_rasterize.raw(f"""
                    float weight = alpha * T;
                    T = next_T;
                    """)
                    if not is_bwd:
                        code_rasterize.raw(f"""
                        weight_sum += weight;
                        contributor = i * {block_size} + j;
                        """)
                    gaussian_id_inited = False
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
                    if is_bwd:
                        code_rasterize.raw(f"""
                        render_rgb -= weight * rgb_val;
                        """)
                    if not is_bwd:
                        code_rasterize.raw(f"""
                        rgb += weight * rgb_val;
                        """)
                        if render_depth:
                            code_rasterize.raw(f"""
                            auto depth_val = collected_depth[j];
                            depth_accum += weight * depth_val;
                            """)
                    if is_bwd:
                        code_rasterize.raw(f"""
                        float dL_dalpha_without_div = (drgb_array.op<op::dot>(T * rgb_val - render_rgb));
                        // grad from T, we don't apply background when training, apply it in torch side.
                        dL_dalpha_without_div += -dT_val * render_T;
                        """)
                        if render_depth:
                            code_rasterize.raw(f"""
                            dL_dalpha_without_div += ddepth * (T * depth_val - (render_depth - depth_accum));
                            """)
                        code_rasterize.raw(f"""
                        auto dL_dalpha = dL_dalpha_without_div / (1.0f - alpha);
                        {"" if use_bwd_reduce else "auto"} dL_drgb = weight * drgb_array;

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
                        code_rasterize.raw(f"""
                        dL_drgb[0] = tv::parallel::warp_sum(dL_drgb[0]);
                        dL_drgb[1] = tv::parallel::warp_sum(dL_drgb[1]);
                        dL_drgb[2] = tv::parallel::warp_sum(dL_drgb[2]);

                        dL_duv[0] = tv::parallel::warp_sum(dL_duv[0]);
                        dL_duv[1] = tv::parallel::warp_sum(dL_duv[1]);

                        dL_dcov2d[0] = tv::parallel::warp_sum(dL_dcov2d[0]);
                        dL_dcov2d[1] = tv::parallel::warp_sum(dL_dcov2d[1]);
                        dL_dcov2d[2] = tv::parallel::warp_sum(dL_dcov2d[2]);

                        dL_dopacity = tv::parallel::warp_sum(dL_dopacity);
                        """)
                        if self.cfg.backward_reduction == "block":
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
                        if self.cfg.backward_reduction == "block":
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
                        code_rasterize.raw(f"""
                        tv::parallel::atomicAdd($dcolor + gaussian_id * 3 + 0, dL_drgb[0]);
                        tv::parallel::atomicAdd($dcolor + gaussian_id * 3 + 1, dL_drgb[1]);
                        tv::parallel::atomicAdd($dcolor + gaussian_id * 3 + 2, dL_drgb[2]);

                        tv::parallel::atomicAdd($duv + gaussian_id * 2 + 0, dL_duv[0]);
                        tv::parallel::atomicAdd($duv + gaussian_id * 2 + 1, dL_duv[1]);

                        tv::parallel::atomicAdd($dconic + gaussian_id * 3 + 0, dL_dcov2d[0]);
                        tv::parallel::atomicAdd($dconic + gaussian_id * 3 + 1, dL_dcov2d[1]);
                        tv::parallel::atomicAdd($dconic + gaussian_id * 3 + 2, dL_dcov2d[2]);
                        tv::parallel::atomicAdd($dopacity + gaussian_id, dL_dopacity);
                        """)
                    # if num_custom_feat > 0:
                    #     code_rasterize.raw(f"""
                    #     for (int channel_idx = 0; channel_idx < {num_custom_feat}; ++channel_idx){{
                    #         custom_feature[channel_idx] += weight * $custom_feat[gaussian_id * {num_custom_feat} + channel_idx];
                    #     }}
                    #     """)
        if not is_bwd:
            if not training:
                code_rasterize.raw(f"""
                auto bg_color_val = $(self.cfg.bg_color);
                """)
            with code_rasterize.if_("inside"):
                code_rasterize.raw(f"""
                $final_T[pixel_id] = T;
                $final_n_contrib[pixel_id] = contributor;
                """)
                if render_depth:
                    if not training:
                        code_rasterize.raw(f"""
                        $final_depth[pixel_id] = depth_accum / (1e-6f + weight_sum);
                        """)
                    else:
                        code_rasterize.raw(f"""
                        $final_depth[pixel_id] = depth_accum;
                        """)

                if not training:
                    # use fused background color on inference.
                    # TODO how to support envmap?
                    code_rasterize.raw(f"""
                    auto rgb_before_clamp = rgb + T * bg_color_val;
                    rgb_before_clamp = rgb_before_clamp.op<op::clamp>(0.0f, 1.0f);
                    $final_color[0 * $height * $width + pixel_id] = rgb_before_clamp[0];
                    $final_color[1 * $height * $width + pixel_id] = rgb_before_clamp[1];
                    $final_color[2 * $height * $width + pixel_id] = rgb_before_clamp[2];
                    """)
                else:
                    code_rasterize.raw(f"""
                    $final_color[0 * $height * $width + pixel_id] = rgb[0];
                    $final_color[1 * $height * $width + pixel_id] = rgb[1];
                    $final_color[2 * $height * $width + pixel_id] = rgb[2];
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
        with measure_and_print_torch(f"BWD-rasterize-{gaussian_idx_sorted_tv.shape[0]}", enable=self.cfg.verbose):
            INLINER.kernel_raw(kernel_unique_name, launch_param, code_rasterize)
        # if is_bwd:
        #     # print(INLINER.get_nvrtc_module(kernel_unique_name).params.debug_code)

        #     print(INLINER.get_nvrtc_kernel_attrs(kernel_unique_name))
        return grad_out

    def backward(self, model: GaussianModelBase,
                 cameras: list[BasicPinholeCamera],
                 ctx: GaussianSplatForwardContext, out: GaussianSplatOutput,
                 grad: GaussianSplatGradients,
                 return_uv_grad: bool = False):
        if len(cameras) > 1:
            raise NotImplementedError("multi camera not supported yet")
        num = model.xyz.shape[0]
        camera = cameras[0]
        intrinsic = camera.intrinsic
        cam2world = camera.pose.to_world
        image_shape_wh = camera.image_shape_wh
        axis_front_u_v = camera.axes_front_u_v

        width = image_shape_wh[0]
        height = image_shape_wh[1]

        tile_num_x = div_up(width, self.cfg.tile_size[0])
        tile_num_y = div_up(height, self.cfg.tile_size[1])
        radii = ctx.radii
        grad_out = self.rasterize_forward_backward(model, ctx, out, grad)
        assert grad_out is not None
        duv = grad_out.duv
        dconic = grad_out.dconic
        dopacity = grad_out.dopacity
        dcolor = grad_out.dcolor

        focal_xy_np = camera.focal_length
        principal_point_np = camera.principal_point
        fov_xy = camera.fov_xy
        tanfov_x = math.tan(fov_xy[0] / 2)
        tanfov_y = math.tan(fov_xy[1] / 2)
        resolution_wh_np = np.array([width, height], np.int32)

        tanfov_xy_np = np.array([tanfov_x, tanfov_y], np.float32)
        cam2world_T_np = np.ascontiguousarray(cam2world[:3].T)
        cam2world_R_np = cam2world_T_np[:3]
        conic_opacity = ctx.conic_opacity
        cam2world_center_np = cam2world_T_np[3]
        cov3d_vecs = ctx.cov3d_vecs
        cov2d_vecs = ctx.cov2d_vecs
        sh_to_rgb_ne_0 = ctx.sh_to_rgb_ne_0
        if not self.cfg.recalc_cov3d_in_bwd:
            assert cov3d_vecs is not None
        assert cov2d_vecs is not None
        assert sh_to_rgb_ne_0 is not None 
        dxyz_res = torch.zeros_like(model.xyz)
        dscale_res = torch.zeros_like(model.scale)
        dquat_res = torch.zeros_like(model.quaternion_xyzw)
        dcolor_sh_res = torch.zeros_like(model.color_sh)
        dcolor_base_sh_res = None 
        has_color_base = model.color_sh_base is not None
        if model.color_sh_base is not None:
            dcolor_base_sh_res = torch.zeros_like(model.color_sh_base)
        # dopacity = torch.zeros_like(model.opacity)
        grad_model = GaussianModelOrigin(xyz=dxyz_res,
                                         color_sh=dcolor_sh_res,
                                         scale=dscale_res,
                                         quaternion_xyzw=dquat_res,
                                         opacity=dopacity,
                                         color_sh_base=dcolor_base_sh_res)
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
        prep_kernel_name = (f"gs3d_preprocess_bwd_{axis_front_u_v}_{self.cfg.tile_size}_{degree}_"
                            f"{fused_scale_act}_{fused_q_act}_{fused_opacity_act}_"
                            f"{cur_degree}_{has_color_base}")
        with tv.measure_and_print("BWD-prep", enable=self.cfg.verbose):
            code_prep = pccm.code()
            code_prep.raw(f"""
            namespace op = tv::arrayops;
            using math_op_t = tv::arrayops::MathScalarOp<float>;
            if ($radii[i] == 0){{
                return;
            }}
            auto uv_grad = op::reinterpret_cast_array_nd<2>($duv)[i];
            auto conic_grad = op::reinterpret_cast_array_nd<3>($dconic)[i];
            // auto cov3d_vec = op::reinterpret_cast_array_nd<6>($cov3d_vecs)[i];
            auto cam2world_R_T = $cam2world_R_np;
            auto cam2world_center = $cam2world_center_np;

            auto focal_xy = $focal_xy_np;
            auto principal_point = $principal_point_np;
            auto tan_fov = $tanfov_xy_np;
            auto point = op::reinterpret_cast_array_nd<3>($xyz)[i];
            point = point - cam2world_center;

            auto point_cam = cam2world_R_T.op<op::mv_rowmajor>(point);
            auto cov2d_arr = op::reinterpret_cast_array_nd<3>($cov2d_vecs)[i];
            
            auto dpoint_cam = CameraOps::pos_cam_to_uv_no_distort_grad(uv_grad, 0.0f, point_cam, principal_point, focal_xy);
            auto dcov2d = Gaussian3D::gaussian_2d_inverse_and_det_grad(conic_grad, cov2d_arr);
            auto scale = op::reinterpret_cast_array_nd<3>($scales)[i];
            auto quat = op::reinterpret_cast_array_nd<4>($quat_xyzw)[i];
            """)
            if fused_scale_act:
                code_prep.raw(f"""
                auto scale_act = scale.op<op::{fused_scale_act[0]}>();
                """)
            else:
                code_prep.raw(f"""
                auto scale_act = scale;
                """)
            if fused_q_act:
                code_prep.raw(f"""
                auto quat_act = quat.op<op::normalize>();
                """)
            else:
                code_prep.raw(f"""
                auto quat_act = quat;
                """)
            if self.cfg.recalc_cov3d_in_bwd:
                code_prep.raw(f"""
                auto cov3d_vec = Gaussian3D::scale_quat_to_cov3d(scale_act, quat_act);
                """)
            else:
                code_prep.raw(f"""
                auto cov3d_vec = op::reinterpret_cast_array_nd<6>($cov3d_vecs)[i];
                """)
            code_prep.raw(f"""
            auto proj_grad_res = Gaussian3D::project_gaussian_to_2d_grad<float, 3>(dcov2d, point_cam, focal_xy, tan_fov, cam2world_R_T, cov3d_vec);
            dpoint_cam += std::get<0>(proj_grad_res);

            auto dxyz = cam2world_R_T.op<op::mv_colmajor>(dpoint_cam);

            """)
            code_prep.raw(f"""
            auto cov3d_grad_res = Gaussian3D::scale_quat_to_cov3d_grad(std::get<1>(proj_grad_res), scale_act, quat_act);
            auto dscale = std::get<0>(cov3d_grad_res);
            auto dquat = std::get<1>(cov3d_grad_res);
            """)
            if fused_scale_act:
                code_prep.raw(f"""
                dscale = dscale.op<op::{fused_scale_act[1]}>(scale);
                """)
            if fused_q_act:
                code_prep.raw(f"""
                dquat = dquat.op<op::{fused_q_act[1]}>(quat);
                """)
            if fused_opacity_act:
                code_prep.raw(f"""
                tv::array<float, 1> opacity_val{{$opacity[i]}};
                tv::array<float, 1> dopacity_val{{$dopacity[i]}};
                dopacity_val = dopacity_val.op<op::{fused_opacity_act[1]}>(opacity_val);
                $dopacity[i] = dopacity_val[0];
                """)
            code_prep.raw(f"""
            op::reinterpret_cast_array_nd<3>($dscale_res)[i] = dscale;
            op::reinterpret_cast_array_nd<4>($dquat_res)[i] = dquat;

            auto normed_dir = (point).op<op::normalize>();
            auto sh_ptr = op::reinterpret_cast_array_nd<3>($color_sh) + i * {(degree + 1) * (degree + 1) - has_color_base};
            auto dsh_ptr = op::reinterpret_cast_array_nd<3>($dcolor_sh_res) + i * {(degree + 1) * (degree + 1) - has_color_base};
            auto color_grad = op::reinterpret_cast_array_nd<3>($dcolor)[i];

            uint8_t nn_packed = $sh_to_rgb_ne_0[i];
            bool rgb0_ne_0 = nn_packed & 0x1;
            bool rgb1_ne_0 = nn_packed & 0x2;
            bool rgb2_ne_0 = nn_packed & 0x4;
            color_grad[0] *= rgb0_ne_0 ? 0.0f : 1.0f;
            color_grad[1] *= rgb1_ne_0 ? 0.0f : 1.0f;
            color_grad[2] *= rgb2_ne_0 ? 0.0f : 1.0f;
            

            """)
            if has_color_base:
                code_prep.raw(f"""
                auto dsh_base_ptr = op::reinterpret_cast_array_nd<3>($dcolor_base_sh_res) + i;
                auto dnormed_dir = Gaussian3D::sh_dir_to_rgb_grad<{cur_degree}>(color_grad, dsh_ptr,
                    normed_dir, sh_ptr, dsh_base_ptr);
                """)
            else:
                code_prep.raw(f"""
                auto dnormed_dir = Gaussian3D::sh_dir_to_rgb_grad<{cur_degree}>(color_grad, dsh_ptr,
                    normed_dir, sh_ptr);
                """)
            code_prep.raw(f"""
            dxyz += dnormed_dir.op<op::normalize_grad>(point);
            op::reinterpret_cast_array_nd<3>($dxyz_res)[i] = dxyz;

            """)
            INLINER.kernel_1d(
                prep_kernel_name, num, 0, code_prep)
        if return_uv_grad:
            return grad_model, duv
        return grad_model, None


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
        model_cls: type[GaussianModelBase],
        # cameras
        cameras,
        # options
        training,
        gaussian_cfg: GaussianSplatConfig,
        uv_grad_holder=None,
    ):
        op = GaussianSplatForward(gaussian_cfg)
        enable_verbose = False
        with tv.measure_and_print("FWD-all-torch-1", enable=enable_verbose):

            model = model_cls(
                xyz=xyz,
                color_sh=color_sh,
                color_sh_base=color_sh_base,
                scale=scale,
                quaternion_xyzw=quaternion_xyzw,
                opacity=opacity,
                cur_sh_degree=cur_sh_degree,
                act_applied=True,
            )
        with tv.measure_and_print("FWD-all-torch", enable=enable_verbose):
            out, fwd_ctx = op.forward(model, cameras, training=training)
        with tv.measure_and_print("FWD-all-torch-2", enable=enable_verbose):

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
                    # outputs
                    out.n_contrib,
                    out.color,
                    out.depth,
                    out.custom_features,
                    out.T,
                )
                ctx.image_shape_wh = fwd_ctx.image_shape_wh

        ctx.op = op
        ctx.cams = cameras
        ctx.cur_sh_degree = cur_sh_degree
        ctx.return_uv_grad = uv_grad_holder is not None
        ctx.model_cls = model_cls
        # torch autograd func only support tuple output
        out_items = (out.color, out.depth,
                     out.custom_features, out.T,
                     out.n_contrib,
                     out.radii)
        return out_items

    @staticmethod
    @once_differentiable
    def backward(ctx, drgb, ddepth, dcustom_features, dT, dn_contrib, dradii):
        if len(ctx.saved_tensors) == 0:
            return (None, ) * 12
        op: GaussianSplatForward = ctx.op

        with tv.measure_and_print("BWD-all-torch", enable=False):
            out = GaussianSplatOutput(
                n_contrib=ctx.saved_tensors[-5],
                color=ctx.saved_tensors[-4],
                depth=ctx.saved_tensors[-3],
                custom_features=ctx.saved_tensors[-2],
                T=ctx.saved_tensors[-1],
            )
            fwd_ctx = GaussianSplatForwardContext(
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
                act_applied=True,
                cur_sh_degree=ctx.cur_sh_degree,
            )
            if op.cfg.transmittance_is_double:
                dT = dT.float()
            gradient = GaussianSplatGradients(
                drgb=drgb,
                ddepth=ddepth,
                dcustom_features=dcustom_features,
                dT=dT # .float(),
            )
            # if op.cfg.verbose:
            #     print("Backward Used Memory", out.get_all_torch_tensor_byte_size() + 
            #         model.get_all_torch_tensor_byte_size() +
            #             fwd_ctx.get_all_torch_tensor_byte_size() +
            #             gradient.get_all_torch_tensor_byte_size())
            #     print("Out Size", out.get_all_torch_tensor_byte_size())
            #     print("Model Size", model.get_all_torch_tensor_byte_size())
            #     print("FwdCtx Size", fwd_ctx.get_all_torch_tensor_byte_size())
            #     print("Gradient Size", gradient.get_all_torch_tensor_byte_size())
            cams: list[BasicPinholeCamera] = ctx.cams
        grad_out, duv = op.backward(model, cams, fwd_ctx, out, gradient, ctx.return_uv_grad)
        return grad_out.xyz, grad_out.color_sh, grad_out.color_sh_base, grad_out.scale, grad_out.quaternion_xyzw, grad_out.opacity, None, None, None, None, None, duv


def rasterize_gaussians(
        model: GaussianModelBase,
        cameras: list[BasicPinholeCamera],
        training=False,
        gaussian_cfg: GaussianSplatConfig = GaussianSplatConfig(),
        background_tensor: torch.Tensor | None = None,
        uv_grad_holder: torch.Tensor | None = None,
):
    if not training:
        assert background_tensor is None, "background_tensor is only used in training mode"
    model = model.create_model_with_act() # apply activation
    res = _RasterizeGaussians.apply(
        model.xyz,
        model.color_sh,
        model.color_sh_base,
        model.scale,
        model.quaternion_xyzw,
        model.opacity,
        model.cur_sh_degree,
        type(model),
        cameras,
        training,
        gaussian_cfg,
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
    if background_tensor is not None:
        out.color = out.color + out.T * background_tensor
    else:
        if training:
            bg_in_cfg_device = gaussian_cfg.get_bg_color_device(model.xyz.device)
            out.color = out.color + out.T * bg_in_cfg_device.view(3, 1, 1)
    return out

import time
from d3sim.core import dataclass_dispatch as dataclasses
from d3sim.ops.d3gs.base import GaussianModelBase
from d3sim.csrc.inliner import INLINER
import numpy as np
import math 
import torch 
import pccm 
from cumm.gemm.codeops import div_up
from d3sim.constants import IsAppleSiliconMacOs
import cumm.tensorview as tv

def fov2focal(fov: float, length: int):
    return float(length) / (2 * math.tan(fov / 2))

def focal2fov(focal: float, length: int):
    return 2*math.atan(float(length)/(2*focal))

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class GaussianSplatConfig:
    tile_size: tuple[int, int] = (16, 16)
    eps: float = 1e-6
    cov2d_radii_eigen_eps: float = 0.1
    projected_clamp_factor: float = 1.3
    gaussian_std_sigma: float = 3.0
    gaussian_lowpass_filter: float = 0.3
    alpha_eps: float = 1.0 / 255.0
    transmittance_eps: float = 0.0001

    bg_color: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros((3), np.float32))

class GaussianSplatForward:
    def __init__(self, cfg: GaussianSplatConfig) -> None:
        self.cfg = cfg

    def forward(self, model: GaussianModelBase, intrinsic: np.ndarray,
                   cam2world: np.ndarray, image_shape_wh: tuple[int, int],
                   axis_front_u_v: tuple[int, int, int], scale_global: float = 1.0):
        num = len(model)

        width = image_shape_wh[0]
        height = image_shape_wh[1]

        focal_x = intrinsic[0, 0]
        focal_y = intrinsic[1, 1]

        fov_x = focal2fov(focal_x, width)
        fov_y = focal2fov(focal_y, height)
        print(fov_x, fov_y)
        tanfov_x = math.tan(fov_x / 2)
        tanfov_y = math.tan(fov_y / 2)
        resolution_wh_np = np.array([width, height], np.int32)

        focal_xy_np = np.array([focal_x, focal_y], np.float32)
        principal_point_np = np.array([intrinsic[0, 2], intrinsic[1, 2]], np.float32) / resolution_wh_np.astype(np.float32)
        tanfov_xy_np = np.array([tanfov_x, tanfov_y], np.float32)
        cam2world_T_np = np.ascontiguousarray(cam2world[:3].T)
        tile_num_x = div_up(width, self.cfg.tile_size[0])
        tile_num_y = div_up(height, self.cfg.tile_size[1])
        block_size = self.cfg.tile_size[0] * self.cfg.tile_size[1]
        tile_num_xy_np = np.array([tile_num_x, tile_num_y], np.int32)
        xyz = model.xyz_act
        # quat should be normed in user-defined quaternion_xyzw_act.
        quat_xyzw = model.quaternion_xyzw_act
        scales = model.scale_act
        opacity = model.opacity_act
        color_sh = model.color_sh_act
        degree = model.color_sh_degree
        depths = torch.empty(num, dtype=torch.float32, device=xyz.device)
        radii = torch.empty(num, dtype=torch.int32, device=xyz.device)
        uvs = torch.empty([num, 2], dtype=torch.float32, device=xyz.device)
        conic_opacity = torch.empty([num, 4], dtype=torch.float32, device=xyz.device)
        tiles_touched = torch.empty(num, dtype=torch.int32, device=xyz.device)
        rgb_gaussian = torch.empty([num, 3], dtype=torch.float32, device=xyz.device)
        # debug_cov3d = torch.empty([num, 6], dtype=torch.float32, device=xyz.device)
        # debug_cov2d = torch.empty([num, 3], dtype=torch.float32, device=xyz.device)

        # debug_ten = torch.empty(num, dtype=torch.float32, device=xyz.device)
        custom_feat = model.custom_features
        num_custom_feat = 0 if custom_feat is None else custom_feat.shape[1]
        t1 = time.time()
        INLINER.kernel_1d(f"gs3d_preprocess_{axis_front_u_v}_{self.cfg.tile_size}_{degree}", num, 0, f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<float>;
        auto cam2world_T = $cam2world_T_np;
        auto focal_xy = $focal_xy_np;
        auto principal_point = $principal_point_np;
        auto tan_fov = $tanfov_xy_np;
        auto resolution_wh = $resolution_wh_np;
        auto point = op::reinterpret_cast_array_nd<3>($xyz)[i];
        auto scale = op::reinterpret_cast_array_nd<3>($scales)[i];
        auto quat = op::reinterpret_cast_array_nd<4>($quat_xyzw)[i];

        auto point_cam = op::slice<0, 3>(cam2world_T).op<op::mv_rowmajor>(point - cam2world_T[3]);
        auto uv_unified_and_z = CameraOps::pos_cam_to_uv_no_distort<{axis_front_u_v[0]}, {axis_front_u_v[1]}, {axis_front_u_v[2]}>(
            point_cam, principal_point, focal_xy / resolution_wh.cast<float>());
        auto uv = std::get<0>(uv_unified_and_z) * resolution_wh.cast<float>() + 0.5f;
        auto z_in_cam = std::get<1>(uv_unified_and_z);
        auto cov3d_vec = Gaussian3D::scale_quat_to_cov3d(scale, $scale_global, quat);
        auto cov2d_vec = Gaussian3D::project_gaussian_to_2d(point_cam, focal_xy, tan_fov, cam2world_T, cov3d_vec, $(self.cfg.projected_clamp_factor));
        cov2d_vec[0] += $(self.cfg.gaussian_lowpass_filter);
        cov2d_vec[2] += $(self.cfg.gaussian_lowpass_filter);
        
        auto cov2d_inv_and_det = Gaussian3D::gaussian_2d_inverse_and_det(cov2d_vec, $(self.cfg.eps));
        auto cov2d_inv = std::get<0>(cov2d_inv_and_det);
        auto det = std::get<1>(cov2d_inv_and_det);
        auto radii_fp = math_op_t::ceil(Gaussian3D::get_gaussian_2d_ellipse(cov2d_vec, det, 
            $(self.cfg.cov2d_radii_eigen_eps), $(self.cfg.gaussian_std_sigma)));
        // op::reinterpret_cast_array_nd<6>($debug_cov3d)[i] = cov3d_vec;
        // op::reinterpret_cast_array_nd<3>($debug_cov2d)[i] = cov2d_vec;
        constexpr auto tile_size_xy = tv::array<int, 2>{{{self.cfg.tile_size[0]}, {self.cfg.tile_size[1]}}};
        auto tile_num_xy = tv::div_up(resolution_wh, tile_size_xy);
        auto tile_size_xy_float = tile_size_xy.cast<float>();
        auto gaussian_rect_min = ((uv - radii_fp) / tile_size_xy_float).cast<int>().op<op::clamp>(0, tile_num_xy);
        auto gaussian_rect_max = ((uv + radii_fp + tile_size_xy_float - 1) / tile_size_xy_float).cast<int>().op<op::clamp>(0, tile_num_xy);
        bool is_empty = (det == 0 || z_in_cam < 0.2);
        int rect_area = (gaussian_rect_max[0] - gaussian_rect_min[0]) * (gaussian_rect_max[1] - gaussian_rect_min[1]);
        $tiles_touched[i] = is_empty ? 0 : rect_area;
        $radii[i] = is_empty ? 0 : int(radii_fp);
        $depths[i] = z_in_cam;
        // $debug_ten[i] = tile_num_xy[0];

        op::reinterpret_cast_array_nd<2>($uvs)[i] = uv;
        tv::array<float, 4> conic_opacity_vec{{cov2d_inv[0], cov2d_inv[1], cov2d_inv[2], $opacity[i]}};
        op::reinterpret_cast_array_nd<4>($conic_opacity)[i] = conic_opacity_vec;
        auto sh_ptr = op::reinterpret_cast_array_nd<3>($color_sh) + i * {(degree + 1) * (degree + 1)};
        op::reinterpret_cast_array_nd<3>($rgb_gaussian)[i] = Gaussian3D::sh_dir_to_rgb<{degree}>(sh_ptr, point_cam.op<op::normalize>());

        """)
        torch.mps.synchronize()
        print("Preprocess", time.time() - t1)
        tiles_touched.cumsum_(0)
        num_rendered = int(tiles_touched[-1].item())
        if (num_rendered == 0):
            return
        keys_tile_idx_depth = torch.empty(num_rendered, dtype=torch.int64, device=xyz.device)
        gaussian_idx = torch.empty(num_rendered, dtype=torch.int32, device=xyz.device)
        INLINER.kernel_1d("prepare_sort_data", num, 0, f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<float>;

        auto radii_val = $radii[i];
        auto radii_fp = float(radii_val);
        constexpr auto tile_size_xy = tv::array<int, 2>{{{self.cfg.tile_size[0]}, {self.cfg.tile_size[1]}}};
        auto tile_num_xy = $tile_num_xy_np;
        auto tile_size_xy_float = tile_size_xy.cast<float>();

        if (radii_val > 0){{
            auto depth_uint = reinterpret_cast<const TV_METAL_DEVICE uint32_t*>($depths)[i];
            auto offset = i == 0 ? 0 : $tiles_touched[i - 1];
            auto uv = op::reinterpret_cast_alignedarray<2>($uvs)[i];
            auto gaussian_rect_min = ((uv - radii_fp) / tile_size_xy_float).cast<int>().op<op::clamp>(0, tile_num_xy);
            auto gaussian_rect_max = ((uv + radii_fp + tile_size_xy_float - 1) / tile_size_xy_float).cast<int>().op<op::clamp>(0, tile_num_xy);
            for (int y = gaussian_rect_min[1]; y < gaussian_rect_max[1]; ++y){{
                for (int x = gaussian_rect_min[0]; x < gaussian_rect_max[0]; ++x){{
                    uint64_t key = y * tile_num_xy[0] + x;
                    key <<= 32;
                    key |= depth_uint; // assume depth always > 0
                    $keys_tile_idx_depth[offset] = key;
                    $gaussian_idx[offset] = i;
                    ++offset;
                }}
            }}
        }}
        """)
        # print(debug_ten.min(), debug_ten.max(), debug_ten)

        # TODO use radix sort with trunated bits (faster) for cuda
        # TODO use 32bit sort, for 1920x1080 tile, the max tile idx is 8100, 13 bits
        # so we can use 18bit for depth.
        sorted_vals, indices = torch.sort(keys_tile_idx_depth)
        gaussian_idx_sorted = torch.gather(gaussian_idx, 0, indices)
        workload_ranges = torch.empty([tile_num_x * tile_num_y, 2], dtype=torch.int32, device=xyz.device)
        INLINER.kernel_1d("prepare_workload_range", num_rendered, 0, f"""
        auto key = $sorted_vals[i];
        uint32_t tile_idx = key >> 32;
        if (i == 0){{
            $workload_ranges[tile_idx * 2 + 0] = 0;
        }}else{{
            auto last_key = $sorted_vals[i - 1];
            uint32_t last_tile_idx = last_key >> 32;
            if (tile_idx != last_tile_idx){{
                $workload_ranges[last_tile_idx * 2 + 1] = i;
                $workload_ranges[tile_idx * 2 + 0] = i;
            }}
        }}
        if (i == $num_rendered - 1){{
            $workload_ranges[tile_idx * 2 + 1] = $num_rendered;
        }}
        """)

        final_T = torch.empty([height * width], dtype=torch.float32, device=xyz.device)
        n_contrib = torch.empty([height * width], dtype=torch.int32, device=xyz.device)
        out_color = torch.empty([3, height, width], dtype=torch.float32, device=xyz.device)
        out_custom_feat = torch.empty([num_custom_feat, height, width], dtype=torch.float32, device=xyz.device)

        code_render_fwd = pccm.code()
        code_render_fwd.raw(f"""
        namespace op = tv::arrayops;
        using math_op_t = tv::arrayops::MathScalarOp<float>;
        """)
        if IsAppleSiliconMacOs:
            # metal nonuniform group, metal divide grid automatically.
            launch_param = tv.LaunchParam((width, height, 1), (self.cfg.tile_size[0], self.cfg.tile_size[1], 1))
            code_render_fwd.raw(f"""
            tv::array<uint32_t, 2> pixel_idx_xy{{threadPositionInGrid.x, threadPositionInGrid.y}};
            tv::array<uint32_t, 2> tile_idx_xy{{threadgroupPositionInGrid.x, threadgroupPositionInGrid.y}};
            bool inside = true;
            // keep in mind that metal only have 32KB shared memory
            threadgroup int collected_id[{self.cfg.tile_size[0] * self.cfg.tile_size[1]}];
            threadgroup tv::array<float, 2> collected_xy[{self.cfg.tile_size[0] * self.cfg.tile_size[1]}];
            threadgroup tv::array<float, 4> collected_conic_opacity[{self.cfg.tile_size[0] * self.cfg.tile_size[1]}];
            threadgroup int num_done_shared[32];
            uint thread_rank = threadPositionInThreadgroup.y * {self.cfg.tile_size[0]} + threadPositionInThreadgroup.x;
            """)
        else:
            launch_param = tv.LaunchParam((tile_num_x, tile_num_y, 1), (self.cfg.tile_size[0], self.cfg.tile_size[1], 1))
            code_render_fwd.raw(f"""
            tv::array<uint32_t, 2> pixel_idx_xy{{blockIdx.x * {self.cfg.tile_size[0]} + threadIdx.x, blockIdx.y * {self.cfg.tile_size[1]} + threadIdx.y}};
            tv::array<uint32_t, 2> tile_idx_xy{{blockIdx.x, blockIdx.y}};
            bool inside = pixel_idx_xy[0] < $width && pixel_idx_xy[1] < $height;
            __shared__ int collected_id[{self.cfg.tile_size[0] * self.cfg.tile_size[1]}];
            __shared__ tv::alignedarray<float, 2> collected_xy[{self.cfg.tile_size[0] * self.cfg.tile_size[1]}];
            __shared__ tv::alignedarray<float, 4> collected_conic_opacity[{self.cfg.tile_size[0] * self.cfg.tile_size[1]}];
            uint32_t thread_rank = threadIdx.y * {self.cfg.tile_size[0]} + threadIdx.x;
            """)
        code_render_fwd.raw(f"""
        uint32_t pixel_id = $width * pixel_idx_xy[1] + pixel_idx_xy[0];
        auto pixel_idx_xy_fp = pixel_idx_xy.cast<float>();

        // Check if this thread is associated with a valid pixel or outside.
        // Done threads can help with fetching, but don't rasterize
        bool done = !inside;

        auto range = op::reinterpret_cast_array_nd<2>($workload_ranges)[tile_idx_xy[1] * $tile_num_x + tile_idx_xy[0]];
        int rounds = tv::div_up(range[1] - range[0], {self.cfg.tile_size[0] * self.cfg.tile_size[1]});
        int toDo = range[1] - range[0];

        tv::array<float, 3> rgb{{}};
        float T = 1.0f;
        uint32_t contributor = 0;
        uint32_t last_contributor = 0;

        """)
        if num_custom_feat > 0:
            code_render_fwd.raw(f"""
            tv::array<float, {num_custom_feat}> custom_feature{{}};
            """)

        with code_render_fwd.for_(f"int i = 0; i < rounds; ++i, toDo -= {block_size}"):
            if IsAppleSiliconMacOs:
                code_render_fwd.raw(f"""
                int num_done_simd = metal::simd_sum(int(done));
                num_done_shared[tv::parallel::warp_index()] = num_done_simd;
                tv::parallel::block_sync();
                int num_done = 0;
                for (int j = 0; j < ({block_size}u / tv::parallel::warp_size()); ++j){{
                    num_done += num_done_shared[j];
                }}
                """)
            else:
                code_render_fwd.raw(f"""
                int num_done = __syncthreads_count(done);
                """)
            code_render_fwd.raw(f"""
            if (num_done == {block_size}){{
                break;
            }}
            int progress = i * {block_size} + thread_rank;

            if (range[0] + progress < range[1]){{
                int gaussian_id = $gaussian_idx_sorted[range[0] + progress];
                collected_id[thread_rank] = gaussian_id;
                collected_xy[thread_rank] = op::reinterpret_cast_alignedarray<2>($uvs)[gaussian_id];
                collected_conic_opacity[thread_rank] = op::reinterpret_cast_alignedarray<4>($conic_opacity)[gaussian_id];
            }}
            tv::parallel::block_sync();
            """)
            with code_render_fwd.for_(f"int j = 0; !done && j < std::min({block_size}, toDo); ++j"):
                code_render_fwd.raw(f"""
                ++contributor;
                auto uv = collected_xy[j];
                auto conic_opacity_vec = collected_conic_opacity[j];
                tv::array<float, 2> dist{{uv[0] - pixel_idx_xy_fp[0], uv[1] - pixel_idx_xy_fp[1]}};
                float power = (-0.5f * (conic_opacity_vec[0] * dist[0] * dist[0] + 
                                       conic_opacity_vec[2] * dist[1] * dist[1]) - 
                                        conic_opacity_vec[1] * dist[0] * dist[1]);
                if (power > 0){{
                    continue;
                }}
                float alpha = math_op_t::min(0.99f, conic_opacity_vec[3] * math_op_t::exp(power));
                if (alpha < {self.cfg.alpha_eps}){{
                    continue;
                }}
                float next_T = T * (1.0f - alpha);
                if (next_T < {self.cfg.transmittance_eps}){{
                    done = true;
                    break;
                }}
                float weight = alpha * T;
                T = next_T;
                last_contributor = contributor;
                auto gaussian_id = collected_id[j];
                auto rgb_val = op::reinterpret_cast_array_nd<3>($rgb_gaussian)[gaussian_id];
                rgb += weight * rgb_val;
                """) 
                if num_custom_feat > 0:
                    code_render_fwd.raw(f"""
                    for (int channel_idx = 0; channel_idx < {num_custom_feat}; ++channel_idx){{
                        custom_feature[channel_idx] += weight * $custom_feat[gaussian_id * {num_custom_feat} + channel_idx];
                    }}
                    """)
        code_render_fwd.raw(f"""
        auto bg_color_val = $(self.cfg.bg_color);
        if (inside)
        {{
            $final_T[pixel_id] = T;
            $n_contrib[pixel_id] = last_contributor;
            $out_color[0 * $height * $width + pixel_id] = rgb[0] + T * bg_color_val[0];
            $out_color[1 * $height * $width + pixel_id] = rgb[1] + T * bg_color_val[1];
            $out_color[2 * $height * $width + pixel_id] = rgb[2] + T * bg_color_val[2];
        }}
        """)
        if num_custom_feat:
            code_render_fwd.raw(f"""
            if (inside)
            {{
                for (int channel_idx = 0; channel_idx < {num_custom_feat}; ++channel_idx){{
                    $out_custom_feat[channel_idx * $height * $width + pixel_id] = custom_feature[channel_idx];
                }}
            }}
            """) 

        kernel_unique_name = (f"render_forward_{self.cfg.tile_size}_{num_custom_feat}_"
            f"{self.cfg.alpha_eps}_{self.cfg.transmittance_eps}")
        INLINER.kernel_raw(kernel_unique_name, launch_param, code_render_fwd)

        return final_T, n_contrib, out_color
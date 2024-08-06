from d3sim.data.scene_def.base import DistortType
from d3sim.csrc.inliner import INLINER
import torch
import numpy as np
import pccm
from tensorpc.flow import observe_function
from cumm.inliner import torch_tensor_to_tv
from cumm import tensorview as tv


@observe_function
def project_points_to_camera(
        points: torch.Tensor,
        cam_to_point: np.ndarray,
        intrinsic: np.ndarray,
        image_size_wh: tuple[int, int],
        distort: np.ndarray,
        distort_type: DistortType = DistortType.kOpencvPinhole,
        axes_front_u_v: tuple[int, int, int] = (2, 0, 1)) -> tuple[torch.Tensor, torch.Tensor]:
    focal_length = np.array([intrinsic[0, 0], intrinsic[1, 1]],
                            dtype=np.float32)
    resolution = np.array([image_size_wh[0], image_size_wh[1]],
                          dtype=np.float32)
    center = np.array([intrinsic[0, 2], intrinsic[1, 2]],
                      dtype=np.float32) / resolution
    
    if distort_type == DistortType.kOpencvPinhole:
        assert distort.shape[0] >= 4
        distort = distort.astype(np.float32)
    else:
        raise NotImplementedError
    cam_to_point_T = np.ascontiguousarray(cam_to_point[:3].T)
    if cam_to_point_T.dtype != np.float32:
        cam_to_point_T = cam_to_point_T.astype(np.float32)
    point_stride = points.stride(0)
    num = points.shape[0]
    res = torch.empty([points.shape[0], 3], dtype=points.dtype, device=points.device)
    res_mask = torch.empty([points.shape[0]], dtype=torch.uint8, device=points.device)
    INLINER.kernel_1d(
        f"project_points_to_cam_{distort.shape[0]}_{axes_front_u_v}", num, 0, f"""
    namespace op = tv::arrayops;
    auto point = op::reinterpret_cast_array_nd<float, 3>($points + i * $point_stride)[0];
    auto center_val = $center;
    auto focal_length_val = $focal_length;
    auto distort_val = $distort;
    auto c2w_T = $cam_to_point_T;
    auto uvd_res = CameraOps::pos_to_uv<{axes_front_u_v[0]}, {axes_front_u_v[1]}, {axes_front_u_v[2]}>(
        point, $resolution, center_val, 
        focal_length_val, distort_val.data(), 
        static_cast<CameraDefs::DistortType>($distort_type), c2w_T, {{}});
    auto uv = std::get<0>(uvd_res);
    auto z = std::get<1>(uvd_res);
    auto valid = std::get<2>(uvd_res);
    valid = true;
    valid &= (uv[0] > 0 && uv[1] > 0 && uv[0] < 1.0f && uv[1] < 1.0f);
    auto dir = (point - c2w_T[3]).op<op::normalize>();
    // assume front axis in camera space is pointing forward
    valid &= (dir.op<op::dot>(c2w_T[{axes_front_u_v[0]}]) >= 1e-4f);
    $res[i * 3] = uv[0];
    $res[i * 3 + 1] = uv[1];
    $res[i * 3 + 2] = z;
    $res_mask[i] = valid;
    """)

    return res, res_mask > 0

@observe_function
def get_depth_map_from_uvd(
    points_uvd: torch.Tensor,
        image_size_wh: tuple[int, int],
        depth_invalid_value: float | None = None,
        ):
    depth_map = torch.empty([image_size_wh[1], image_size_wh[0]], dtype=torch.float32, device=points_uvd.device)
    assert points_uvd.shape[1] == 3
    width = image_size_wh[0]
    height = image_size_wh[1]
    if depth_invalid_value is None:
        INLINER.kernel_1d(
            "fill_depth_with_max_fmax", depth_map.numel(), 0, f"""
        $depth_map[i] = std::numeric_limits<float>::max();
        """)
    else:
        INLINER.kernel_1d(
            "fill_depth_with_max_user_val", depth_map.numel(), 0, f"""
        $depth_map[i] = $depth_invalid_value;
        """)
    INLINER.kernel_1d(
        "fill_depth", points_uvd.shape[0], 0, f"""
    namespace op = tv::arrayops;
    using math_op_t = op::MathScalarOp<float>;
    auto point_uvd = op::reinterpret_cast_array_nd<float, 3>($points_uvd)[i];
    // uv is unified, convert to pixel
    auto u = point_uvd[0] * $width;
    auto v = point_uvd[1] * $height;
    auto depth = point_uvd[2];
    auto u_int = int(math_op_t::round(u));
    auto v_int = int(math_op_t::round(v));
    if (u_int >= 0 && u_int < $width && v_int >= 0 && v_int < $height){{
        tv::parallel::atomicMin($depth_map + v_int * $width + u_int, depth);
    }}
    """)
    return depth_map

@observe_function
def projected_point_uvd_to_jet_rgb(
        points_uvd: torch.Tensor,
        image_size_wh: tuple[int, int],
        val_range: tuple[float, float] = (0.5, 60.0),
        image_res: torch.Tensor | None = None,
        size: int = 2):
    if image_res is None:
        image_res = torch.zeros([image_size_wh[1], image_size_wh[0], 3], dtype=torch.uint8, device=points_uvd.device)
    else:
        image_res = image_res.clone()
    assert points_uvd.shape[1] == 3
    width = image_size_wh[0]
    height = image_size_wh[1]
    range_upper = val_range[1]
    range_lower = val_range[0]
    INLINER.kernel_1d(
        "projected_point_uvd_to_rgb", points_uvd.shape[0], 0, f"""
    namespace op = tv::arrayops;
    using math_op_t = op::MathScalarOp<float>;
    auto point_uvd = op::reinterpret_cast_array_nd<float, 3>($points_uvd)[i];
    // uv is unified, convert to pixel
    auto u = point_uvd[0] * $width;
    auto v = point_uvd[1] * $height;
    auto depth = point_uvd[2];
    auto u_int = int(math_op_t::round(u));
    auto v_int = int(math_op_t::round(v));
    // jet map
    auto rgb = VisUtils::value_to_jet_rgb(depth, $range_lower, $range_upper);
    // write to image map
    for (int j = 0; j < $size * 2 - 1; ++j){{
        for (int k = 0; k < $size * 2 - 1; ++k){{
            int u = u_int - $size + j;
            int v = v_int - $size + k;
            if (u >= 0 && u < $width && v >= 0 && v < $height){{
                $image_res[v * $width * 3 + u * 3 + 0] = rgb[0];
                $image_res[v * $width * 3 + u * 3 + 1] = rgb[1];
                $image_res[v * $width * 3 + u * 3 + 2] = rgb[2];
            }}
        }}
    }}
    """)
    return image_res

@observe_function
def depth_map_to_jet_rgb(
        depth_map: torch.Tensor,
        val_range: tuple[float, float] = (0.5, 60.0)):

    width = depth_map.shape[1]
    height = depth_map.shape[0]
    image_res = torch.zeros([height, width, 3], dtype=torch.uint8, device=depth_map.device)
    range_upper = val_range[1]
    range_lower = val_range[0]
    INLINER.kernel_1d(
        "depth_map_to_jet_rgb", height * width, 0, f"""
    namespace op = tv::arrayops;
    using math_op_t = op::MathScalarOp<float>;
    auto depth = $depth_map[i];
    auto rgb = VisUtils::value_to_jet_rgb(depth, $range_lower, $range_upper);
    // write to image map
    $image_res[i * 3 + 0] = rgb[0];
    $image_res[i * 3 + 1] = rgb[1];
    $image_res[i * 3 + 2] = rgb[2];
    """)
    return image_res
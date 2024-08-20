import time
from d3sim.constants import D3SIM_DEFAULT_DEVICE
from d3sim.csrc.inliner import INLINER 
import torch 
import numpy as np 
from d3sim.data.scene_def.base import DistortType
import pccm 
from tensorpc.flow import observe_function
from cumm.inliner import torch_tensor_to_tv
from cumm import tensorview as tv 

@observe_function
def get_ray_dirs(cam_to_target: np.ndarray,
        intrinsic: np.ndarray,
        image_size_wh: tuple[int, int],
        distort: np.ndarray,
        distort_type: DistortType = DistortType.kOpencvPinhole):
    width = image_size_wh[0]
    height = image_size_wh[1]
    focal_length = np.array([intrinsic[0, 0], intrinsic[1, 1]],
                            dtype=np.float32)
    center = np.array([intrinsic[0, 2], intrinsic[1, 2]],
                      dtype=np.float32)
    
    if distort_type == DistortType.kOpencvPinhole or distort_type == DistortType.kOpencvPinholeWaymo:
        assert distort.shape[0] >= 4
        distort = distort.astype(np.float32)
    elif distort_type == DistortType.kNone:
        distort = distort.astype(np.float32)
    else:
        raise NotImplementedError
    cam_to_target_T = np.ascontiguousarray(cam_to_target[:3].T)
    if cam_to_target_T.dtype != np.float32:
        cam_to_target_T = cam_to_target_T.astype(np.float32)
    res = torch.empty([image_size_wh[1], image_size_wh[0], 3], dtype=torch.float32, device=D3SIM_DEFAULT_DEVICE)
    INLINER.kernel_1d(
        f"get_ray_dirs_{distort.shape[0]}", image_size_wh[1] * image_size_wh[0], 0, f"""
    namespace op = tv::arrayops;

    int x_idx = i % $width;
    int y_idx = i / $width;

    tv::array<float, 2> uv{{
        float(x_idx) + 0.5f,
        float(y_idx) + 0.5f,
    }};
    auto center_val = $center;
    auto focal_length_val = $focal_length;
    auto distort_val = $distort;
    auto c2w_T = $cam_to_target_T;
    auto dir_in_cam = CameraOps::uv_to_dir(
        uv, center_val, 
        focal_length_val, distort_val.data(), 
        static_cast<CameraDefs::DistortType>($distort_type));
    auto dir_in_world = op::slice<0, 3>(c2w_T).op<op::mv_colmajor>(dir_in_cam);
    op::reinterpret_cast_array_nd<3>($res)[i] = dir_in_world.op<op::normalize>();
    """)

    return res


@observe_function
def create_cam_box3d_mask(cam_origin: np.ndarray, ray_dirs: torch.Tensor, image_size_wh: tuple[int, int], 
        box3d: torch.Tensor, box3d_scale: np.ndarray | None = None):
    # assume all params are inside same coordinate system
    assert box3d.ndim == 2 and box3d.shape[1] == 7 and box3d.dtype == torch.float32
    cam_origin = cam_origin.astype(np.float32)
    num_box = box3d.shape[0]
    if box3d_scale is None:
        box3d_scale = np.ones([3], np.float32)
    else:
        if box3d_scale.dtype != np.float32:
            box3d_scale = box3d_scale.astype(np.float32)
    box3d_heading_sin = torch.sin(box3d[:, 6:7])
    box3d_heading_cos = torch.cos(box3d[:, 6:7])
    box3d_sincos = torch.concatenate([box3d[:, :6], box3d_heading_sin, box3d_heading_cos], dim=1)
    res = torch.empty([image_size_wh[1], image_size_wh[0]], dtype=torch.uint8, device=ray_dirs.device)

    INLINER.kernel_1d(
        "create_cam_box3d_mask", image_size_wh[1] * image_size_wh[0], 0, f"""
    namespace op = tv::arrayops;

    auto box_ptr = op::reinterpret_cast_array_nd<8>($box3d_sincos);
    auto box_scale = $box3d_scale;
    auto origin = $cam_origin;
    auto dir = op::reinterpret_cast_array_nd<3>($ray_dirs)[i];
    namespace op = tv::arrayops;
    using math_op_t = op::MathScalarOp<float>;
    bool is_intersect = false;
    for (int j = 0; j < $num_box; ++j){{
        auto box = box_ptr[j];        
        auto center = op::slice<0, 3>(box);
        auto size = op::slice<3, 6>(box) * box_scale / 2.0f;
        auto sin_val = box[6];
        auto cos_val = box[7];
        // convert point to box space
        auto origin_local = origin - center;
        auto origin_x_in_box = origin_local[0] * cos_val + origin_local[1] * sin_val;
        auto origin_y_in_box = -origin_local[0] * sin_val + origin_local[1] * cos_val;
        auto origin_z_in_box = origin_local[2];
        auto dir_local = dir;
        auto dir_x_in_box = dir_local[0] * cos_val + dir_local[1] * sin_val;
        auto dir_y_in_box = -dir_local[0] * sin_val + dir_local[1] * cos_val;
        auto dir_z_in_box = dir_local[2];
        tv::array<float, 3> dir_in_box = {{dir_x_in_box, dir_y_in_box, dir_z_in_box}};

        auto aabb_inter_res = GeometryOps::ray_aabb_intersection(
            {{origin_x_in_box, origin_y_in_box, origin_z_in_box}},
            dir_in_box.op<op::normalize>(),
            -size, size);
        auto t0 = std::get<0>(aabb_inter_res)[0];
        is_intersect |= (std::get<1>(aabb_inter_res) && t0 > 0);
    }}
    $res[i] = is_intersect;
    """)
    return res > 0

import time
from d3sim.csrc.inliner import INLINER 
import torch 
import numpy as np 
import pccm 
from tensorpc.flow import observe_function

@observe_function
def range_image_to_point_cloud(ri: torch.Tensor, extrinsic: torch.Tensor, inclination: torch.Tensor, 
         pixel_pose: torch.Tensor | None = None, frame_pose: torch.Tensor | None = None,
         pixel_pose_transform: np.ndarray | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes range image polar coordinates.

    Args:
        range_image: [B, H, W] tensor. Lidar range images.
        extrinsic: [B, 4, 4] tensor. Lidar extrinsic.
        inclination: [B, H] / [B, 2] tensor. Inclination for each row of the range image.
            0-th entry corresponds to the 0-th row of the range image.
        pixel_pose: [B, H, W, 6] float tensor. If not None, it sets pose for each
            range image pixel.
    Returns:
        range_image_polar: [B, H, W, 3] polar coordinates.
    """

    # ri: [B, H, W, 4]
    if ri.ndim == 3:
        ri = ri.unsqueeze(0)
    batch_size = ri.shape[0]
    height = ri.shape[-3]
    width = ri.shape[-2]
    channel = ri.shape[-1]
    assert ri.dtype == torch.float32
    if extrinsic.dtype != torch.float32:
        extrinsic = extrinsic.to(torch.float32)
    if frame_pose is not None and frame_pose.dtype != torch.float32:
        frame_pose = frame_pose.to(torch.float32)

    assert inclination.dtype == torch.float32
    inclination = inclination.reshape(batch_size, -1)
    num_element = batch_size * height * width
    num_element_per_batch = height * width

    res = torch.empty([batch_size, height, width, 3], dtype=torch.float32, device=ri.device)
    valid_mask = ri[..., 0] > 0
    code = pccm.code()
    code.raw(f"""
    namespace op = tv::arrayops;
    using math_op_t = op::MathScalarOp<float>;
    int batch_idx = i / $num_element_per_batch;
    int pixel_idx = i % $num_element_per_batch;
    int y_idx = pixel_idx / $width;
    int x_idx = pixel_idx % $width;
    auto ri_val = $ri[i * $channel + 0];

    auto ext_matrix = op::reinterpret_cast_array_nd<float, 4, 4>($extrinsic)[batch_idx];
    auto az_correction = math_op_t::atan2(ext_matrix[1][0], ext_matrix[0][0]);
    auto ratios = (($width - x_idx) - 0.5f) / float($width);
    auto azimuth = (ratios * 2 - 1) * M_PI - az_correction;
    """)
    if inclination.shape[1] == height:
        code.raw(f"""
        auto inclination_val = $inclination[batch_idx * $height + y_idx];

        """)
    else:
        code.raw(f"""
        auto inclination_min_val = $inclination[batch_idx * 2 + 0];
        auto inclination_max_val = $inclination[batch_idx * 2 + 1];
        auto inclination_val = (0.5 + y_idx) / $height * (inclination_max_val - inclination_min_val) + inclination_min_val;

        """)

    code.raw(f"""
    auto cos_azimuth = math_op_t::cos(azimuth);
    auto sin_azimuth = math_op_t::sin(azimuth);
    auto cos_incl = math_op_t::cos(inclination_val);
    auto sin_incl = math_op_t::sin(inclination_val);

    float x = cos_azimuth * cos_incl * ri_val;
    float y = sin_azimuth * cos_incl * ri_val;
    float z = sin_incl * ri_val;
    tv::array<float, 3> xyz = {{x, y, z}};
    // to vehicle frame
    """)
    if pixel_pose is not None:
        assert frame_pose is not None 
        assert pixel_pose_transform is not None 
        if pixel_pose_transform.dtype != np.float32:
            pixel_pose_transform = pixel_pose_transform.astype(np.float32)
        code.raw(f"""
        auto pixel_pose_transform_mat = $pixel_pose_transform; // captured array in metal is constant, which doesn't well supported, so convert it to thread
        auto pixel_pose_6 = op::reinterpret_cast_array_nd<float, 6>($pixel_pose)[i];
        auto pixel_pose_rot = RotationMath::euler_to_rotmat_zyx(pixel_pose_6[0], pixel_pose_6[1], pixel_pose_6[2]);
        tv::array<float, 3> pixel_pose_xyz = {{pixel_pose_6[3], pixel_pose_6[4], pixel_pose_6[5]}};
        auto pixel_pose_mat = pixel_pose_rot.op<op::transform_matrix>(pixel_pose_xyz);
        auto frame_pose_mat = op::reinterpret_cast_array_nd<float, 4, 4>($frame_pose)[batch_idx];
        auto final_mat = frame_pose_mat.op<op::inverse>().op<op::mm_ttt>(pixel_pose_transform_mat.op<op::mm_ttt>(pixel_pose_mat.op<op::mm_ttt>(ext_matrix)));        
        """)
    else:
        code.raw(f"""
        auto final_mat = ext_matrix;
        """)
    code.raw(f"""
    auto xyz_res = xyz.op<op::transform_3d>(final_mat);
    op::reinterpret_cast_array_nd<float, 3>($res)[i] = xyz_res;
    """)
    name = f"waymo_range_image_to_point_cloud_{pixel_pose is None}_{inclination.shape[1] == height}"
    INLINER.kernel_1d(name, num_element, 0, code)
    return res.reshape(-1, 3), valid_mask.reshape(-1)

@observe_function
def range_image_to_point_cloud_v2(ri: torch.Tensor, extrinsic: np.ndarray, inclination: torch.Tensor, 
         pixel_pose: torch.Tensor | None = None, frame_pose: np.ndarray | None = None,
         pixel_pose_transform: np.ndarray | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes range image polar coordinates.

    Args:
        range_image: [B, H, W] tensor. Lidar range images.
        extrinsic: [14, 4] array. Lidar extrinsic.
        inclination: [H] / [2] tensor. Inclination for each row of the range image.
            0-th entry corresponds to the 0-th row of the range image.
        pixel_pose: [H, W, 6] float tensor. If not None, it sets pose for each
            range image pixel.
    Returns:
        range_image_polar: [B, H, W, 3] polar coordinates.
    """
    # ri: [B, H, W, 4]
    if ri.ndim == 3:
        ri = ri.unsqueeze(0)
    batch_size = ri.shape[0]
    height = ri.shape[-3]
    width = ri.shape[-2]
    channel = ri.shape[-1]
    assert ri.dtype == torch.float32
    if extrinsic.dtype != np.float32:
        extrinsic = extrinsic.astype(np.float32)
    if frame_pose is not None and frame_pose.dtype != np.float32:
        frame_pose = frame_pose.astype(np.float32)

    assert inclination.dtype == torch.float32
    inclination = inclination.reshape(-1)
    num_element = batch_size * height * width
    num_element_per_batch = height * width
    res = torch.empty([batch_size, height, width, 3], dtype=torch.float32, device=ri.device)
    valid_mask = ri[..., 0] > 0
    code = pccm.code()
    code.raw(f"""
    namespace op = tv::arrayops;
    using math_op_t = op::MathScalarOp<float>;
    int batch_idx = i / $num_element_per_batch;
    int pixel_idx = i % $num_element_per_batch;
    int y_idx = pixel_idx / $width;
    int x_idx = pixel_idx % $width;
    auto ri_val = $ri[i * $channel + 0];
    auto ext_matrix = $extrinsic;
    auto az_correction = math_op_t::atan2(ext_matrix[1][0], ext_matrix[0][0]);
    auto ratios = (($width - x_idx) - 0.5f) / float($width);
    auto azimuth = (ratios * 2 - 1) * M_PI - az_correction;
    """)
    if inclination.shape[0] == height:
        code.raw(f"""
        auto inclination_val = $inclination[y_idx];
        """)
    else:
        code.raw(f"""
        auto inclination_min_val = $inclination[0];
        auto inclination_max_val = $inclination[1];
        auto inclination_val = (0.5 + y_idx) / $height * (inclination_max_val - inclination_min_val) + inclination_min_val;
        """)

    code.raw(f"""
    auto cos_azimuth = math_op_t::cos(azimuth);
    auto sin_azimuth = math_op_t::sin(azimuth);
    auto cos_incl = math_op_t::cos(inclination_val);
    auto sin_incl = math_op_t::sin(inclination_val);

    float x = cos_azimuth * cos_incl * ri_val;
    float y = sin_azimuth * cos_incl * ri_val;
    float z = sin_incl * ri_val;
    tv::array<float, 3> xyz = {{x, y, z}};
    // to vehicle frame
    """)
    if pixel_pose is not None:
        assert frame_pose is not None 
        assert pixel_pose_transform is not None 
        if pixel_pose_transform.dtype != np.float32:
            pixel_pose_transform = pixel_pose_transform.astype(np.float32)
        code.raw(f"""
        auto pixel_pose_transform_mat = $pixel_pose_transform; // captured array in metal is constant, which doesn't well supported, so convert it to thread
        auto pixel_pose_6 = op::reinterpret_cast_array_nd<float, 6>($pixel_pose)[pixel_idx];
        auto pixel_pose_rot = RotationMath::euler_to_rotmat_zyx(pixel_pose_6[0], pixel_pose_6[1], pixel_pose_6[2]);
        tv::array<float, 3> pixel_pose_xyz = {{pixel_pose_6[3], pixel_pose_6[4], pixel_pose_6[5]}};
        auto pixel_pose_mat = pixel_pose_rot.op<op::transform_matrix>(pixel_pose_xyz);
        auto frame_pose_mat = $frame_pose;
        // auto frame_pose_mat = op::reinterpret_cast_array_nd<float, 4, 4>($frame_pose)[batch_idx];
        auto final_mat = frame_pose_mat.op<op::inverse>().op<op::mm_ttt>(pixel_pose_transform_mat.op<op::mm_ttt>(pixel_pose_mat.op<op::mm_ttt>(ext_matrix)));        
        """)
    else:
        code.raw(f"""
        auto final_mat = ext_matrix;
        """)
    code.raw(f"""
    auto xyz_res = xyz.op<op::transform_3d>(final_mat);
    op::reinterpret_cast_array_nd<float, 3>($res)[i] = xyz_res;
    """)
    name = f"waymo_range_image_to_point_cloud_v2_{pixel_pose is None}_{inclination.shape[0] == height}"
    INLINER.kernel_1d(name, num_element, 0, code)
    return res.reshape(-1, 3), valid_mask.reshape(-1)


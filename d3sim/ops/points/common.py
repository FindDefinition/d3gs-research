from d3sim.data.scene_def.base import DistortType
from d3sim.csrc.inliner import INLINER
import torch
import numpy as np
import pccm
from tensorpc.flow import observe_function
from cumm.inliner import torch_tensor_to_tv
from cumm import tensorview as tv


@observe_function
def points_in_which_3d_box(points: torch.Tensor, box3d: torch.Tensor, box3d_scale: np.ndarray | None = None, box_ids: torch.Tensor | None = None) -> torch.Tensor:
    point_stride = points.stride(0)
    num = points.shape[0]
    num_box = box3d.shape[0]
    assert box3d.ndim == 2 and box3d.shape[1] == 7 and box3d.dtype == torch.float32
    if box_ids is not None:
        assert box_ids.shape[0] == box3d.shape[0]
    if box3d_scale is None:
        box3d_scale = np.ones([3], np.float32)
    else:
        if box3d_scale.dtype != np.float32:
            box3d_scale = box3d_scale.astype(np.float32)
    box3d_heading_sin = torch.sin(box3d[:, 6:7])
    box3d_heading_cos = torch.cos(box3d[:, 6:7])
    box3d_sincos = torch.concatenate([box3d[:, :6], box3d_heading_sin, box3d_heading_cos], dim=1)
    res = torch.empty([points.shape[0]], dtype=torch.int32, device=points.device)

    INLINER.kernel_1d(
        f"create_image_bbox_mask_{box_ids is None}", num, 0, f"""
    namespace op = tv::arrayops;
    using math_op_t = op::MathScalarOp<float>;
    auto point = op::reinterpret_cast_array_nd<3>($points + i * $point_stride)[0];
    auto box_ptr = op::reinterpret_cast_array_nd<8>($box3d_sincos);
    auto box_scale = $box3d_scale;
    int index = -1;
    for (int j = 0; j < $num_box; ++j){{
        auto box = box_ptr[j];
        auto box_id = {"$box_ids[j]" if box_ids is not None else "j"};
        auto center = op::slice<0, 3>(box);
        auto size = op::slice<3, 6>(box) * box_scale / 2.0f; // TODO calc this outside
        auto sin_val = box[6];
        auto cos_val = box[7];
        // convert point to box space
        auto point_local = point - center;
        auto x_in_box = point_local[0] * cos_val + point_local[1] * sin_val;
        auto y_in_box = -point_local[0] * sin_val + point_local[1] * cos_val;
        auto z_in_box = point_local[2];
        bool in_box = (x_in_box >= -size[0] && x_in_box <= size[0] &&
                          y_in_box >= -size[1] && y_in_box <= size[1] &&
                          z_in_box >= -size[2] && z_in_box <= size[2]);
        index = in_box ? j : index;
    }}
    $res[i] = index;
    """)
    return res

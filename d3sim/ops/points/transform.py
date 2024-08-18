import time
from d3sim.csrc.inliner import INLINER 
import torch 
import numpy as np 
import pccm 
from tensorpc.flow import observe_function
from cumm.inliner import torch_tensor_to_tv
from cumm import tensorview as tv 


@observe_function
def transform_xyz(points: torch.Tensor, matrix: np.ndarray) -> torch.Tensor:
    points_stride = points.stride(0)
    if matrix.dtype != np.float32:
        matrix = matrix.astype(np.float32)
    assert points.dtype == torch.float32 
    assert points.ndim == 2
    res_points = torch.empty_like(points)
    with INLINER.enter_inliner_scope():
        INLINER.kernel_1d("transform_xyz", points.shape[0], 0, f"""
        namespace op = tv::arrayops;
        auto point_ptr = op::reinterpret_cast_array_nd<3>($points + $points_stride * i);
        auto p = point_ptr[0];
        auto mat = $matrix;
        tv::array<float, 3> res;
        res[0] = mat[0][0] * p[0] + mat[0][1] * p[1] + mat[0][2] * p[2] + mat[0][3];
        res[1] = mat[1][0] * p[0] + mat[1][1] * p[1] + mat[1][2] * p[2] + mat[1][3];
        res[2] = mat[2][0] * p[0] + mat[2][1] * p[1] + mat[2][2] * p[2] + mat[2][3];

        op::reinterpret_cast_array_nd<3>($res_points + $points_stride * i)[0] = res;
        
        """)
    return res_points
import numpy as np
import torch 
from cumm.gemm.layout_tensorop import rowmajor_inverse
from cumm.inliner import NVRTCInlineBuilder
from cumm.common import TensorView, TensorViewCPU, TensorViewNVRTC, TensorViewNVRTCHashKernel, TensorViewArrayLinalg, TensorViewNVRTCDev, EigenLib
from cumm import tensorview as tv
from d3sim.core.geodef import EulerIntrinsicOrder
from d3sim.core.ops.rotation import euler_to_rotmat_np
from d3sim.core.thtools import np_to_torch_dev
from d3sim.csrc.inliner import INLINER
from ccimport import compat 


def check_cov3d_grad(num_check: int = 5, delta: float = 1e-4):
    np.random.seed(50051)
    dtype = np.float64
    tv_dtype = torch.float64
    dtype_str = "double"
    if compat.IsAppleSiliconMacOs:
        dtype = np.float32 
        tv_dtype = torch.float32 
        dtype_str = "float"
    grad_scale_np = np.random.uniform(0.5, 1.5, size=[6]).astype(dtype)
    scales = np.random.uniform(0.5, 1.5, size=[num_check, 3]).astype(dtype)
    quats = np.random.uniform(-1, 1, size=[num_check, 4]).astype(dtype)

    scales_th = np_to_torch_dev(scales)
    quats_th = np_to_torch_dev(quats)
    inputs_th = torch.cat([scales_th, quats_th], dim=1)
    input_ndim = inputs_th.shape[1]
    my_val_tv = torch.zeros([num_check], dtype= inputs_th.dtype, device=inputs_th.device)
    ref_val_tv = torch.zeros([num_check], dtype= inputs_th.dtype, device=inputs_th.device)
    print("input_ndim", input_ndim)
    for i in range(inputs_th.shape[1]):
        inp_delta = np.zeros(inputs_th.shape[1:], dtype)
        index = i
        inp_delta.reshape(-1)[i] = delta 
        INLINER.kernel_1d(f"check_grad_op_{dtype_str}", num_check, 0, f"""
        namespace op = tv::arrayops;
        auto inp_ptr = op::reinterpret_cast_array_nd<{input_ndim}>($inputs_th);
        auto inp_arr = inp_ptr[i];
        auto grad_scale = $grad_scale_np;
        auto inp_delta_val = $inp_delta;
        
        auto inp_with_delta = inp_arr + inp_delta_val;
        auto res = Gaussian3D::scale_quat_to_cov3d(op::slice<0, 3>(inp_arr), op::slice<3, 7>(inp_arr)) * grad_scale;
        auto res_with_delta = Gaussian3D::scale_quat_to_cov3d(op::slice<0, 3>(inp_with_delta), op::slice<3, 7>(inp_with_delta)) * grad_scale;
        auto out_arr_with_delta_sum = op::reshape<-1>(res_with_delta - res).op<op::sum>(); 
        
        auto grad_res = Gaussian3D::scale_quat_to_cov3d_grad(grad_scale, op::slice<0, 3>(inp_arr), op::slice<3, 7>(inp_arr));
        auto grad_res_cat = op::concat(std::get<0>(grad_res), std::get<1>(grad_res));
        $my_val_tv[i] = op::reshape<-1>(grad_res_cat)[$index];
        $ref_val_tv[i] = out_arr_with_delta_sum / op::reshape<-1>(inp_delta_val)[$index];
        """)

        my_val = my_val_tv.cpu().numpy()
        ref_val = ref_val_tv.cpu().numpy()
        # currently only double can get high precision result,
        # apple silicon don't support double, so we just print here for now.
        print(my_val)
        print(ref_val)

def check_cov2d_grad(num_check: int = 10, delta: float = 1e-4):
    np.random.seed(50051)
    dtype = np.float64
    tv_dtype = torch.float64
    dtype_str = "double"
    if compat.IsAppleSiliconMacOs:
        dtype = np.float32 
        tv_dtype = torch.float32 
        dtype_str = "float"
    grad_scale_np = np.random.uniform(0.5, 1.5, size=[3]).astype(dtype)
    eulers = np.random.uniform(-1, 1, size=[num_check, 3]).astype(dtype)
    cam2world_Ts = np.zeros((num_check, 4, 3), np.float32)
    for j in range(num_check):
        cam2world_Ts[j, :3, :3] = euler_to_rotmat_np(*eulers[j], order=EulerIntrinsicOrder.ZYX)

    cov3d_vecs = np.random.uniform(-1, 1, size=[num_check, 6]).astype(dtype)
    mean_cameras = np.random.uniform(0.5, 1.5, size=[num_check, 3]).astype(dtype)
    cam2world_Ts_th = np_to_torch_dev(cam2world_Ts)
    cov3d_vecs_th = np_to_torch_dev(cov3d_vecs)
    mean_cameras_th = np_to_torch_dev(mean_cameras)

    inputs_th = torch.cat([ mean_cameras_th, cov3d_vecs_th], dim=1)
    input_ndim = inputs_th.shape[1]
    my_val_tv = torch.zeros([num_check], dtype= inputs_th.dtype, device=inputs_th.device)
    ref_val_tv = torch.zeros([num_check], dtype= inputs_th.dtype, device=inputs_th.device)
    print("input_ndim", input_ndim)
    for i in range(inputs_th.shape[1]):
        inp_delta = np.zeros(inputs_th.shape[1:], dtype)
        index = i
        inp_delta.reshape(-1)[i] = delta 
        INLINER.kernel_1d(f"check_grad_op_{dtype_str}", num_check, 0, f"""
        namespace op = tv::arrayops;
        auto inp_ptr = op::reinterpret_cast_array_nd<{input_ndim}>($inputs_th);
        auto inp_arr = inp_ptr[i];
        auto c2w_T_ptr = op::reinterpret_cast_array_nd<4, 3>($cam2world_Ts_th);

        auto c2w_T = c2w_T_ptr[i];
        auto grad_scale = $grad_scale_np;
        auto inp_delta_val = $inp_delta;
        
        auto inp_with_delta = inp_arr + inp_delta_val;
        auto res = Gaussian3D::project_gaussian_to_2d(op::slice<0, 3>(inp_arr), {{2.f, 2.f}}, {{1.f, 1.f}}, c2w_T, op::slice<3, 9>(inp_arr)) * grad_scale;
        auto res_with_delta = Gaussian3D::project_gaussian_to_2d(op::slice<0, 3>(inp_with_delta), {{2.f, 2.f}}, {{1.f, 1.f}}, c2w_T, op::slice<3, 9>(inp_with_delta)) * grad_scale;
        auto out_arr_with_delta_sum = op::reshape<-1>(res_with_delta - res).op<op::sum>(); 
        
        auto grad_res = Gaussian3D::project_gaussian_to_2d_grad(grad_scale, op::slice<0, 3>(inp_arr), {{2.f, 2.f}}, {{1.f, 1.f}}, c2w_T, op::slice<3, 9>(inp_arr));
        auto grad_res_cat = op::concat(std::get<0>(grad_res), std::get<1>(grad_res));
        $my_val_tv[i] = op::reshape<-1>(grad_res_cat)[$index];
        $ref_val_tv[i] = out_arr_with_delta_sum / op::reshape<-1>(inp_delta_val)[$index];
        """)

        my_val = my_val_tv.cpu().numpy()
        ref_val = ref_val_tv.cpu().numpy()
        print(f"------ {i} ------")
        # currently only double can get high precision result,
        # apple silicon don't support double, so we just print here for now.
        print(my_val)
        print(ref_val)

if __name__ == "__main__":
    check_cov2d_grad()
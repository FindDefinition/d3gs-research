from plyfile import PlyData
import torch 
from d3sim.constants import D3SIM_DEFAULT_DEVICE, PACKAGE_ROOT
from d3sim.ops.d3gs.base import GaussianModelBase, GaussianModelOrigin, GaussianModelOriginFused
import numpy as np
from d3sim.csrc.inliner import INLINER
from d3sim.csrc.gs3d import SHConstants
from d3sim.ops.d3gs.ops import simple_knn

def rgb_to_sh(rgb):
    return (rgb - 0.5) / SHConstants.C0

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def load_3dgs_origin_model(ply_path: str, fused: bool = True, split: bool = True):
    ply = PlyData.read(ply_path)
    vertex = ply["vertex"]
    positions = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32) # [:2000000]
    dc_vecs = []
    for i in range(3):
        dc_vecs.append(vertex[f"f_dc_{i}"])
    f_dc = np.stack(dc_vecs, axis=1)
    rest_vecs = []
    for i in range(45):
        rest_vecs.append(vertex[f"f_rest_{i}"])
    f_rest = np.stack(rest_vecs, axis=1).reshape(-1, 3, 15).transpose(0, 2, 1).reshape(f_dc.shape[0], -1)
    sh_coeffs = np.concatenate([f_dc, f_rest], axis=1).astype(np.float32) 
    sh_coeffs = np.ascontiguousarray(sh_coeffs)
    rot_vecs = []
    for i in range(4):
        rot_vecs.append(vertex[f"rot_{i}"])
    rots = np.stack(rot_vecs, axis=1).astype(np.float32)[:, [1, 2, 3, 0]]
    rots = np.ascontiguousarray(rots)
    scale_vecs = []
    for i in range(3):
        scale_vecs.append(vertex[f"scale_{i}"])
    scales = np.stack(scale_vecs, axis=1).astype(np.float32)
    opacity = vertex["opacity"].astype(np.float32)
    positions_th = torch.from_numpy(positions).to(D3SIM_DEFAULT_DEVICE)
    sh_coeffs_th = torch.from_numpy(sh_coeffs).to(D3SIM_DEFAULT_DEVICE).reshape(-1, 16, 3)
    if split:
        sh_coeffs_base_th = sh_coeffs_th[:, 0].contiguous()
        sh_coeffs_th = sh_coeffs_th[:, 1:].contiguous()
    else:
        sh_coeffs_base_th = None
    rots_th = torch.from_numpy(rots).to(D3SIM_DEFAULT_DEVICE)
    scales_th = torch.from_numpy(scales).to(D3SIM_DEFAULT_DEVICE)
    opacity_th = torch.from_numpy(opacity).to(D3SIM_DEFAULT_DEVICE)

    if fused:
        model = GaussianModelOriginFused(xyz=positions_th, quaternion_xyzw=rots_th, scale=scales_th, opacity=opacity_th, color_sh=sh_coeffs_th, color_sh_base=sh_coeffs_base_th)
    else:
        model = GaussianModelOrigin(xyz=positions_th, quaternion_xyzw=rots_th, scale=scales_th, opacity=opacity_th, color_sh=sh_coeffs_th, color_sh_base=sh_coeffs_base_th)
    
    
    model.cur_sh_degree = 3
    return model

def points_to_gaussian_init_scale(points: torch.Tensor):
    knn_res = simple_knn(points, 3)
    knn_res_mean2 = torch.square(knn_res).mean(1)
    dist2 = torch.clamp_min(knn_res_mean2, 0.0000001)       
    scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
    return scales

def init_original_3dgs_model(model: GaussianModelBase, points: np.ndarray, points_rgb: np.ndarray):
    assert model.xyz.shape[0] == points.shape[0]
    points_th = torch.from_numpy(points).to(D3SIM_DEFAULT_DEVICE)
    points_rgb_th = torch.from_numpy(points_rgb).to(D3SIM_DEFAULT_DEVICE)
    points_sh = rgb_to_sh(points_rgb_th)    
    model.color_sh[:, 0] = points_sh
    scales = points_to_gaussian_init_scale(points_th)
    model.scale[:] = scales
    quat_xyzw = torch.zeros((points_rgb_th.shape[0], 4), device="cuda")
    quat_xyzw[:, 3] = 1
    model.quaternion_xyzw[:] = quat_xyzw
    opacities = inverse_sigmoid(0.1 * torch.ones((points_th.shape[0], 1), dtype=torch.float32, device=D3SIM_DEFAULT_DEVICE))
    model.opacity[:] = opacities
    model.cur_sh_degree = 0
    model.xyz[:] = points_th
    return 
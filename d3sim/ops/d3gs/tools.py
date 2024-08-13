from plyfile import PlyData
import torch 
from d3sim.constants import D3SIM_DEFAULT_DEVICE, PACKAGE_ROOT
from d3sim.ops.d3gs.base import GaussianModelOrigin, GaussianModelOriginFused
import numpy as np

def load_3dgs_origin_model(ply_path: str, fused: bool = True):
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
    rots_th = torch.from_numpy(rots).to(D3SIM_DEFAULT_DEVICE)
    scales_th = torch.from_numpy(scales).to(D3SIM_DEFAULT_DEVICE)
    opacity_th = torch.from_numpy(opacity).to(D3SIM_DEFAULT_DEVICE)
    if fused:
        model = GaussianModelOriginFused(xyz=positions_th, quaternion_xyzw=rots_th, scale=scales_th, opacity=opacity_th, color_sh=sh_coeffs_th)
    else:
        model = GaussianModelOrigin(xyz=positions_th, quaternion_xyzw=rots_th, scale=scales_th, opacity=opacity_th, color_sh=sh_coeffs_th)
    model.cur_sh_degree = 3
    return model

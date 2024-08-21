import math
from plyfile import PlyData
import torch 
from d3sim.constants import D3SIM_DEFAULT_DEVICE, PACKAGE_ROOT, IsAppleSiliconMacOs
from d3sim.algos.d3gs.origin.model import GaussianModelOriginBase, GaussianModelOrigin, GaussianModelOriginFused
import numpy as np
from d3sim.csrc.inliner import INLINER
from d3sim.csrc.gs3d import SHConstants
from d3sim.algos.d3gs import config_def
from d3sim.algos.d3gs.ops import simple_knn
from d3sim.core.pytorch.optim import NgpAdam

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

def init_original_3dgs_model(model: GaussianModelOriginBase, points: np.ndarray, points_rgb: np.ndarray):
    with torch.no_grad():
        assert model.xyz.shape[0] == points.shape[0]
        points_th = torch.from_numpy(points.astype(np.float32)).to(D3SIM_DEFAULT_DEVICE).contiguous()
        points_rgb_th = torch.from_numpy(points_rgb.astype(np.float32)).to(D3SIM_DEFAULT_DEVICE)
        points_color_sh = rgb_to_sh(points_rgb_th)
        if model.color_sh_base is None:
            model.color_sh[:, 0] = points_color_sh
        else:
            model.color_sh_base[:] = points_color_sh
        scales = points_to_gaussian_init_scale(points_th)
        model.scale[:] = scales
        quat_xyzw = torch.zeros((points_rgb_th.shape[0], 4), device=D3SIM_DEFAULT_DEVICE)
        quat_xyzw[:, 3] = 1
        model.quaternion_xyzw[:] = quat_xyzw
        opacities = inverse_sigmoid(0.1 * torch.ones((points_th.shape[0], 1), dtype=torch.float32, device=D3SIM_DEFAULT_DEVICE))
        model.opacity[:] = opacities.reshape(-1)
        model.cur_sh_degree = 0
        model.xyz[:] = points_th
    return 

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def create_origin_3dgs_optimizer(model: GaussianModelOriginBase, optim_cfg: config_def.Optimizer, spatial_lr_scale: float):

    pg = [
        {'params': [model.xyz], 'lr': optim_cfg.position_lr_init * spatial_lr_scale, "name": "xyz"},
        {'params': [model.color_sh_base], 'lr': optim_cfg.feature_lr, "name": "color_sh_base"},
        {'params': [model.color_sh], 'lr': optim_cfg.feature_lr / 20.0, "name": "color_sh"},
        {'params': [model.opacity], 'lr': optim_cfg.opacity_lr, "name": "opacity"},
        {'params': [model.scale], 'lr': optim_cfg.scaling_lr, "name": "scale"},
        {'params': [model.quaternion_xyzw], 'lr': optim_cfg.rotation_lr, "name": "quaternion_xyzw"}
    ]
    optim = torch.optim.Adam(pg, lr=0.0, eps=1e-15)

    xyz_schedule_xyz = get_expon_lr_func(lr_init=optim_cfg.position_lr_init*spatial_lr_scale,
                                                    lr_final=optim_cfg.position_lr_final*spatial_lr_scale,
                                                    lr_delay_mult=optim_cfg.position_lr_delay_mult,
                                                    max_steps=optim_cfg.position_lr_max_steps)
    return optim, xyz_schedule_xyz

def create_origin_3dgs_optimizers(model: GaussianModelOriginBase, optim_cfg: config_def.Optimizer, batch_size: int, spatial_lr_scale: float, fused: bool = True):
    bs_scale = math.sqrt(batch_size)
    pgs = [
        {'params': [model.xyz], 'lr': optim_cfg.position_lr_init * spatial_lr_scale * bs_scale, "name": "xyz"},
        {'params': [model.color_sh_base], 'lr': optim_cfg.feature_lr * bs_scale, "name": "color_sh_base"},
        {'params': [model.color_sh], 'lr': optim_cfg.feature_lr * bs_scale / 20.0, "name": "color_sh"},
        {'params': [model.opacity], 'lr': optim_cfg.opacity_lr * bs_scale, "name": "opacity"},
        {'params': [model.scale], 'lr': optim_cfg.scaling_lr * bs_scale, "name": "scale"},
        {'params': [model.quaternion_xyzw], 'lr': optim_cfg.rotation_lr * bs_scale, "name": "quaternion_xyzw"}
    ]
    if IsAppleSiliconMacOs:
        fused = False
    optimizers = {
        pg["name"] :torch.optim.Adam(
            [{"params": pg["params"], "lr": pg["lr"], "name": pg["name"]}],
            fused=fused,
            eps=1e-15 / bs_scale,
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),

        )
        for pg in pgs
    }

    xyz_schedule_xyz = get_expon_lr_func(lr_init=optim_cfg.position_lr_init*spatial_lr_scale * bs_scale,
                                                    lr_final=optim_cfg.position_lr_final*spatial_lr_scale * bs_scale,
                                                    lr_delay_mult=optim_cfg.position_lr_delay_mult,
                                                    max_steps=optim_cfg.position_lr_max_steps)
    return optimizers, xyz_schedule_xyz
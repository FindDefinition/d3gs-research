
from plyfile import PlyData
import torch 
import numpy as np
from d3sim.constants import D3SIM_DEFAULT_DEVICE, PACKAGE_ROOT, IsAppleSiliconMacOs
from d3sim.algos.d3gs.pvg.model import PVGGaussianModel
def load_pvg_model(ckpt_path: str, split: bool = True):
    (model_params, first_iter) = torch.load(ckpt_path, map_location=D3SIM_DEFAULT_DEVICE)
    (active_sh_degree,
        _xyz,
        _features_dc,
        _features_rest,
        _scaling,
        _rotation,
        _opacity,
        _t,
        _scaling_t,
        _velocity,
        max_radii2D,
        xyz_gradient_accum,
        t_gradient_accum,
        denom,
        opt_dict,
        spatial_lr_scale,
        T,
        velocity_decay,
    ) = model_params
    # breakpoint()
    # wxyz to xyzw
    print("PVG Model Meta", T, velocity_decay)
    _rotation = _rotation[:, [1, 2, 3, 0]].contiguous()
    sh_coeffs = torch.cat([_features_dc, _features_rest], dim=1).contiguous()
    positions_th = _xyz.to(D3SIM_DEFAULT_DEVICE)
    sh_coeffs_th = sh_coeffs.to(D3SIM_DEFAULT_DEVICE).reshape(-1, 16, 3)
    if split:
        sh_coeffs_base_th = sh_coeffs_th[:, 0].contiguous()
        sh_coeffs_th = sh_coeffs_th[:, 1:].contiguous()
    else:
        sh_coeffs_base_th = None
    rots_th = _rotation.to(D3SIM_DEFAULT_DEVICE)
    scales_th = _scaling.to(D3SIM_DEFAULT_DEVICE)
    opacity_th = _opacity.to(D3SIM_DEFAULT_DEVICE).reshape(-1)
    t_th = _t.to(D3SIM_DEFAULT_DEVICE)
    scaling_t_th = _scaling_t.to(D3SIM_DEFAULT_DEVICE)
    velocity_th = _velocity.to(D3SIM_DEFAULT_DEVICE)
    model = PVGGaussianModel(xyz=positions_th, quaternion_xyzw=rots_th, scale=scales_th, opacity=opacity_th, color_sh=sh_coeffs_th, color_sh_base=sh_coeffs_base_th,
        t=t_th, scaling_t=scaling_t_th, velocity=velocity_th)
    
    model.cur_sh_degree = active_sh_degree
    return model


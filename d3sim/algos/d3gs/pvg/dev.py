

import torch
from d3sim.algos.d3gs.pvg.load_pvg import load_pvg_model
from d3sim.algos.d3gs.render import GaussianSplatOp, rasterize_gaussians_dynamic
from d3sim.constants import D3SIM_DEFAULT_DEVICE, IsAppleSiliconMacOs
from d3sim.algos.d3gs.config_def import GaussianSplatConfig

import numpy as np

from d3sim.data.scene_def.base import Pose, Resource
from d3sim.data.scene_def.camera import BasicPinholeCamera 

def main():
    if IsAppleSiliconMacOs:
        path = "/Users/yanyan/Downloads/PVG-chkpnt30000.pth"
        debug_inp_path = "/Users/yanyan/Downloads/debug_pvg.pth"
    else:
        path = "/root/Projects/3dgs/PVG/eval_output/waymo_reconstruction/0145050/chkpnt30000.pth"
        debug_inp_path = "/root/debug_pvg.pth"
    model = load_pvg_model(path)
    # breakpoint()
    op = GaussianSplatOp(GaussianSplatConfig(render_depth=True, render_rgba=True))

    (viewpoint_camera, bg_color_from_envmap, rendered_image_before, rendered_image, rendered_opacity, rendered_depth) = torch.load(debug_inp_path, map_location="cpu")

    c2w = viewpoint_camera["c2w"].cpu().numpy()
    fx = viewpoint_camera["fx"]
    fy = viewpoint_camera["fy"]
    cx = viewpoint_camera["cx"]
    cy = viewpoint_camera["cy"]
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    intrinsic_4x4 = np.eye(4, dtype=np.float32)
    intrinsic_4x4[:3, :3] = intrinsic
    width = viewpoint_camera["resolution"][0]
    height = viewpoint_camera["resolution"][1]
    timestamp = viewpoint_camera["timestamp"]
    cam = BasicPinholeCamera(id="", timestamp=0, pose=Pose(np.eye(4), c2w), 
        image_rc=Resource(base_uri="", loader_type=""), intrinsic=intrinsic_4x4, distortion=np.zeros(4, np.float32),
        image_shape_wh=(width, height), objects=[])
    
    res = rasterize_gaussians_dynamic(model, cam, op, prep_input_tensors={
        "batch_ts": torch.tensor([timestamp], dtype=torch.float32, device=D3SIM_DEFAULT_DEVICE),
    }, prep_inputs={
        "time_shift": 0,
    })
    out_color = res.color
    print(out_color.shape)

    if not op.is_nchw:
        # to nchw
        out_color = out_color.permute(0, 3, 1, 2)
    out_color = torch.concat([out_color[i] for i in range(out_color.shape[0])], dim=1)

    out_color_u8 = out_color.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    if op.is_rgba:
        out_color_u8 = out_color_u8[..., [2, 1, 0, 3]]
    else:
        out_color_u8 = out_color_u8[..., ::-1]
    # if res.depth is not None:
    #     depth = res.depth
    #     depth_rgb = depth_map_to_jet_rgb(depth, (0.2, 25.0)).cpu().numpy()
    #     out_color_u8 = np.concatenate([out_color_u8, depth_rgb], axis=0)

    # out_color_u8 = (out_color.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    import cv2 
    cv2.imwrite("test_pvg.png", out_color_u8)

    breakpoint()
    # rasterize_gaussians_dynamic



if __name__ == "__main__":
    main()
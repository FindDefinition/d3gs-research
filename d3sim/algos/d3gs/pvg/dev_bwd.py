

import torch
from d3sim.algos.d3gs.origin.data.scene.cameras import fov2focal
from d3sim.algos.d3gs.pvg.load_pvg import load_pvg_model
from d3sim.algos.d3gs.render import GaussianSplatOp, rasterize_gaussians, rasterize_gaussians_dynamic
from d3sim.constants import D3SIM_DEFAULT_DEVICE, IsAppleSiliconMacOs
from d3sim.algos.d3gs.config_def import GaussianSplatConfig

import numpy as np
from cumm import tensorview as tv 
from d3sim.data.scene_def.base import Pose, Resource
from d3sim.data.scene_def.camera import BasicPinholeCamera 

def main():
    if IsAppleSiliconMacOs:
        path = "/Users/yanyan/Downloads/PVG-chkpnt30000.pth"
        debug_bwd_path = "/Users/yanyan/Downloads/debug_pvg.pth"
    else:
        path = "/root/Projects/3dgs/PVG/eval_output/waymo_reconstruction/0145050/chkpnt30000.pth"
        debug_bwd_path = "/root/Projects/3dgs/PVG/pvg_grads.pt"
    model = load_pvg_model(path).to_parameter()
    # breakpoint()
    op = GaussianSplatOp(GaussianSplatConfig(render_depth=True, render_rgba=True, verbose=True))

    (viewpoint_camera, rendered_image_before, rendered_opacity, rendered_depth, 
            xyz_grads, opacity_grads,
             scale_grads, rotation_grads, feature_grads, t_grad, scaling_t_grad, velo_grad,
              _, dout_color) = torch.load(debug_bwd_path, map_location=D3SIM_DEFAULT_DEVICE)

    c2w = viewpoint_camera["c2w"].cpu().numpy()
    fx = viewpoint_camera["fx"]
    fy = viewpoint_camera["fy"]
    cx = viewpoint_camera["cx"]
    cy = viewpoint_camera["cy"]
    # fov_x = -1
    # fov_y = -1
    width = int(viewpoint_camera["resolution"][0])
    height = int(viewpoint_camera["resolution"][1])

    # fx = fov2focal(fov_x, width)
    # fy = fov2focal(fov_y, height)

    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    intrinsic_4x4 = np.eye(4, dtype=np.float32)
    intrinsic_4x4[:3, :3] = intrinsic
    timestamp = viewpoint_camera["timestamp"]
    cam = BasicPinholeCamera(id="", timestamp=0, pose=Pose(np.eye(4), c2w), 
        image_rc=Resource(base_uri="", loader_type=""), intrinsic=intrinsic_4x4, distortion=np.zeros(4, np.float32),
        image_shape_wh=(width, height), objects=[])
    
    res = rasterize_gaussians_dynamic(model, cam, op, training=True, prep_input_tensors={
        "batch_ts": torch.tensor([timestamp], dtype=torch.float32, device=D3SIM_DEFAULT_DEVICE),
    }, prep_inputs={
        "time_shift": 0,
    })
    res.color
    # res = rasterize_gaussians(model, cam, op)

    out_color = res.color
    # print(out_color.shape, bg_color_from_envmap.shape, rendered_image.shape)
    # out_color = rendered_image_before.permute(1, 2, 0)[None].to(D3SIM_DEFAULT_DEVICE)
    if not op.is_nchw:
        # to nchw
        out_color = out_color.permute(0, 3, 1, 2)
    print(torch.linalg.norm(out_color[:, :3] - rendered_image_before))
    out_color[0, :3].backward(dout_color.to(D3SIM_DEFAULT_DEVICE))    
    print("depth", torch.linalg.norm(res.depth.view(-1) - rendered_depth.view(-1)))
    print("alpha", torch.linalg.norm(out_color[0, 3].view(-1) - rendered_opacity.view(-1)))

    print(torch.linalg.norm(xyz_grads - model.xyz.grad))
    print("q", torch.linalg.norm(model.quaternion_xyzw.grad - rotation_grads[:, [1, 2, 3, 0]]))
    print("t", torch.linalg.norm(model.t.grad - t_grad))
    print("st", torch.linalg.norm(model.scaling_t.grad - scaling_t_grad))
    print("velo", torch.linalg.norm(model.velocity.grad - velo_grad))

    breakpoint()
    # rasterize_gaussians_dynamic



if __name__ == "__main__":
    main()
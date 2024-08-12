from d3sim.ops.d3gs.tools import load_3dgs_origin_model
import pickle
import time
import torch 
from d3sim.constants import D3SIM_DEFAULT_DEVICE, PACKAGE_ROOT, IsAppleSiliconMacOs
from d3sim.data.scene_def.base import Pose, Resource
from d3sim.data.scene_def.camera import BasicPinholeCamera
from d3sim.ops.d3gs.render import GaussianSplatConfig, GaussianSplatForward, rasterize_gaussians
from d3sim.ops.d3gs.data.scene.dataset_readers import readColmapSceneInfo
import numpy as np
from d3sim.ops.points.projection import depth_map_to_jet_rgb 

from torch.profiler import profile, record_function, ProfilerActivity
# def _validate_color_sh(color_sh: torch.Tensor, xyz: torch.Tensor,
#                        rgb_gaussian: torch.Tensor, cam2world_T_np: np.ndarray):
#     from d3sim.ops.d3gs.data.utils.sh_utils import eval_sh
#     shs_view = color_sh.transpose(1, 2).view(-1, 3, (3 + 1)**2)
#     dir_pp = (xyz - torch.from_numpy(cam2world_T_np[3]).cuda().repeat(
#         color_sh.shape[0], 1))
#     dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
#     sh2rgb = eval_sh(3, shs_view, dir_pp_normalized)
#     colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
#     print(torch.linalg.norm(colors_precomp - rgb_gaussian))

#     # print(INLINER.get_nvrtc_module(prep_kernel_name).params.debug_code)
#     # breakpoint()


# def build_covariance_from_scaling_rotation(scaling, scaling_modifier,
#                                            rotation):
#     L = build_scaling_rotation(scaling_modifier * scaling, rotation)
#     actual_covariance = L @ L.transpose(1, 2)
#     symm = strip_symmetric(actual_covariance)
#     return symm

class _Args:
    def __init__(self):
        self.resolution = -1
        self.data_device = "mps" if IsAppleSiliconMacOs else "cuda"

def sync():
    if IsAppleSiliconMacOs:
        torch.mps.synchronize()
    else:
        torch.cuda.synchronize()

def _load_model_and_cam():
    from d3sim.ops.d3gs.data.utils.camera_utils import cameraList_from_camInfos_gen, camera_to_JSON

    data_path = "/Users/yanyan/Downloads/360_v2/garden"
    # data_path = "/root/autodl-tmp/garden_scene/garden"
    path = "/Users/yanyan/Downloads/models/garden/point_cloud/iteration_30000/point_cloud.ply"
    # path = "/root/autodl-tmp/garden_model/garden/point_cloud/iteration_30000/point_cloud.ply"

    test_data_path = PACKAGE_ROOT.parent / "scripts/d3gs_cam.pkl"
    if test_data_path.exists():
        with open(test_data_path, "rb") as f:
            intrinsic, world2cam, image_shape_wh = pickle.load(f)
    else:
        scene_info = readColmapSceneInfo(data_path, "images_4", True)
        train_camera_infos = scene_info.train_cameras
        train_camera_first = next(cameraList_from_camInfos_gen(train_camera_infos, 1, _Args()))
        intrinsic = train_camera_first.get_intrinsic()
        world2cam = np.eye(4, dtype=np.float32)
        world2cam[:3, :3] = train_camera_first.R.T
        world2cam[:3, 3] = train_camera_first.T
        image_shape_wh = (train_camera_first.image_width, train_camera_first.image_height)
        with open(test_data_path, "wb") as f:
            pickle.dump((intrinsic, world2cam, image_shape_wh), f)
    # breakpoint()
    intrinsic_4x4 = np.eye(4, dtype=np.float32)
    intrinsic_4x4[:3, :3] = intrinsic

    mod = load_3dgs_origin_model(path, fused=True)
    cam = BasicPinholeCamera(id="", timestamp=0, pose=Pose(np.eye(4), np.linalg.inv(world2cam)), 
        image_rc=Resource(base_uri="", loader_type=""), intrinsic=intrinsic_4x4, distortion=np.zeros(4, np.float32),
        image_shape_wh=image_shape_wh, objects=[])

    return mod, cam


def _main_bwd():
    """
    benchmark
    fwd
    apple m3: 180ms (64bit sort), 113ms (32bit sort)
    cuda (RTX 4090): 9ms (64bit sort), 8.1ms (32bit sort)
    bwd:
    apple m3: 150ms
    cuda (RTX 4090): 16ms

    TDP: 16W vs 425W

    for forward, the bottleneck in m3 is sorting in 64bit. this may due to
    memory bandwidth. 4090 has 1000GB/s memory bandwidth, while m3 only
    has 100GB/s.
    """
    mod, cam = _load_model_and_cam()
    print(mod.xyz.shape)
    print("LOADED")
    cfg = GaussianSplatConfig()
    fwd = GaussianSplatForward(cfg)
    mod.set_requires_grad(True)
    grad_path = "/root/Projects/3dgs/gaussian-splatting/grads.pt"
    grad_path = "/Users/yanyan/grads.pt"
    grads = torch.load(grad_path, map_location=D3SIM_DEFAULT_DEVICE)
    check_grad: bool = True
    (xyz_grads, opacity_grads, scale_grads, rotation_grads, feature_grads, mean2d_grads, dout) = grads
    mean2d_grads = mean2d_grads[:, :2]
    uv_grad_holder = torch.empty_like(mean2d_grads)
    uv_grad_holder.requires_grad = True
    # breakpoint()
    for j in range(5):
        mod.clear_grad()
        uv_grad_holder.grad = None
        t = time.time()
        # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        # res, _ = fwd.forward(mod, [cam])
        if check_grad:
            res = rasterize_gaussians(mod, [cam], training=True, uv_grad_holder=uv_grad_holder)
        else:
            res = rasterize_gaussians(mod, [cam], training=True)
        sync()
        print("FWD", time.time() - t)
        assert res is not None 
        out_color = res.final_color
        # torch.manual_seed(50051)
        # dout_color = torch.rand(out_color.shape, device=out_color.device)
        sync()

        t = time.time()

        out_color.backward(dout)
        sync()
        # print("BWD", time.time() - t)
        # assert not mod.act_applied
        if check_grad:
            # print(dout_color.reshape(3, -1)[:, 1])
            print(mod.opacity.grad[1], res.final_T.reshape(-1)[1].item())
            print(opacity_grads.shape, mod.scale.grad.shape)
            print("opacity", torch.linalg.norm(mod.opacity.grad.reshape(opacity_grads.shape) - opacity_grads))
            print("scales", torch.linalg.norm(mod.scale.grad - scale_grads))
            print("q", torch.linalg.norm(mod.quaternion_xyzw.grad - rotation_grads[:, [1, 2, 3, 0]]))

            # print(torch.linalg.norm(mod.color_sh.grad - feature_grads))
            print(mod.xyz.grad[1])
            print(xyz_grads[1])
            print(torch.linalg.norm(mod.xyz.grad - xyz_grads))
            breakpoint()

def _main():
    mod, cam = _load_model_and_cam()

    for j in range(15):
        import torchvision
        torchvision.utils.save_image
        t = time.time()
        # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        # res, _ = fwd.forward(mod, [cam])
        res = rasterize_gaussians(mod, [cam])
        assert res is not None 
        out_color = res.final_color
        if j == 0:
            out_color_u8 = out_color.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            out_color_u8 = out_color_u8[..., ::-1]
            if res.final_depth is not None:
                depth = res.final_depth
                depth_rgb = depth_map_to_jet_rgb(depth, (0.2, 25.0)).cpu().numpy()
                out_color_u8 = np.concatenate([out_color_u8, depth_rgb], axis=0)

            # out_color_u8 = (out_color.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
            import cv2 
            cv2.imwrite("test.png", out_color_u8)
            # breakpoint()
        # else:
        #     print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        sync()
        print(time.time() - t)

    
if __name__ == "__main__":
    _main_bwd()
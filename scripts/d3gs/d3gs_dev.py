from d3sim.algos.d3gs.tools import load_3dgs_origin_model
import pickle
import time
import torch 
from d3sim.constants import D3SIM_DEFAULT_DEVICE, PACKAGE_ROOT, IsAppleSiliconMacOs
from d3sim.data.scene_def.base import Pose, Resource
from d3sim.data.scene_def.camera import BasicPinholeCamera
from d3sim.algos.d3gs.render import CameraBundle, GaussianSplatConfig, GaussianSplatOp, rasterize_gaussians
from d3sim.algos.d3gs.data.scene.dataset_readers import readColmapSceneInfo
import numpy as np
from d3sim.ops.points.projection import depth_map_to_jet_rgb 
from d3sim.algos.d3gs.data.load import load_model_and_2_cam, load_model_and_cam
from torch.profiler import profile, record_function, ProfilerActivity

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
    if IsAppleSiliconMacOs:
        data_path = "/Users/yanyan/Downloads/360_v2/garden"
        path = "/Users/yanyan/Downloads/models/garden/point_cloud/iteration_30000/point_cloud.ply"
    else:
        path = "/root/autodl-tmp/garden_model/garden/point_cloud/iteration_30000/point_cloud.ply"
        data_path = "/root/autodl-tmp/garden_scene/garden"

    return load_model_and_cam(data_path, path)

def _load_model_and_2cam():
    if IsAppleSiliconMacOs:
        data_path = "/Users/yanyan/Downloads/360_v2/garden"
        path = "/Users/yanyan/Downloads/models/garden/point_cloud/iteration_30000/point_cloud.ply"
    else:
        path = "/root/autodl-tmp/garden_model/garden/point_cloud/iteration_30000/point_cloud.ply"
        data_path = "/root/autodl-tmp/garden_scene/garden"

    return load_model_and_2_cam(data_path, path)

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
    fwd = GaussianSplatOp(cfg)
    mod.set_requires_grad(True)
    if IsAppleSiliconMacOs:
        grad_path = "/Users/yanyan/grads.pt"
    else:
        grad_path = "/root/Projects/3dgs/gaussian-splatting/grads.pt"
    # 
    check_grad: bool = True
    if check_grad:
        grads = torch.load(grad_path, map_location=D3SIM_DEFAULT_DEVICE)

        (xyz_grads, opacity_grads, scale_grads, rotation_grads, feature_grads, mean2d_grads, dout) = grads
        mean2d_grads = mean2d_grads[:, :2]

    uv_grad_holder = torch.empty(mod.xyz.shape[0], 2, dtype=torch.float32, device=mod.xyz.device)
    uv_grad_holder.requires_grad = True
    # breakpoint()
    cfg = GaussianSplatConfig(verbose=False)
    op = GaussianSplatOp(cfg)
    for j in range(5):
        mod.clear_grad()
        print("cur_sh_degree", mod.cur_sh_degree)
        uv_grad_holder.grad = None
        t = time.time()
        # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        # res, _ = fwd.forward(mod, [cam])
        if check_grad:
            res = rasterize_gaussians(mod, cam, op=op, training=True, uv_grad_holder=uv_grad_holder)
        else:
            res = rasterize_gaussians(mod, cam, op=op, training=True)
        sync()
        print("FWD", time.time() - t)
        assert res is not None 
        out_color = res.color[0]


        # torch.manual_seed(50051)
        sync()
        if not check_grad:
            dout = torch.rand(out_color.shape, device=out_color.device)

        t = time.time()
        if not op.is_nchw:
            dout = dout.permute(1, 2, 0)
        out_color.backward(dout)
        sync()
        print("BWD", time.time() - t)
        # assert not mod.act_applied
        if check_grad:
            # print(dout_color.reshape(3, -1)[:, 1])
            print(mod.opacity.grad[1], res.T.reshape(-1)[1].item())
            print(opacity_grads.shape, mod.scale.grad.shape)
            print("opacity", torch.linalg.norm(mod.opacity.grad.reshape(opacity_grads.shape) - opacity_grads))
            print("scales", torch.linalg.norm(mod.scale.grad - scale_grads))
            print("q", torch.linalg.norm(mod.quaternion_xyzw.grad - rotation_grads[:, [1, 2, 3, 0]]))

            # print(torch.linalg.norm(mod.color_sh.grad - feature_grads))
            print(mod.xyz.grad[1])
            print(xyz_grads[1])
            print(torch.linalg.norm(mod.xyz.grad - xyz_grads))
            breakpoint()
    breakpoint()
    print("?")

def _main():
    mod, cams = _load_model_and_2cam()
    # cams[1] = cams[0]
    cfg = GaussianSplatConfig(verbose=False, enable_32bit_sort=True, gaussian_std_sigma=2.0)
    # cfg = GaussianSplatConfig(verbose=False)

    op = GaussianSplatOp(cfg)
    cam2worlds = [cam.pose.to_world for cam in cams]
    cam2world_Ts = [cam2world[:3].T for cam2world in cam2worlds]
    cam2world_Ts_onearray = np.stack(cam2world_Ts, axis=0).reshape(-1, 4, 3)
    cam2world_Ts_th = torch.from_numpy(np.ascontiguousarray(cam2world_Ts_onearray)).to(D3SIM_DEFAULT_DEVICE)

    focal_xys = np.stack([cam.focal_length for cam in cams], axis=0).reshape(-1, 2)
    ppoints = np.stack([cam.principal_point for cam in cams], axis=0).reshape(-1, 2)
    focal_xy_ppoints = np.concatenate([focal_xys, ppoints], axis=1)
    focal_xy_ppoints_th = torch.from_numpy(np.ascontiguousarray(focal_xy_ppoints)).to(D3SIM_DEFAULT_DEVICE)
    cam_bundle = CameraBundle(focal_xy_ppoints_th, cams[0].image_shape_wh, None, cam2world_Ts_th)
    for j in range(15):
        t = time.time()
        # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        # res, _ = fwd.forward(mod, [cam])
        res = rasterize_gaussians(mod, cam_bundle, op=op)
        assert res is not None 
        out_color = res.color # [N, C, H, W]
        assert res.depth is None
        if j == 0:
            if not op.is_nchw:
                # to nchw
                out_color = out_color.permute(0, 3, 1, 2)

            # breakpoint()
            #  -> [C, H * N, W]
            out_color = torch.concat([out_color[i] for i in range(out_color.shape[0])], dim=1)

            out_color_u8 = out_color.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            if op.is_rgba:
                out_color_u8 = out_color_u8[..., [2, 1, 0, 3]]
            else:
                out_color_u8 = out_color_u8[..., ::-1]
            if res.depth is not None:
                depth = res.depth
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
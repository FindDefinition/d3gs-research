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

class _Args:
    def __init__(self):
        self.resolution = -1
        self.data_device = "mps"

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

    mod = load_3dgs_origin_model(path)
    cam = BasicPinholeCamera(id="", timestamp=0, pose=Pose(np.eye(4), np.linalg.inv(world2cam)), 
        image_rc=Resource(base_uri="", loader_type=""), intrinsic=intrinsic_4x4, distortion=np.zeros(4, np.float32),
        image_shape_wh=image_shape_wh, objects=[])

    return mod, cam


def _main_bwd():
    mod, cam = _load_model_and_cam()
    print("LOADED")
    cfg = GaussianSplatConfig()
    fwd = GaussianSplatForward(cfg)
    mod.set_requires_grad(True)

    for j in range(3):
        mod.clear_grad()
        t = time.time()
        # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        # res, _ = fwd.forward(mod, [cam])
        res = rasterize_gaussians(mod, [cam], training=True)
        sync()
        print("FWD", time.time() - t)
        t = time.time()
        assert res is not None 
        out_color = res.final_color
        dout_color = torch.rand(out_color.shape, device=out_color.device)
        out_color.backward(dout_color)
        sync()
        print("BWD", time.time() - t)


def _main():
    mod, cam = _load_model_and_cam()
    with torch.mps.profiler.profile("interval,event"):

        for j in range(3):
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
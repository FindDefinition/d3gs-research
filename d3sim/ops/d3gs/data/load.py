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
from d3sim.ops.d3gs.data.utils.camera_utils import cameraList_from_camInfos_gen, camera_to_JSON

class _Args:
    def __init__(self):
        self.resolution = -1
        self.data_device = "mps" if IsAppleSiliconMacOs else "cuda"

def load_scene_info_and_first_cam(data_path: str):
    scene_info = readColmapSceneInfo(data_path, "images_4", True)
    train_camera_infos = scene_info.train_cameras
    train_camera_first = next(cameraList_from_camInfos_gen(train_camera_infos, 1, _Args()))
    intrinsic = train_camera_first.get_intrinsic()
    world2cam = np.eye(4, dtype=np.float32)
    world2cam[:3, :3] = train_camera_first.R.T
    world2cam[:3, 3] = train_camera_first.T
    image_shape_wh = (train_camera_first.image_width, train_camera_first.image_height)
    intrinsic_4x4 = np.eye(4, dtype=np.float32)
    intrinsic_4x4[:3, :3] = intrinsic

    cam = BasicPinholeCamera(id="", timestamp=0, pose=Pose(np.eye(4), np.linalg.inv(world2cam)), 
        image_rc=Resource(base_uri="", loader_type=""), intrinsic=intrinsic_4x4, distortion=np.zeros(4, np.float32),
        image_shape_wh=image_shape_wh, objects=[])

    return scene_info, cam

def load_model_and_cam(data_path: str, model_path: str):
    from d3sim.ops.d3gs.data.utils.camera_utils import cameraList_from_camInfos_gen, camera_to_JSON
    path = model_path
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


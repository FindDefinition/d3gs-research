from d3sim.algos.d3gs.pvg.data.waymo_loader import readWaymoInfo, cameraList_from_camInfos_gen, PVGArgs, Scene as PVGScene
import numpy as np 
from d3sim.data.scene_def.base import CameraFieldTypes, Pose, Resource
from d3sim.data.scene_def.camera import BasicPinholeCamera


def load_scene_info_and_first_cam(data_path: str, model_path: str):
    args = PVGArgs(data_path, model_path)
    scene_info = readWaymoInfo(args)
    train_camera_infos = scene_info.train_cameras
    train_camera_first = next(cameraList_from_camInfos_gen(train_camera_infos, 1, args))
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

def load_scene(data_path: str, model_path: str):
    args = PVGArgs(data_path, model_path)
    scene = PVGScene(args)
    return scene


def __main():
    path = "/Users/yanyan/Downloads/waymo_scenes/0145050"
    si, cam = load_scene_info_and_first_cam(path, "") 

if __name__ == "__main__":
    __main()

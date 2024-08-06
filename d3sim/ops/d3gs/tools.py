from plyfile import PlyData
import torch 
from d3sim.constants import D3SIM_DEFAULT_DEVICE
from d3sim.ops.d3gs.base import GaussianModelOrigin
from d3sim.ops.d3gs.forward import GaussianSplatConfig, GaussianSplatForward
from d3sim.ops.d3gs.data.scene.dataset_readers import readColmapSceneInfo
import numpy as np 

def load_3dgs_origin_model(ply_path: str):
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
    f_rest = np.stack(rest_vecs, axis=1)
    sh_coeffs = np.concatenate([f_dc, f_rest], axis=1).astype(np.float32) 
    rot_vecs = []
    for i in range(4):
        rot_vecs.append(vertex[f"rot_{i}"])
    rots = np.stack(rot_vecs, axis=1).astype(np.float32)[:, [1, 2, 3, 0]]
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
    model = GaussianModelOrigin(xyz=positions_th, quaternion_xyzw=rots_th, scale=scales_th, opacity=opacity_th, color_sh=sh_coeffs_th)
    return model

class _Args:
    def __init__(self):
        self.resolution = -1
        self.data_device = "mps"

def _main():
    from d3sim.ops.d3gs.data.utils.camera_utils import cameraList_from_camInfos_gen, camera_to_JSON

    data_path = "/Users/yanyan/Downloads/360_v2/garden"
    scene_info = readColmapSceneInfo(data_path, None, False)
    train_camera_infos = scene_info.train_cameras
    train_camera_first = next(cameraList_from_camInfos_gen(train_camera_infos, 1, _Args()))
    intrinsic = train_camera_first.get_intrinsic()
    world2cam = np.eye(4, dtype=np.float32)
    world2cam[:3, :3] = train_camera_first.R.T
    world2cam[:3, 3] = train_camera_first.T
    image_shape_wh = (train_camera_first.image_width, train_camera_first.image_height)
    # breakpoint()
    print("REF", train_camera_first.FoVx, train_camera_first.FoVy)
    path = "/Users/yanyan/Downloads/models/garden/point_cloud/iteration_30000/point_cloud.ply"
    mod = load_3dgs_origin_model(path)
    cfg = GaussianSplatConfig()
    fwd = GaussianSplatForward(cfg)
    fwd.forward(mod, intrinsic, np.linalg.inv(world2cam),image_shape_wh, axis_front_u_v=(2, 0, 1))

if __name__ == "__main__":
    _main()
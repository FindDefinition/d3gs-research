import time
from d3sim.constants import D3SIM_DEFAULT_DEVICE
from d3sim.core.thtools import np_to_torch_dev
from d3sim.data.scene_def.base import DistortType
from d3sim.ops.points.common import points_in_which_3d_box
from d3sim.ops.points.transform import transform_xyz
from d3sim.ops.points.projection import project_points_to_camera, projected_point_uvd_to_jet_rgb
from .base import BaseFrame, CoordSystem
from .camera import BasicCamera
from .lidar import BasicLidar

import d3sim.core.dataclass_dispatch as dataclasses
import torch
import numpy as np 

@dataclasses.dataclass
class BasicFrame(BaseFrame):
    def get_camera_by_id(self, sensor_id: str):
        return self.get_sensor_by_id_checked(sensor_id, BasicCamera)

    def get_lidar_by_id(self, sensor_id: str):
        return self.get_sensor_by_id_checked(sensor_id, BasicLidar)

    def get_cameras(self):
        return self.get_sensors_by_type(BasicCamera)

    def get_lidars(self):
        return self.get_sensors_by_type(BasicLidar)

    def get_lidar_xyz(self,
                      lidar_id: str,
                      target_coord: str | CoordSystem,
                      use_cuda: bool = True):
        lidar = self.get_lidar_by_id(lidar_id)
        res = self.get_lidar_xyz_raw(lidar_id, target_coord,
                                        use_cuda)
        if lidar.mask is not None:
            res = res[torch.from_numpy(lidar.mask).to(D3SIM_DEFAULT_DEVICE)]
        return res

    def get_lidar_xyz_np(self,
                      lidar_id: str,
                      target_coord: str | CoordSystem,
                      use_cuda: bool = True):
        return self.get_lidar_xyz(lidar_id, target_coord, use_cuda).cpu().numpy()

    def get_lidar_xyz_raw(self,
                          lidar_id: str,
                          target_coord: str | CoordSystem,
                          use_cuda: bool = True):
        lidar = self.get_lidar_by_id(lidar_id)
        if lidar.data_coord is not None and lidar.data_coord == target_coord:
            return np_to_torch_dev(lidar.xyz)
        if not use_cuda:
            raise NotImplementedError
        xyz = lidar.xyz
        mat = self.get_transform_matrix(
            lidar.data_coord if lidar.data_coord is not None else lidar.id,
            target_coord)
        xyz_th = np_to_torch_dev(xyz)
        res = transform_xyz(xyz_th, mat)
        return res

    def get_projected_pc_in_cam(self, lidar_id_or_pc: str | np.ndarray | torch.Tensor, cam_id: str):
        cam = self.get_camera_by_id(cam_id)
        if isinstance(lidar_id_or_pc, str):
            lidar = self.get_lidar_by_id(lidar_id_or_pc)
            lidar_xyz_th = self.get_lidar_xyz_raw(lidar_id_or_pc, CoordSystem.VEHICLE)
            lidar_mask = lidar.mask
        else:
            lidar_xyz_th = np_to_torch_dev(lidar_id_or_pc)
            lidar_mask = None
        c2v = self.get_transform_matrix(cam.id, CoordSystem.VEHICLE)
        uvd, mask = project_points_to_camera(
            lidar_xyz_th, c2v,
            cam.intrinsic, cam.image_shape_wh, cam.distortion,
            DistortType.kOpencvPinhole, cam.axes_front_u_v)
        if lidar_mask is not None:
            mask = mask & np_to_torch_dev(lidar_mask)
        return uvd[mask]

    def get_projected_image_in_cam_np(self,
                                   lidar_id_or_pc: str | np.ndarray | torch.Tensor,
                                   cam_id: str,
                                   val_range: tuple[float, float] = (0.5, 60.0),
                                   size: int = 2):
        uvd = self.get_projected_pc_in_cam(lidar_id_or_pc, cam_id)
        cam = self.get_camera_by_id(cam_id)
        img = cam.image
        img = projected_point_uvd_to_jet_rgb(
            uvd, cam.image_shape_wh,
            val_range,
            np_to_torch_dev(img), size)
        return img.cpu().numpy()

    def get_box3d_np(self, to_world: bool = False, source: str = ""):
        objects = self.get_objects_by_source(source)
        box3d = np.empty((len(objects), 7), dtype=np.float32)
        for i, obj in enumerate(objects):
            if to_world:
                mat = obj.pose.to_world
                euler = obj.pose.get_euler_in_world()
            else:
                mat = obj.pose.to_vehicle
                euler = obj.pose.get_euler_in_vehicle()
            center = mat[:3, 3]
            size = obj.size 
            yaw = euler[2]
            box3d[i, :3] = center
            box3d[i, 3:6] = size
            box3d[i, 6] = yaw
        return box3d

    def get_point_in_which_3d_box(self, lidar_id_or_pc: str | np.ndarray | torch.Tensor):
        if isinstance(lidar_id_or_pc, str):
            lidar_xyz_th = self.get_lidar_xyz_raw(lidar_id_or_pc, CoordSystem.VEHICLE)
        else:
            lidar_xyz_th = np_to_torch_dev(lidar_id_or_pc)
        box3d = torch.from_numpy(self.get_box3d_np()).to(D3SIM_DEFAULT_DEVICE)
        res = points_in_which_3d_box(lidar_xyz_th, box3d)
        return res
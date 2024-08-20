import time
from d3sim.constants import D3SIM_DEFAULT_DEVICE
from d3sim.core.thtools import np_to_torch_dev
from d3sim.data.scene_def.base import DistortType
from d3sim.ops.image.rayops import create_cam_box3d_mask, get_ray_dirs
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
        res = self.get_lidar_xyz_raw(lidar_id, target_coord, use_cuda)
        if lidar.mask is not None:
            res = res[torch.from_numpy(lidar.mask).to(D3SIM_DEFAULT_DEVICE)]
        return res

    def get_lidar_xyz_np(self,
                         lidar_id: str,
                         target_coord: str | CoordSystem,
                         use_cuda: bool = True):
        return self.get_lidar_xyz(lidar_id, target_coord,
                                  use_cuda).cpu().numpy()

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

    def get_projected_pc_in_cam(self, lidar_id_or_pc: str | np.ndarray
                                | torch.Tensor, cam_id: str):
        cam = self.get_camera_by_id(cam_id)
        if isinstance(lidar_id_or_pc, str):
            lidar = self.get_lidar_by_id(lidar_id_or_pc)
            lidar_xyz_th = self.get_lidar_xyz_raw(lidar_id_or_pc,
                                                  CoordSystem.VEHICLE)
            lidar_mask = lidar.mask
        else:
            lidar_xyz_th = np_to_torch_dev(lidar_id_or_pc)
            lidar_mask = None
        c2v = self.get_transform_matrix(cam.id, CoordSystem.VEHICLE)
        uvd, mask = project_points_to_camera(lidar_xyz_th, c2v, cam.intrinsic,
                                             cam.image_shape_wh,
                                             cam.distortion,
                                             cam.distortion_type)
        if lidar_mask is not None:
            mask = mask & np_to_torch_dev(lidar_mask)
        return uvd, mask

    def get_projected_image_in_cam_np(
            self,
            lidar_id_or_pc: str | np.ndarray | torch.Tensor,
            cam_id: str,
            val_range: tuple[float, float] = (0.5, 60.0),
            size: int = 2,
            z_min: float = 0.0):
        uvd, uvd_mask = self.get_projected_pc_in_cam(lidar_id_or_pc, cam_id)
        uvd = uvd[uvd_mask]
        cam = self.get_camera_by_id(cam_id)
        img = cam.image
        img = projected_point_uvd_to_jet_rgb(uvd,
                                             cam.image_shape_wh, val_range,
                                             np_to_torch_dev(img), size,
                                             z_min)
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

    def get_box3d_local_track_ids_np(self, source: str = ""):
        objects = self.get_objects_by_source(source)
        res = np.empty((len(objects), ), dtype=np.int32)
        for i, obj in enumerate(objects):
            res[i] = obj.track_local_id
        return res

    def get_point_in_which_3d_box(self,
                                  lidar_id_or_pc: str | np.ndarray
                                  | torch.Tensor,
                                  box3d_scale: np.ndarray | None = None,
                                  box3d: np.ndarray | None = None):
        if isinstance(lidar_id_or_pc, str):
            lidar_xyz_th = self.get_lidar_xyz_raw(lidar_id_or_pc,
                                                  CoordSystem.VEHICLE)
        else:
            lidar_xyz_th = np_to_torch_dev(lidar_id_or_pc)
        if box3d is None:
            box3d = self.get_box3d_np()
        box3d_th = torch.from_numpy(box3d).to(D3SIM_DEFAULT_DEVICE)
        res = points_in_which_3d_box(lidar_xyz_th, box3d_th, box3d_scale)
        return res

    def get_point_in_which_global_3d_box(self,
                                         lidar_id_or_pc: str | np.ndarray
                                         | torch.Tensor,
                                         box3d_scale: np.ndarray | None = None,
                                         box3d: np.ndarray | None = None,
                                         box3d_ids: np.ndarray | None = None):
        if isinstance(lidar_id_or_pc, str):
            lidar_xyz_th = self.get_lidar_xyz_raw(lidar_id_or_pc,
                                                  CoordSystem.VEHICLE)
        else:
            lidar_xyz_th = np_to_torch_dev(lidar_id_or_pc)
        if box3d is None:
            box3d = self.get_box3d_np()
        if box3d_ids is None:
            box3d_ids = self.get_box3d_local_track_ids_np()
        box3d_th = torch.from_numpy(box3d).to(D3SIM_DEFAULT_DEVICE)
        box3d_ids_th = torch.from_numpy(box3d_ids).to(D3SIM_DEFAULT_DEVICE)
        res = points_in_which_3d_box(lidar_xyz_th, box3d_th, box3d_scale,
                                     box3d_ids_th)
        return res

    def get_image_3d_object_mask(self,
                                    cam_id: str,
                                    box3d_scale: np.ndarray | None = None,
                                    box3d: np.ndarray | None = None):
        cam = self.get_camera_by_id(cam_id)

        dirs = self.get_camera_ray_dirs(cam_id, CoordSystem.VEHICLE)
        if box3d is None:
            box3d = self.get_box3d_np()
        cam_to_vehicle = cam.pose.to_vehicle
        cam_origin = cam_to_vehicle[:3, 3]
        box3d_th = np_to_torch_dev(box3d)
        return create_cam_box3d_mask(cam_origin, dirs, cam.image_shape_wh,
                                     box3d_th, box3d_scale)

    def get_image_3d_object_mask_np(self,
                                    cam_id: str,
                                    box3d_scale: np.ndarray | None = None,
                                    box3d: np.ndarray | None = None):
        return self.get_image_3d_object_mask(cam_id, box3d_scale,
                                               box3d).cpu().numpy()

    def get_camera_ray_dirs(self, cam_id: str, target_coord: str | CoordSystem):
        cam = self.get_camera_by_id(cam_id)
        cam_to_target = self.get_transform_matrix(cam.id, target_coord)
        res = get_ray_dirs(cam_to_target, cam.intrinsic, cam.image_shape_wh,
                           cam.distortion, cam.distortion_type) 
        return res

    def get_camera_ray_dirs_np(self, cam_id: str, target_coord: str | CoordSystem):
        return self.get_camera_ray_dirs(cam_id, target_coord).cpu().numpy()
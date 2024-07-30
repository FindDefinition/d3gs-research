from d3sim.constants import D3SIM_DEFAULT_DEVICE
from d3sim.ops.points.transform import transform_xyz
from .base import BaseFrame, CoordSystem
from .camera import BasicCamera
from .lidar import BasicLidar

import d3sim.core.dataclass_dispatch as dataclasses
import torch 

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

    def get_lidar_xyz(self, lidar_id: str, target_coord: str | CoordSystem, use_cuda: bool = True):
        lidar = self.get_lidar_by_id(lidar_id)
        if lidar.data_coord is not None and lidar.data_coord == target_coord:
            return lidar.xyz[lidar.mask]
        if not use_cuda:
            raise NotImplementedError
        xyz = lidar.xyz
        mat = self.get_transform_matrix(lidar.data_coord if lidar.data_coord is not None else lidar.id, target_coord)
        xyz_th = torch.from_numpy(xyz).to(D3SIM_DEFAULT_DEVICE)
        res = transform_xyz(xyz_th, mat)
        return res.cpu().numpy()[lidar.mask]
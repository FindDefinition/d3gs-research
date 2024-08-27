import abc
from concurrent.futures import Executor
from typing import Any, Literal 
from d3sim.core import dataclass_dispatch as dataclasses
import numpy as np
import importlib
from d3sim.data.scene_def.camera import BasicPinholeCamera
from d3sim.data.scene_def.base import BaseFrame, Scene
from d3sim.data.scene_def.lidar import BasicLidar
from d3sim.data.scene_def.transform import ALL_TRANSFORM_TYPES, run_transforms_on_scene 


@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class AnyObject:
    type: str 
    config: dict[str, Any]

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class DatasetPointCloud:
    xyz: np.ndarray
    rgb: np.ndarray | None = None 
    # assume relative value
    timestamp: np.ndarray | None = None
    # assume uint8 (0-255)
    intensity: np.ndarray | None = None
    # d3sim standard segmentation (unknown, ground, vehicle, pedestrian, etc)
    segmentation: np.ndarray | None = None

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class MultipleSceneDatasetBase(abc.ABC):
    scenes: list[Scene[BaseFrame]] = dataclasses.field(default_factory=list)
    # transforms that only need to be applied once
    # such as colmap.
    # offline transforms often save addifional fields
    # to root of origin scene, so don't forget to set
    # root path of scenes
    offline_transforms: list[ALL_TRANSFORM_TYPES] = dataclasses.field(default_factory=list)
    # transforms that need to be applied for each batch.
    # such as data augmentation.
    online_transforms: list[ALL_TRANSFORM_TYPES] = dataclasses.field(default_factory=list)

    _id_to_scene: dict[str, Scene[BaseFrame]] = dataclasses.field(default_factory=dict)

    _global_id_to_frame: dict[tuple[str, str], BaseFrame] = dataclasses.field(default_factory=dict)
    _global_id_to_camera: dict[tuple[str, str], BasicPinholeCamera] = dataclasses.field(default_factory=dict)
    _global_id_to_lidar: dict[tuple[str, str], BasicLidar] = dataclasses.field(default_factory=dict)

    def apply_offline_transform_inplace(self, ex: Executor | None = None):
        for i in range(len(self.scenes)):
            scene = self.scenes[i]
            scene = run_transforms_on_scene(scene, self.offline_transforms, ex)
            self.scenes[i] = scene
        self._prepare_all_datas()

    def _prepare_all_datas(self):
        for scene in self.scenes:
            self._id_to_scene[scene.id] = scene
            for frame in scene.frames:
                self._global_id_to_frame[(scene.id, frame.global_id)] = frame
                for sensor in frame.sensors:
                    if isinstance(sensor, BasicPinholeCamera):
                        self._global_id_to_camera[(scene.id, sensor.global_id)] = sensor
                    elif isinstance(sensor, BasicLidar):
                        self._global_id_to_lidar[(scene.id, sensor.global_id)] = sensor

    def get_scene_by_id(self, scene_id: str) -> Scene[BaseFrame]:
        return self._id_to_scene[scene_id]

    def get_frame_by_id(self, scene_id: str, frame_id: str) -> BaseFrame:
        return self._global_id_to_frame[(scene_id, frame_id)]

    def get_camera_by_id(self, scene_id: str, camera_id: str) -> BasicPinholeCamera:
        return self._global_id_to_camera[(scene_id, camera_id)]

    def get_lidar_by_id(self, scene_id: str, lidar_id: str) -> BasicLidar:
        return self._global_id_to_lidar[(scene_id, lidar_id)]

    def get_frame_global_ids(self) -> list[tuple[str, str]]:
        return list(self._global_id_to_frame.keys())

    def get_camera_global_ids(self) -> list[tuple[str, str]]:
        return list(self._global_id_to_camera.keys())

    def get_lidar_global_ids(self) -> list[tuple[str, str]]:
        return list(self._global_id_to_lidar.keys())

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class D3simDataset(MultipleSceneDatasetBase):
    @property 
    @abc.abstractmethod
    def dataset_point_cloud(self) -> DatasetPointCloud:
        raise NotImplementedError

    @property 
    @abc.abstractmethod
    def extent(self) -> float:
        raise NotImplementedError

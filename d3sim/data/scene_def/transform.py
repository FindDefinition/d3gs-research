import abc
from concurrent.futures import Executor
from pathlib import Path
import shutil
import d3sim.core.dataclass_dispatch as dataclasses
from typing import Any, Sequence, TypeAlias, TypeVar
import importlib 
from d3sim.core.inspecttools import get_qualname_of_type
from d3sim.data.scene_def import Scene, BasicFrame, BasicPinholeCamera, BasicLidar
from d3sim.data.scene_def.base import BaseFrame
import numpy as np 
import json 

T = TypeVar("T")

def get_module_id_of_type(klass: type) -> str:
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + '::' + "::".join(klass.__qualname__.split("."))

def get_object_type_from_module_id(module_id: str):
    """Get object type from module id."""
    module_key = module_id.split("::")[0]
    mod = importlib.import_module(module_key)

    local_key = "::".join(module_id.split("::")[1:])
    module_dict = mod.__dict__
    if module_dict is None:
        return None
    parts = local_key.split("::")
    obj = module_dict[parts[0]]
    for part in parts[1:]:
        obj = getattr(obj, part)
    return obj


class SceneTransform(abc.ABC):
    @abc.abstractmethod
    def __call__(self, scene: Scene) -> Scene:
        raise NotImplementedError


class FrameTransform(abc.ABC):
    @abc.abstractmethod
    def __call__(self, frame: BasicFrame) -> BasicFrame:
        raise NotImplementedError


class CameraTransform(abc.ABC):
    @abc.abstractmethod
    def __call__(self, camera: BasicPinholeCamera) -> BasicPinholeCamera:
        raise NotImplementedError

class LidarTransform(abc.ABC):
    @abc.abstractmethod
    def __call__(self, camera: BasicLidar) -> BasicLidar:
        raise NotImplementedError

class DynamicTransform:
    def __init__(self, module_id: str, config: dict[str, Any]):
        self._module_id = module_id
        self._config = config
        transform_cls = get_object_type_from_module_id(module_id)
        assert transform_cls is not None, f"Cannot find class {module_id}"
        self._transform = transform_cls(**config)

    def get_transform_by_type(self, type: type[T]) -> T:
        if isinstance(self._transform, type):
            return self._transform
        raise NotImplementedError

    def __call__(self, obj: Any) -> Any:
        if isinstance(obj, Scene):
            assert isinstance(self._transform, SceneTransform)
            return self._transform(obj)
        elif isinstance(obj, BasicFrame):
            assert isinstance(self._transform, FrameTransform)
            return self._transform(obj)
        elif isinstance(obj, BasicPinholeCamera):
            assert isinstance(self._transform, CameraTransform)
            return self._transform(obj)
        elif isinstance(obj, BasicLidar):
            assert isinstance(self._transform, LidarTransform)
            return self._transform(obj)
        raise NotImplementedError


ALL_TRANSFORM_TYPES: TypeAlias = SceneTransform | FrameTransform | CameraTransform | LidarTransform | DynamicTransform
ALL_ONLINE_TRANSFORM_TYPES: TypeAlias = FrameTransform | CameraTransform | LidarTransform | DynamicTransform

def apply_frame_transform(scene: Scene, frame_transform: FrameTransform, ex: Executor | None = None) -> Scene:
    if ex is not None:
        scene = dataclasses.replace(scene, frames=list(ex.map(frame_transform, scene.frames)))
    else:
        scene = dataclasses.replace(scene, frames=[frame_transform(frame) for frame in scene.frames])
    return scene

def apply_camera_transform(scene: Scene, camera_transform: CameraTransform, ex: Executor | None = None) -> Scene:
    sensors = scene.get_sensors_by_type(BasicPinholeCamera)
    
    if ex is not None:
        new_cameras = list(ex.map(camera_transform, sensors))
    else:
        new_cameras = [camera_transform(camera) for camera in sensors]
    frame_id_to_cams: dict[str, list[BasicPinholeCamera]] = {}
    for c in new_cameras:
        if c.frame_id not in frame_id_to_cams:
            frame_id_to_cams[c.frame_id] = []
        frame_id_to_cams[c.frame_id].append(c)
    new_frames: list[BasicFrame] = []
    for frame_id, cams in frame_id_to_cams.items():
        frame = scene.get_frame_by_id(frame_id)
        frame = frame.override_sensors(cams)
        new_frames.append(frame)
    
    return dataclasses.replace(scene, frames=new_frames)

def apply_lidar_transform(scene: Scene, lidar_transform: LidarTransform, ex: Executor | None = None) -> Scene:
    sensors = scene.get_sensors_by_type(BasicLidar)
    
    if ex is not None:
        new_lidars = list(ex.map(lidar_transform, sensors))
    else:
        new_lidars = [lidar_transform(lidar) for lidar in sensors]
    frame_id_to_lidars: dict[str, list[BasicLidar]] = {}
    for l in new_lidars:
        if l.frame_id not in frame_id_to_lidars:
            frame_id_to_lidars[l.frame_id] = []
        frame_id_to_lidars[l.frame_id].append(l)
    new_frames: list[BasicFrame] = []
    for frame_id, lidars in frame_id_to_lidars.items():
        frame = scene.get_frame_by_id(frame_id)
        frame = frame.override_sensors(lidars)
        new_frames.append(frame)
    
    return dataclasses.replace(scene, frames=new_frames)


def run_transforms_on_scene(scene: Scene, transforms: Sequence[ALL_TRANSFORM_TYPES], ex: Executor | None = None) -> Scene:
    for transform in transforms:
        if isinstance(transform, SceneTransform):
            if isinstance(transform, DynamicTransform):
                transform = transform.get_transform_by_type(SceneTransform)
            scene = transform(scene)
        elif isinstance(transform, FrameTransform):
            if isinstance(transform, DynamicTransform):
                transform = transform.get_transform_by_type(FrameTransform)
            scene = apply_frame_transform(scene, transform, ex)
        elif isinstance(transform, CameraTransform):
            if isinstance(transform, DynamicTransform):
                transform = transform.get_transform_by_type(CameraTransform)
            scene = apply_camera_transform(scene, transform, ex)
        elif isinstance(transform, LidarTransform):
            if isinstance(transform, DynamicTransform):
                transform = transform.get_transform_by_type(LidarTransform)
            scene = apply_lidar_transform(scene, transform, ex)
        else:
            raise NotImplementedError
    return scene

def run_online_transforms_on_frame(frame: BasicFrame, transforms: Sequence[ALL_ONLINE_TRANSFORM_TYPES]) -> BasicFrame:
    for transform in transforms:
        if isinstance(transform, FrameTransform):
            if isinstance(transform, DynamicTransform):
                transform = transform.get_transform_by_type(FrameTransform)
            frame = transform(frame)
        elif isinstance(transform, CameraTransform):
            if isinstance(transform, DynamicTransform):
                transform = transform.get_transform_by_type(CameraTransform)
            sensors = frame.get_sensors_by_type(BasicPinholeCamera)
            new_cameras = [transform(camera) for camera in sensors]
            frame = frame.override_sensors(new_cameras)
        elif isinstance(transform, LidarTransform):
            if isinstance(transform, DynamicTransform):
                transform = transform.get_transform_by_type(LidarTransform)
            sensors = frame.get_sensors_by_type(BasicLidar)
            new_lidars = [transform(lidar) for lidar in sensors]
            frame = frame.override_sensors(new_lidars)
        else:
            raise NotImplementedError("Unknown transform type, only support cam/lidar/frame transform.")
    return frame


@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class MultipleSceneDatasetBase(abc.ABC):
    scenes: list[Scene[BasicFrame]] = dataclasses.field(default_factory=list)
    # transforms that only need to be applied once
    # such as colmap.
    # offline transforms often save addifional fields
    # to root of origin scene, or just load them if exists,
    # so don't forget to set root path of scenes
    offline_transforms: list[ALL_TRANSFORM_TYPES] = dataclasses.field(default_factory=list)
    # transforms that need to be applied for each batch.
    # such as data augmentation. 
    online_transforms: list[ALL_ONLINE_TRANSFORM_TYPES] = dataclasses.field(default_factory=list)

    _id_to_scene: dict[str, Scene[BasicFrame]] = dataclasses.field(default_factory=dict)

    _global_id_to_frame: dict[tuple[str, str], BasicFrame] = dataclasses.field(default_factory=dict)
    _global_id_to_camera: dict[tuple[str, str], BasicPinholeCamera] = dataclasses.field(default_factory=dict)
    _global_id_to_lidar: dict[tuple[str, str], BasicLidar] = dataclasses.field(default_factory=dict)

    def apply_offline_transform_inplace(self, ex: Executor | None = None):
        for i in range(len(self.scenes)):
            scene = self.scenes[i]
            scene = run_transforms_on_scene(scene, self.offline_transforms, ex)
            self.scenes[i] = scene
        self._prepare_all_datas()

    def apply_offline_transform(self, ex: Executor | None = None):
        new_scenes = [run_transforms_on_scene(scene, self.offline_transforms, ex) for scene in self.scenes]
        return dataclasses.replace(self, scenes=new_scenes)

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

    def get_scene_by_id(self, scene_id: str) -> Scene[BasicFrame]:
        return self._id_to_scene[scene_id]

    def get_frame_by_id(self, scene_id: str, frame_id: str) -> BasicFrame:
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

    def apply_world_transform_inplace(self, world2new: np.ndarray):
        for s in self.scenes:
            s.apply_world_transform_inplace(world2new)
        return self

    def apply_world_scale_inplace(self, scale: float):
        for s in self.scenes:
            s.apply_world_scale_inplace(scale)
        return self

    def apply_world_transform(self, world2new: np.ndarray):
        new_scenes = [s.apply_world_transform(world2new) for s in self.scenes]
        return dataclasses.replace(self, scenes=new_scenes)

    def apply_world_scale(self, scale: float):
        new_scenes = [s.apply_world_scale(scale) for s in self.scenes]
        return dataclasses.replace(self, scenes=new_scenes)

    def get_online_transformed_frame(self, global_frame_id: tuple[str, str]):
        frame = self.get_frame_by_id(global_frame_id[0], global_frame_id[1])
        return run_online_transforms_on_frame(frame, self.online_transforms)


class SceneTransformOfflineDisk(abc.ABC):
    """scene transform that save output to disk.
    """
    def create_work_dir(self, root: Path, file_key: str = "workdir"):
        qname = get_qualname_of_type(type(self))
        qname = qname.replace(".", "_")
        work_dir = root / f"{qname}_{file_key}"
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(exist_ok=True, parents=True, mode=0o755)
        return work_dir

    def file_flag_exists(self, root: Path, file_key: str = "done") -> bool:
        qname = get_qualname_of_type(type(self))
        qname = qname.replace(".", "_")
        return (root / f"{qname}_{file_key}.json").exists()

    def write_file_flag(self, root: Path, file_key: str = "done", **additional_fields):
        qname = get_qualname_of_type(type(self))
        qname = qname.replace(".", "_")
        with open(root / f"{qname}_{file_key}.json", "w") as f:
            json.dump(additional_fields, f)
import copy
import enum
import d3sim.core.dataclass_dispatch as dataclasses 
import abc 
from typing import Any, Callable, Hashable, Self, Sequence, TypeVar, Generic, overload
import numpy as np 
import torch
from scipy.spatial.transform import Rotation
from d3sim.core.geodef import EulerIntrinsicOrder
from d3sim.core.registry import HashableRegistry 
from pydantic_core import PydanticCustomError, core_schema
from pydantic import (
    GetCoreSchemaHandler, )

from d3sim.core.ops.rotation import euler_from_matrix_np
from d3sim.core.thtools import np_to_torch_dev
from concurrent.futures import Executor, ProcessPoolExecutor

from d3sim.core.unique_tree_id import UniqueTreeId
T = TypeVar("T")


def _d3sim_dict_factory(obj_dict: list[tuple[str, Any]]):
    res: dict[str, Any] = {}
    for k, v in obj_dict:
        if not isinstance(v, ResourceLoader):
            res[k] = v
    return res


def asdict(obj: Any):
    return dataclasses.asdict(obj, dict_factory=_d3sim_dict_factory)


class ResourceLoader(abc.ABC):
    def __init__(self):
        super().__init__()
        self._base_uri_data_cache: dict[str, Any] = {}

    def read(self, resource: "Resource") -> Any:
        if resource.base_uri not in self._base_uri_data_cache:
            self._base_uri_data_cache[resource.base_uri] = self.read_base_uri(resource)
        return self._base_uri_data_cache[resource.base_uri]

    def parse(self, resource: "Resource", base_uri_data: Any) -> Any:
        return base_uri_data

    @abc.abstractmethod
    def read_base_uri(self, resource: "Resource") -> Any:
        raise NotImplementedError

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any,
                                     _handler: GetCoreSchemaHandler):
        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.any_schema(),
        )

    @classmethod
    def validate(cls, v):
        if not isinstance(v, ResourceLoader):
            raise ValueError('ResourceLoader required, but get', type(v))
        return v

class SingleFileLoader(ResourceLoader):
    # for resource that only have single file, not shared
    # common shared loader is video and tfrecord.
    pass

class Undefined:
    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any,
                                     _handler: GetCoreSchemaHandler):
        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.any_schema(),
        )

    @classmethod
    def validate(cls, v):
        if not isinstance(v, Undefined):
            raise ValueError('Undefined required, but get', type(v))
        return v

ALL_RESOURCE_LOADERS = HashableRegistry[type[ResourceLoader]]()

@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class Resource(Generic[T]):
    # different resource instance may share same base_uri, such as
    # video sequence.
    base_uri: str
    loader_type: str
    # used to load resource.
    loader: ResourceLoader | None = None
    _cached_data: T | Undefined = dataclasses.field(default=Undefined())
    # if all resource share single uri file, use this to distinguish
    uid: Hashable = ""

    @property 
    def is_loaded(self) -> bool:
        return not isinstance(self._cached_data, Undefined)

    @property 
    def data(self) -> T:
        if isinstance(self._cached_data, Undefined):
            self.read()
        assert self.is_loaded, "Resource data is not loaded"
        assert not isinstance(self._cached_data, Undefined), "Resource data is not loaded"
        return self._cached_data

    def read(self, cache_data: bool = True) -> T:
        if not isinstance(self._cached_data, Undefined):
            return self._cached_data
        if self.loader is None:
            raise ValueError("Resource loader is not set")
        res = self.loader.parse(self, self.loader.read(self))
        if cache_data:
            self._cached_data = res
        return res 

    def clear_cache(self):
        self._cached_data = Undefined()

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class Pose:
    to_vehicle: np.ndarray = dataclasses.field(default_factory=lambda: np.eye(4))
    to_world: np.ndarray = dataclasses.field(default_factory=lambda: np.eye(4))
    velocity: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))
    scale: float = 1.0
    @property 
    def vehicle_to_world(self):
        sensor_to_world = self.to_world
        sensor_to_vehicle = self.to_vehicle
        return sensor_to_world @ np.linalg.inv(sensor_to_vehicle)

    def apply_world_transform(self, world2new: np.ndarray):
        return dataclasses.replace(self, to_world=world2new @ self.to_world)
   
    def apply_world_transform_inplace(self, world2new: np.ndarray):
        self.to_world = world2new @ self.to_world
        return self

    def apply_world_scale(self, scale: float):
        # for uniform scale, we can scale center part only.
        # non-uniform scale (xyz have different scale) is not supported.
        to_world_scaled = self.to_world.copy()
        to_world_scaled[:3, 3] *= scale
        to_vehicle_scaled = self.to_vehicle.copy()
        to_vehicle_scaled[:3, 3] *= scale
        velo_scaled = self.velocity * scale
        accel_scaled = self.acceleration * scale
        return dataclasses.replace(self, scale=self.scale * scale, to_world=to_world_scaled,
                                   to_vehicle=to_vehicle_scaled, velocity=velo_scaled, acceleration=accel_scaled)

    def apply_world_scale_inplace(self, scale: float):
        self.scale *= scale
        self.to_world[:3, 3] *= scale
        self.to_vehicle[:3, 3] *= scale
        self.velocity *= scale
        self.acceleration *= scale
        return self

    def get_euler_in_vehicle(self, order: EulerIntrinsicOrder = EulerIntrinsicOrder.XYZ) -> np.ndarray:
        r, p, y = euler_from_matrix_np(self.to_vehicle[:3, :3], order)
        return np.array([r, p, y], np.float64)

    def get_euler_in_world(self, order: EulerIntrinsicOrder = EulerIntrinsicOrder.XYZ) -> np.ndarray:
        r, p, y = euler_from_matrix_np(self.to_world[:3, :3], order)
        return np.array([r, p, y], np.float64)

@dataclasses.dataclass(kw_only=True)
class ObjectBase:
    track_id: int | str 
    type: int | str 
    source: str = ""
    track_local_id: int = -1

    def to_dict_no_nested(self):
        res: dict[str, Any] = {}
        for field in dataclasses.fields(self):
            field_data = getattr(self, field.name)
            res[field.name] = field_data
        return res

@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class Object3d(ObjectBase):
    pose: Pose
    size: np.ndarray

    def apply_world_transform(self, world2new: np.ndarray):
        return dataclasses.replace(self, pose=self.pose.apply_world_transform(world2new))

    def apply_world_transform_inplace(self, world2new: np.ndarray):
        self.pose.apply_world_transform_inplace(world2new)
        return self

    def apply_world_scale(self, scale: float):
        return dataclasses.replace(self, pose=self.pose.apply_world_scale(scale), size=self.size * scale)

    def apply_world_scale_inplace(self, scale: float):
        self.pose.apply_world_scale_inplace(scale)
        self.size *= scale
        return self


@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class Object2d(ObjectBase):
    bbox_xywh: np.ndarray # f32
    track_id_3d: int | str | None = None
    track_local_id_3d: int = -1

    def crop(self, crop_xy: tuple[int, int], crop_wh: tuple[int, int]) -> Self | None:
        x, y = crop_xy
        w, h = crop_wh
        rew_xy = np.maximum(self.bbox_xywh[:2], [x, y])
        new_xy_max = np.minimum(self.bbox_xywh[:2] + self.bbox_xywh[2:], [x + w, y + h])
        new_wh = new_xy_max - rew_xy
        if np.any(new_wh <= 0):
            return None
        return dataclasses.replace(self, bbox_xywh=np.concatenate([rew_xy, new_wh]))
    
    def resize(self, scale: float) -> Self:
        new_bbox = self.bbox_xywh.copy()
        new_bbox[:2] *= scale
        new_bbox[2:] *= scale
        return dataclasses.replace(self, bbox_xywh=new_bbox)

class CoordSystem(enum.IntEnum):
    WORLD = 0
    VEHICLE = 1

class CameraFieldTypes(enum.Enum):
    IMAGE = "image"
    SEGMENTATION = "segmentation"
    VALID_MASK = "valid_mask"
    DEPTH = "depth"

class LidarFieldTypes(enum.Enum):
    POINT_CLOUD = "point_cloud"
    SEGMENTATION = "segmentation"
    INTENSITY = "intensity"
    VALID_MASK = "valid_mask"
    TIMESTAMP = "timestamp"

class SceneFieldTypes(enum.Enum):
    COLMAP_POINT_CLOUD = "colmap_point_cloud"


T_field_type = TypeVar("T_field_type", CameraFieldTypes, LidarFieldTypes)

@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class Sensor(Generic[T_field_type]):
    id: str 
    timestamp: int # ns
    pose: Pose
    # if not None, indicate the coordinate system of sensor data (e.g. pointcloud)
    data_coord: CoordSystem | None = None
    frame_local_id: int | None = None
    scene_local_id: int | None = None
    # np field shouldn't support cache
    fields: dict[str | T_field_type, np.ndarray | Resource[np.ndarray]] = dataclasses.field(default_factory=dict)
    _cached_torch_fields: dict[str | T_field_type, torch.Tensor] = dataclasses.field(default_factory=dict)
    _global_id: str | None = None
    scene_uri: str | None = None

    @property 
    def global_id(self):
        assert self._global_id is not None, "you must save frame in a scene to generate global id"
        return self._global_id

    @property 
    def frame_id(self):
        return UniqueTreeId(self.global_id).parts[1]

    @property 
    def scene_id(self):
        return UniqueTreeId(self.global_id).parts[0]

    def apply_world_transform(self, world2new: np.ndarray):
        new_pose = self.pose.apply_world_transform(world2new)
        return self.replace_with_cache_cleared(pose=new_pose)

    def apply_world_transform_inplace(self, world2new: np.ndarray):
        self.pose.apply_world_transform_inplace(world2new)
        return self

    def apply_world_scale(self, scale: float):
        new_pose = self.pose.apply_world_scale(scale)
        return self.replace_with_cache_cleared(pose=new_pose)

    def apply_world_scale_inplace(self, scale: float):
        self.pose.apply_world_scale_inplace(scale)
        return self

    def to_dict_no_nested(self):
        res: dict[str, Any] = {}
        for field in dataclasses.fields(self):
            field_data = getattr(self, field.name)
            res[field.name] = field_data
        return res

    def has_field(self, field: T_field_type | str) -> bool:
        return field in self.fields

    def set_field_np(self, field: T_field_type | str, data: np.ndarray):
        assert isinstance(data, np.ndarray)
        self.fields[field] = data

    def set_field_torch(self, field: T_field_type | str, data: torch.Tensor):
        assert isinstance(data, torch.Tensor)
        self._cached_torch_fields[field] = data

    def get_field_np(self, field: T_field_type | str) -> np.ndarray | None:
        if field not in self.fields:
            return None
        res = self.fields[field]
        if isinstance(res, Resource):
            return res.data
        return res

    def get_field_np_required(self, field: T_field_type | str) -> np.ndarray:
        res = self.get_field_np(field)
        if res is None:
            raise ValueError(f"Field {field} not found")
        return res

    def has_field_torch(self, field: T_field_type | str) -> bool:
        return field in self._cached_torch_fields

    def get_field_torch(self, field: T_field_type | str, device: str | torch.device | None = None) -> torch.Tensor | None:
        if field in self._cached_torch_fields:
            return self._cached_torch_fields[field]
        npf = self.get_field_np(field)
        if npf is None:
            return None
        return np_to_torch_dev(npf, device)

    def get_field_torch_cached(self, field: T_field_type | str, device: str | torch.device | None = None) -> torch.Tensor | None:
        thf = self.get_field_torch(field, device)
        if thf is None:
            return None 
        if field not in self._cached_torch_fields:
            self._cached_torch_fields[field] = thf
        return self._cached_torch_fields[field]

    def remove_field_np(self, field: T_field_type | str):
        if field in self.fields:
            self.fields.pop(field)

    def remove_field_torch(self, field: T_field_type | str):
        if field in self._cached_torch_fields:
            self._cached_torch_fields.pop(field)

    def clear_torch_cache(self):
        self._cached_torch_fields.clear()

    def shallow_copy(self):
        return dataclasses.replace(self, fields=self.fields.copy(), _cached_torch_fields=self._cached_torch_fields.copy())

    def replace(self, /, **changes: Any) -> Self:
        return dataclasses.replace(self, fields=self.fields.copy(), _cached_torch_fields=self._cached_torch_fields.copy(), **changes)

    def replace_with_cache_cleared(self, /, **changes: Any) -> Self:
        return dataclasses.replace(self, fields=self.fields.copy(), _cached_torch_fields={}, **changes)

T_sensor = TypeVar("T_sensor", bound=Sensor)

@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class BaseCamera(Sensor[CameraFieldTypes], abc.ABC):
    objects: list[Object2d] = dataclasses.field(default_factory=list)

@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class BaseLidar(Sensor[LidarFieldTypes], abc.ABC):
    pass 

@dataclasses.dataclass
class BaseFrame:
    id: str
    timestamp: int
    pose: Pose
    sensors: list[Sensor] = dataclasses.field(default_factory=list)
    objects: list[Object3d] = dataclasses.field(default_factory=list)
    local_id: int = -1
    _global_id: str | None = None
    scene_uri: str | None = None
    _user_datas: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property 
    def global_id(self):
        assert self._global_id is not None, "you must save frame in a scene to generate global id"
        return self._global_id

    @property 
    def scene_id(self):
        return UniqueTreeId(self.global_id).parts[0]

    def get_objects_by_source(self, source: str = "") -> list[Object3d]:
        res: list[Object3d] = []
        for obj in self.objects:
            if obj.source == source:
                res.append(obj)
        return res

    def apply_world_transform(self, world2new: np.ndarray):
        new_pose = self.pose.apply_world_transform(world2new)
        new_sensors = [s.apply_world_transform(world2new) for s in self.sensors]
        new_objs = [o.apply_world_transform(world2new) for o in self.objects]
        return dataclasses.replace(self, pose=new_pose, sensors=new_sensors, objects=new_objs)

    def apply_world_transform_inplace(self, world2new: np.ndarray):
        self.pose.apply_world_transform_inplace(world2new)
        for s in self.sensors:
            s.apply_world_transform_inplace(world2new)
        for obj in self.objects:
            obj.apply_world_transform_inplace(world2new)
        return self

    def apply_world_scale(self, scale: float):
        new_pose = self.pose.apply_world_scale(scale)
        new_sensors = [s.apply_world_scale(scale) for s in self.sensors]
        new_objs = [o.apply_world_scale(scale) for o in self.objects]
        return dataclasses.replace(self, pose=new_pose, sensors=new_sensors, objects=new_objs)
        
    def apply_world_scale_inplace(self, scale: float):
        self.pose.apply_world_scale_inplace(scale)
        for s in self.sensors:
            s.apply_world_scale_inplace(scale)
        for obj in self.objects:
            obj.apply_world_scale_inplace(scale)
        return self

    def release_all_loaders(self):
        for sensor in self.sensors:
            for field in dataclasses.fields(sensor):
                field_data = getattr(sensor, field.name)
                if isinstance(field_data, Resource):
                    field_data.loader = None
            # for field, resource in sensor.fields.items():
            #     resource.loader = None

    def release_all_resource_caches(self):
        for sensor in self.sensors:
            for field in dataclasses.fields(sensor):
                field_data = getattr(sensor, field.name)
                if isinstance(field_data, Resource):
                    field_data.clear_cache()
            # for field, resource in sensor.fields.items():
            #     resource.clear_cache()

    def get_sensor_by_id(self, sensor_id: str) -> Sensor:
        for sensor in self.sensors:
            if sensor.id == sensor_id:
                return sensor
        raise ValueError(f"Sensor {sensor_id} not found")

    def get_sensor_by_id_checked(self, sensor_id: str, sensor_type: type[T]) -> T:
        for sensor in self.sensors:
            if sensor.id == sensor_id and isinstance(sensor, sensor_type):
                return sensor
        raise ValueError(f"Sensor {sensor_id} with type {sensor_type} not found")

    def get_sensors_by_type(self, sensor_type: type[T_sensor]) -> list[T_sensor]:
        res: list[T_sensor] = []
        for sensor in self.sensors:
            if isinstance(sensor, sensor_type):
                res.append(sensor)
        return res 

    def get_transform_matrix(self, src: str | CoordSystem, dst: str | CoordSystem) -> np.ndarray:
        # if src/dst isn't CoordSystem, it's sensor id
        if src == dst:
            return np.eye(4)
        if src == CoordSystem.WORLD:
            if dst == CoordSystem.VEHICLE:
                return np.linalg.inv(self.pose.to_world)
            elif dst == CoordSystem.WORLD:
                return np.eye(4)
            else:
                # dst is sensor
                dst_sensor = self.get_sensor_by_id(dst)
                dst_to_world = dst_sensor.pose.to_world
                return np.linalg.inv(dst_to_world)
        elif src == CoordSystem.VEHICLE:
            if dst == CoordSystem.WORLD:
                return self.pose.to_world.copy()
            else:
                # dst is sensor, use recursive
                mat_inv = self.get_transform_matrix(dst, CoordSystem.VEHICLE)
                return np.linalg.inv(mat_inv)
        else:
            # src is sensor
            mat_world_src = self.get_transform_matrix(CoordSystem.WORLD, src)
            mat_world_dst = self.get_transform_matrix(CoordSystem.WORLD, dst)
            return mat_world_dst @ np.linalg.inv(mat_world_src)

    def to_dict_no_nested(self):
        res: dict[str, Any] = {}
        for field in dataclasses.fields(self):
            field_data = getattr(self, field.name)
            res[field.name] = field_data
        return res

    def override_sensor(self, sensor: Sensor):
        new_sensors = self.sensors.copy()
        for i, s in enumerate(self.sensors):
            if s.id == sensor.id:
                new_sensors[i] = sensor
                break
        return dataclasses.replace(self, sensors=new_sensors)

    def override_sensors(self, sensors: Sequence[Sensor]):
        new_sensors = self.sensors.copy()
        new_sensor_id_to_index = {s.id: i for i, s in enumerate(new_sensors)}
        for i, s in enumerate(sensors):
            if s.id in new_sensor_id_to_index:
                new_sensors[new_sensor_id_to_index[s.id]] = s
                break
        return dataclasses.replace(self, sensors=new_sensors)

    def replace_objects(self, objects: Sequence[Object3d]):
        return dataclasses.replace(self, objects=objects)

T_frame = TypeVar("T_frame", bound=BaseFrame)

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class Scene(Generic[T_frame]):
    id: str 
    frames: list[T_frame]
    pose: Pose = dataclasses.field(default_factory=Pose)

    fields: dict[SceneFieldTypes | str, np.ndarray | Resource[np.ndarray]] = dataclasses.field(default_factory=dict)
    scene_local_id: int | None = None
    uri: str | None = None

    _frame_id_to_frame: dict[str, T_frame] = dataclasses.field(default_factory=dict)
    # should only be used on transforms
    _user_datas: dict[str, Any] = dataclasses.field(default_factory=dict)
    def __post_init__(self):
        # create resource loaders
        self.init_loaders()
        self.assign_frame_local_id_inplace()
        self.assign_track_local_id_inplace()
        self.assign_global_id_and_uri_inplace()
        for frame in self.frames:
            self._frame_id_to_frame[frame.id] = frame

    def get_frame_by_id(self, frame_id: str) -> T_frame:
        return self._frame_id_to_frame[frame_id]

    def set_user_data(self, key: str, value: Any):
        self._user_datas[key] = value

    def get_user_data_type_checked(self, key: str, expected_type: type[T]) -> T:
        res = self._user_datas["key"]
        if not isinstance(res, expected_type):
            raise ValueError(f"User data {key} is not of type {expected_type}")
        return res

    def init_loaders(self):
        base_uri_type_to_loader: dict[tuple[str, str, Hashable], ResourceLoader] = {}
        for frame in self.frames:
            for sensor in frame.sensors:
                for field in dataclasses.fields(sensor):
                    field_data = getattr(sensor, field.name)
                    if isinstance(field_data, Resource):
                        resource: Resource = field_data
                        key = (resource.base_uri, resource.loader_type, resource.uid)
                        if key not in base_uri_type_to_loader:
                            base_uri_type_to_loader[key] = ALL_RESOURCE_LOADERS[resource.loader_type]()
                        if resource.loader is None:
                            loader = base_uri_type_to_loader[key]
                            resource.loader = loader
                for field, resource_or_arr in sensor.fields.items():
                    if isinstance(resource_or_arr, Resource):
                        resource: Resource = resource_or_arr
                        key = (resource.base_uri, resource.loader_type, resource.uid)
                        if key not in base_uri_type_to_loader:
                            base_uri_type_to_loader[key] = ALL_RESOURCE_LOADERS[resource.loader_type]()
                        if resource.loader is None:
                            loader = base_uri_type_to_loader[key]
                            resource.loader = loader
        for field, resource_or_arr in self.fields.items():
            if isinstance(resource_or_arr, Resource):
                resource: Resource = resource_or_arr
                key = (resource.base_uri, resource.loader_type, resource.uid)
                if key not in base_uri_type_to_loader:
                    base_uri_type_to_loader[key] = ALL_RESOURCE_LOADERS[resource.loader_type]()
                if resource.loader is None:
                    loader = base_uri_type_to_loader[key]
                    resource.loader = loader
        
    def release_all_loaders(self):
        for frame in self.frames:
            frame.release_all_loaders()

    def release_all_resource_caches(self):
        for frame in self.frames:
            frame.release_all_resource_caches()

    def apply_world_transform(self, world2new: np.ndarray):
        new_frames = [f.apply_world_transform(world2new) for f in self.frames]
        return dataclasses.replace(self, frames=new_frames)

    def apply_world_transform_inplace(self, world2new: np.ndarray):
        for f in self.frames:
            f.apply_world_transform_inplace(world2new)
        return self

    def apply_world_scale(self, scale: float):
        new_frames = [f.apply_world_scale(scale) for f in self.frames]
        new_pose = self.pose.apply_world_scale(scale)
        return dataclasses.replace(self, frames=new_frames, pose=new_pose)

    def apply_world_scale_inplace(self, scale: float):
        for f in self.frames:
            f.apply_world_scale_inplace(scale)
        self.pose.apply_world_scale_inplace(scale)
        return self

    def assign_frame_local_id_inplace(self):
        for i, frame in enumerate(self.frames):
            frame.local_id = i
            for sensor in frame.sensors:
                sensor.frame_local_id = i
                sensor.scene_local_id = self.scene_local_id

    def assign_scene_local_id_inplace(self, scene_local_id: int):
        self.scene_local_id = scene_local_id
        for i, frame in enumerate(self.frames):
            frame.local_id = i
            for sensor in frame.sensors:
                sensor.scene_local_id = scene_local_id

    def assign_global_id_and_uri_inplace(self):
        for frame in self.frames:
            frame_uid = UniqueTreeId.from_parts([self.id, frame.id])
            frame._global_id = frame_uid.uid_encoded
            frame.scene_uri = self.uri
            for sensor in frame.sensors:
                sensor_uid = UniqueTreeId.from_parts([self.id, frame.id, sensor.id])
                sensor._global_id = sensor_uid.uid_encoded
                sensor.scene_uri = self.uri

    def assign_track_local_id_inplace(self):
        track_id_str_to_local_id: dict[str | int, int] = {}
        cam_track_id_str_to_local_id: dict[str | int, int] = {}
        for i, frame in enumerate(self.frames):
            for obj in frame.objects:
                if obj.track_id not in track_id_str_to_local_id:
                    track_id_str_to_local_id[obj.track_id] = len(track_id_str_to_local_id)
                obj.track_local_id = track_id_str_to_local_id[obj.track_id]
        for i, frame in enumerate(self.frames):
            for cam in frame.get_sensors_by_type(BaseCamera):
                for obj in cam.objects:
                    if obj.track_id not in cam_track_id_str_to_local_id:
                        cam_track_id_str_to_local_id[obj.track_id] = len(cam_track_id_str_to_local_id)
                    obj.track_local_id = cam_track_id_str_to_local_id[obj.track_id]
                    if obj.track_id_3d is not None and obj.track_id_3d in track_id_str_to_local_id:
                        obj.track_local_id_3d = track_id_str_to_local_id[obj.track_id_3d]

    def get_track_local_id_to_frame_local_ids(self):
        track_local_id_to_frame_ids: dict[int, list[int]] = {}
        for i, frame in enumerate(self.frames):
            for obj in frame.objects:
                if obj.track_local_id not in track_local_id_to_frame_ids:
                    track_local_id_to_frame_ids[obj.track_local_id] = []
                track_local_id_to_frame_ids[obj.track_local_id].append(i)
        return track_local_id_to_frame_ids

    def get_frames_by_frame_local_ids(self, frame_local_ids: list[int]):
        return [self.frames[i] for i in frame_local_ids]

    def get_sensors_by_type(self, sensor_type: type[T_sensor]) -> list[T_sensor]:
        res: list[T_sensor] = []
        for frame in self.frames:
            res.extend(frame.get_sensors_by_type(sensor_type))
        return res

    def get_sensor_id_to_sensors(self, sensor_type: type[T_sensor]) -> dict[str, list[T_sensor]]:
        res: dict[str, list[T_sensor]] = {}
        for frame in self.frames:
            for sensor in frame.get_sensors_by_type(sensor_type):
                if sensor.id not in res:
                    res[sensor.id] = []
                res[sensor.id].append(sensor)
        return res

class DistortType(enum.IntEnum):
    kNone = 0
    kOpencvPinhole = 1
    kOpencvPinholeWaymo = 2



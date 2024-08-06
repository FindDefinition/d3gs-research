import enum
from pathlib import Path
import time
from typing import Any
from typing_extensions import override

import torch
from d3sim.constants import D3SIM_DEFAULT_DEVICE
import d3sim.core.dataclass_dispatch as dataclasses

from d3sim.data.scene_def.base import CoordSystem, Object2d, Object3d, Pose, Scene, Sensor, Resource, BaseFrame, ALL_RESOURCE_LOADERS, ResourceLoader
from d3sim.data.scene_def.camera import BasicCamera, BasicPinholeCamera
import dask.dataframe as dd
import numpy as np
from d3sim.data.datasets.waymo.ops import range_image_to_point_cloud, range_image_to_point_cloud_v2
from d3sim.data.scene_def.frame import BasicFrame
from d3sim.data.scene_def.lidar import BasicLidar
from cumm import tensorview as tv 
import cv2

from d3sim.core.ops.rotation import get_rotation_matrix_np

class CameraName(enum.IntEnum):
    """Name of camera."""
    UNKNOWN = 0
    FRONT = 1
    FRONT_LEFT = 2
    FRONT_RIGHT = 3
    SIDE_LEFT = 4
    SIDE_RIGHT = 5


class LaserName(enum.IntEnum):
    """Name of laser/LiDAR."""
    UNKNOWN = 0
    TOP = 1
    FRONT = 2
    SIDE_LEFT = 3
    SIDE_RIGHT = 4
    REAR = 5

class CameraSegType(enum.IntEnum):
    # Anything that does not fit the other classes or is too ambiguous to
    # label.
    TYPE_UNDEFINED = 0
    # The Waymo vehicle.
    TYPE_EGO_VEHICLE = 1
    # Small vehicle such as a sedan, SUV, pickup truck, minivan or golf cart.
    TYPE_CAR = 2
    # Large vehicle that carries cargo.
    TYPE_TRUCK = 3
    # Large vehicle that carries more than 8 passengers.
    TYPE_BUS = 4
    # Large vehicle that is not a truck or a bus.
    TYPE_OTHER_LARGE_VEHICLE = 5
    # Bicycle with no rider.
    TYPE_BICYCLE = 6
    # Motorcycle with no rider.
    TYPE_MOTORCYCLE = 7
    # Trailer attached to another vehicle or horse.
    TYPE_TRAILER = 8
    # Pedestrian. Does not include objects associated with the pedestrian, such
    # as suitcases, strollers or cars.
    TYPE_PEDESTRIAN = 9
    # Bicycle with rider.
    TYPE_CYCLIST = 10
    # Motorcycle with rider.
    TYPE_MOTORCYCLIST = 11
    # Birds, including ones on the ground.
    TYPE_BIRD = 12
    # Animal on the ground such as a dog, cat, cow, etc.
    TYPE_GROUND_ANIMAL = 13
    # Cone or short pole related to construction.
    TYPE_CONSTRUCTION_CONE_POLE = 14
    # Permanent horizontal and vertical lamp pole, traffic sign pole, etc.
    TYPE_POLE = 15
    # Large object carried/pushed/dragged by a pedestrian.
    TYPE_PEDESTRIAN_OBJECT = 16
    # Sign related to traffic, including front and back facing signs.
    TYPE_SIGN = 17
    # The box that contains traffic lights regardless of front or back facing.
    TYPE_TRAFFIC_LIGHT = 18
    # Permanent building and walls, including solid fences.
    TYPE_BUILDING = 19
    # Drivable road with proper markings, including parking lots and gas
    # stations.
    TYPE_ROAD = 20
    # Marking on the road that is parallel to the ego vehicle and defines
    # lanes.
    TYPE_LANE_MARKER = 21
    # All markings on the road other than lane markers.
    TYPE_ROAD_MARKER = 22
    # Paved walkable surface for pedestrians, including curbs.
    TYPE_SIDEWALK = 23
    # Vegetation including tree trunks, tree branches, bushes, tall grasses,
    # flowers and so on.
    TYPE_VEGETATION = 24
    # The sky, including clouds.
    TYPE_SKY = 25
    # Other horizontal surfaces that are drivable or walkable.
    TYPE_GROUND = 26
    # Object that is not permanent in its current position and does not belong
    # to any of the above classes.
    TYPE_DYNAMIC = 27
    # Object that is permanent in its current position and does not belong to
    # any of the above classes.
    TYPE_STATIC = 28

class PointSegType(enum.IntEnum):
    TYPE_UNDEFINED = 0
    TYPE_CAR = 1
    TYPE_TRUCK = 2
    TYPE_BUS = 3
    # Other small vehicles (e.g. pedicab) and large vehicles (e.g. construction
    # vehicles, RV, limo, tram).
    TYPE_OTHER_VEHICLE = 4
    TYPE_MOTORCYCLIST = 5
    TYPE_BICYCLIST = 6
    TYPE_PEDESTRIAN = 7
    TYPE_SIGN = 8
    TYPE_TRAFFIC_LIGHT = 9
    # Lamp post, traffic sign pole etc.
    TYPE_POLE = 10
    # Construction cone/pole.
    TYPE_CONSTRUCTION_CONE = 11
    TYPE_BICYCLE = 12
    TYPE_MOTORCYCLE = 13
    TYPE_BUILDING = 14
    # Bushes, tree branches, tall grasses, flowers etc.
    TYPE_VEGETATION = 15
    TYPE_TREE_TRUNK = 16
    # Curb on the edge of roads. This does not include road boundaries if
    # there’s no curb.
    TYPE_CURB = 17
    # Surface a vehicle could drive on. This include the driveway connecting
    # parking lot and road over a section of sidewalk.
    TYPE_ROAD = 18
    # Marking on the road that’s specifically for defining lanes such as
    # single/double white/yellow lines.
    TYPE_LANE_MARKER = 19
    # Marking on the road other than lane markers, bumps, cateyes, railtracks
    # etc.
    TYPE_OTHER_GROUND = 20
    # Most horizontal surface that’s not drivable, e.g. grassy hill,
    # pedestrian walkway stairs etc.
    TYPE_WALKABLE = 21
    # Nicely paved walkable surface when pedestrians most likely to walk on.
    TYPE_SIDEWALK = 22



@dataclasses.dataclass(kw_only=True)
class WaymoImageResource(Resource):
    image_name: CameraName
    frame_timestamp_micros: int


@dataclasses.dataclass(kw_only=True)
class WaymoLidarResource(Resource):
    laser_name: LaserName
    frame_timestamp_micros: int

@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class WaymoCamera(BasicPinholeCamera):
    seg_label_rc: Resource[np.ndarray | None]
    _seg_cache: np.ndarray | None = None

    def _load_seg(self):
        if self._seg_cache is not None:
            return self._seg_cache
        segs = self.seg_label_rc.data
        if segs is None:
            return None
        return segs

    @property
    def segmentation(self):
        return self._load_seg()

@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class WaymoLidar(BasicLidar):
    range_images_rc: Resource[tuple[np.ndarray, np.ndarray]]
    point_pose_rc: Resource[np.ndarray]
    seg_label_rc: Resource[tuple[np.ndarray | None, np.ndarray | None]]

    beam_inclination: np.ndarray
    # xyz, inten, mask
    _point_load_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    _seg_cache: np.ndarray | None = None

    _tmp_pixel_pose_transform: np.ndarray = dataclasses.field(default_factory=lambda: np.eye(4))

    @override
    def apply_world_transform_inplace(self, world2new: np.ndarray):
        super().apply_world_transform_inplace(world2new)
        self._tmp_pixel_pose_transform = world2new @ self._tmp_pixel_pose_transform
        return self

    def _load_xyz_inten_mask(self):
        enable = False
        if self._point_load_cache is not None:
            return self._point_load_cache
        with tv.measure_and_print_cpu("load_xyz_inten_mask", enable=enable):
            veh_to_world = torch.from_numpy(self.pose.vehicle_to_world.astype(np.float32)).to(D3SIM_DEFAULT_DEVICE)
            extrinsic = torch.from_numpy(self.pose.to_vehicle.astype(np.float32)).to(D3SIM_DEFAULT_DEVICE)
            ris = self.range_images_rc.data 
            pixel_pose_data = self.point_pose_rc.data
        with tv.measure_and_print_cpu("read rg", enable=enable):
            inclination = torch.from_numpy(self.beam_inclination).to(D3SIM_DEFAULT_DEVICE)
            ri_0 = torch.from_numpy(ris[0]).to(D3SIM_DEFAULT_DEVICE)
            ri_1 = torch.from_numpy(ris[1]).to(D3SIM_DEFAULT_DEVICE)
            pixel_pose: torch.Tensor | None = None
            if self.id == f"lidar_{LaserName.TOP}": # TOP
                pixel_pose = torch.from_numpy(pixel_pose_data).to(D3SIM_DEFAULT_DEVICE)
            # breakpoint()
            pixel_tr_np = self._tmp_pixel_pose_transform.astype(np.float32)
        with tv.measure_and_print_cpu("range_image_to_point_cloud", enable=enable):
            res_0, res_0_mask = range_image_to_point_cloud(ri_0, extrinsic, inclination, pixel_pose, veh_to_world, pixel_tr_np)
            res_1, res_1_mask = range_image_to_point_cloud(ri_1, extrinsic, inclination, pixel_pose, veh_to_world, pixel_tr_np)
        with tv.measure_and_print_cpu("post", enable=enable):
            res = torch.cat([res_0, res_1], dim=0).cpu().numpy()
            inten_0 = ri_0[..., 1].reshape(-1)
            inten_1 = ri_1[..., 1].reshape(-1)
            inten = torch.cat([inten_0, inten_1], dim=0).cpu().numpy()
            mask = torch.cat([res_0_mask, res_1_mask], dim=0).cpu().numpy()
        # breakpoint()
        res_all = res, inten, mask
        self._point_load_cache = res_all
        return res_all

    def _load_xyz_inten_mask_v2(self):
        enable = False
        if self._point_load_cache is not None:
            return self._point_load_cache
        with tv.measure_and_print_cpu("load_xyz_inten_mask", enable=enable):
            ris = self.range_images_rc.data 
            pixel_pose_data = self.point_pose_rc.data
        with tv.measure_and_print_cpu("read rg", enable=enable):
            # veh_to_world = torch.from_numpy(self.pose.vehicle_to_world.astype(np.float32)).to(D3SIM_DEFAULT_DEVICE)
            # extrinsic = torch.from_numpy(self.pose.to_vehicle.astype(np.float32)).to(D3SIM_DEFAULT_DEVICE)

            inclination = torch.from_numpy(self.beam_inclination).to(D3SIM_DEFAULT_DEVICE)
            # ri_0 = torch.from_numpy(ris[0]).to(D3SIM_DEFAULT_DEVICE)
            # ri_1 = torch.from_numpy(ris[1]).to(D3SIM_DEFAULT_DEVICE)
            ris = np.stack([ris[0], ris[1]], axis=0)
            ris_th = torch.from_numpy(ris).to(D3SIM_DEFAULT_DEVICE)
            # ris_th = torch.stack([ri_0, ri_1], dim=0)

            pixel_pose: torch.Tensor | None = None
            if self.id == f"lidar_{LaserName.TOP}": # TOP
                pixel_pose = torch.from_numpy(pixel_pose_data).to(D3SIM_DEFAULT_DEVICE)
            # breakpoint()
            pixel_tr_np = self._tmp_pixel_pose_transform.astype(np.float32)
        with tv.measure_and_print_cpu("range_image_to_point_cloud", enable=enable):
            res, mask = range_image_to_point_cloud_v2(ris_th, self.pose.to_vehicle, 
                inclination, pixel_pose, self.pose.vehicle_to_world, pixel_tr_np)
            # res_1, mask = range_image_to_point_cloud_v2(ris_th[1], self.pose.to_vehicle, 
            #     inclination, pixel_pose, veh_to_world, pixel_tr_np)

            # res_0, res_0_mask = range_image_to_point_cloud(ri_0, extrinsic, inclination, pixel_pose, veh_to_world, pixel_tr_np)
            # res_1, res_1_mask = range_image_to_point_cloud(ri_1, extrinsic, inclination, pixel_pose, veh_to_world, pixel_tr_np)
            # res = torch.cat([res_0, res_1], dim=0)
            # res_0, res_0_mask = range_image_to_point_cloud(ri_0, extrinsic, inclination, pixel_pose, veh_to_world, pixel_tr_np)
            # res_1, res_1_mask = range_image_to_point_cloud(ri_1, extrinsic, inclination, pixel_pose, veh_to_world, pixel_tr_np)
        with tv.measure_and_print_cpu("post", enable=enable):
            # resx = torch.cat([res_0, res_1], dim=0)
            # print(torch.linalg.norm(resx - res))
            # breakpoint()
            # inten_0 = ri_0[..., 1].reshape(-1)
            # inten_1 = ri_1[..., 1].reshape(-1)
            # inten = torch.cat([inten_0, inten_1], dim=0).cpu().numpy()
            inten = ris[..., 1].reshape(-1)
            # mask = torch.cat([res_0_mask, res_1_mask], dim=0).cpu().numpy()
            res_all = res.cpu().numpy(), inten, mask.cpu().numpy()
        self._point_load_cache = res_all
        return res_all

    def _load_seg(self):
        if self._seg_cache is not None:
            return self._seg_cache
        segs = self.seg_label_rc.data
        if segs[0] is None or segs[1] is None:
            return None 
        seg_0 = segs[0].reshape(-1, 2)[:, 1]
        seg_1 = segs[1].reshape(-1, 2)[:, 1]
        seg = np.concatenate([seg_0, seg_1], axis=-1)
        self._seg_cache = seg
        return seg

    @property 
    def xyz(self):
        return self._load_xyz_inten_mask_v2()[0]

    @property
    def intensity(self):
        return self._load_xyz_inten_mask_v2()[1]

    @property
    def mask(self):
        return self._load_xyz_inten_mask_v2()[2]

    @property
    def segmentation(self):
        return self._load_seg()


@ALL_RESOURCE_LOADERS.register_no_key
class WaymoImageLoader(ResourceLoader):
    def read_base_uri(self, resource: Resource):
        assert isinstance(resource, WaymoImageResource)
        data = dd.read_parquet(resource.base_uri,
                               columns=[
                                   'key.frame_timestamp_micros',
                                   'key.camera_name',
                                   '[CameraImageComponent].image'
                               ],
                               filters=[('key.frame_timestamp_micros', '==',
                                         resource.frame_timestamp_micros),
                                        ('key.camera_name', '==',
                                         resource.image_name.value)])
        img_jpeg_bytes = data.compute()['[CameraImageComponent].image'].iloc[0]
        img_decoded = cv2.imdecode(np.frombuffer(img_jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        return img_decoded

@ALL_RESOURCE_LOADERS.register_no_key
class WaymoImageSegLoader(ResourceLoader):
    def read_base_uri(self, resource: Resource):
        assert isinstance(resource, WaymoImageResource)
        data = dd.read_parquet(resource.base_uri,
                               columns=[
                                   'key.frame_timestamp_micros',
                                   'key.camera_name',
                                   '[CameraSegmentationLabelComponent].panoptic_label',
                                   '[CameraSegmentationLabelComponent].panoptic_label_divisor',
                               ],
                               filters=[('key.frame_timestamp_micros', '==',
                                         resource.frame_timestamp_micros),
                                        ('key.camera_name', '==',
                                         resource.image_name.value)])
        if data.compute()['[CameraSegmentationLabelComponent].panoptic_label'].shape[0] == 0:
            return None
        img_png_bytes = data.compute()['[CameraSegmentationLabelComponent].panoptic_label'].iloc[0]
        img_decoded = cv2.imdecode(np.frombuffer(img_png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        dividor = data.compute()['[CameraSegmentationLabelComponent].panoptic_label_divisor'].iloc[0]
        return img_decoded // dividor

class _WaymoLidarLoaderBase(ResourceLoader):
    def __init__(self, key: str, store_whole_seq: bool = False, is_pair: bool = True):
        super().__init__()
        self.key = key
        self.store_whole_seq = store_whole_seq
        self.is_pair = is_pair

    def parse(self, resource: Resource, base_uri_data: Any):
        key = self.key
        if not self.store_whole_seq:
            return super().parse(resource, base_uri_data)
        assert isinstance(resource, WaymoLidarResource)
        ff = base_uri_data["key.frame_timestamp_micros"] == resource.frame_timestamp_micros
        ff = ff & base_uri_data["key.laser_name"] == resource.laser_name.value
        data = base_uri_data[ff]
        if len(data[
            f'[{key}].range_image_return1.values']) == 0:
            if self.is_pair:
                return None, None
            else:
                return None 
        range_image_return1 = data[
            f'[{key}].range_image_return1.values'].iloc[0]
        range_image_return1_shape = data[
            f'[{key}].range_image_return1.shape'].iloc[0]
        range_image_return1 = range_image_return1.reshape(
            range_image_return1_shape)
        # print(self.key)
        # breakpoint()
        if self.is_pair:
            range_image_return2 = data[
                f'[{key}].range_image_return2.values'].iloc[0]
            range_image_return2_shape = data[
                f'[{key}].range_image_return2.shape'].iloc[0]
            range_image_return2 = range_image_return2.reshape(
                range_image_return2_shape)
            return (range_image_return1, range_image_return2)
        else:
            return range_image_return1

    def read_base_uri(self, resource: Resource):
        assert isinstance(resource, WaymoLidarResource)
        key = self.key
        columns = [
                    'key.frame_timestamp_micros',
                    'key.laser_name',
                    f'[{key}].range_image_return1.shape',
                    f'[{key}].range_image_return1.values',
                ]
        if self.is_pair:
            columns.extend([f'[{key}].range_image_return2.shape',
                    f'[{key}].range_image_return2.values',])
        if self.store_whole_seq:
            data = dd.read_parquet(resource.base_uri, columns=columns)
            # print(resource.base_uri)
            
            # breakpoint()
            return data.compute()

        data = dd.read_parquet(
            resource.base_uri,
            columns=columns,
            filters=[('key.frame_timestamp_micros', '==',
                      resource.frame_timestamp_micros),
                     ('key.laser_name', '==', resource.laser_name.value)
                     ])
        if len(data[
            f'[{key}].range_image_return1.values'].compute()) == 0:
            if self.is_pair:
                return None, None
            else:
                return None 
        range_image_return1 = data[
            f'[{key}].range_image_return1.values'].compute().iloc[0]
        range_image_return1_shape = data[
            f'[{key}].range_image_return1.shape'].compute().iloc[0]
        range_image_return1 = range_image_return1.reshape(
            range_image_return1_shape)

        if self.is_pair:
            range_image_return2 = data[
                f'[{key}].range_image_return2.values'].compute().iloc[0]
            range_image_return2_shape = data[
                f'[{key}].range_image_return2.shape'].compute().iloc[0]
            range_image_return2 = range_image_return2.reshape(
                range_image_return2_shape)
            return (range_image_return1, range_image_return2)
        else:
            return range_image_return1

@ALL_RESOURCE_LOADERS.register_no_key
class WaymoLidarLoader(_WaymoLidarLoaderBase):
    def __init__(self):
        super().__init__('LiDARComponent')

@ALL_RESOURCE_LOADERS.register_no_key
class WaymoLidarSegLoader(_WaymoLidarLoaderBase):
    def __init__(self):
        super().__init__('LiDARSegmentationLabelComponent')

@ALL_RESOURCE_LOADERS.register_no_key
class WaymoLidarWholeStorageLoader(_WaymoLidarLoaderBase):
    def __init__(self):
        super().__init__('LiDARComponent', True)

@ALL_RESOURCE_LOADERS.register_no_key
class WaymoLidarPoseLoader(_WaymoLidarLoaderBase):
    def __init__(self):
        super().__init__('LiDARPoseComponent', is_pair=False)

@ALL_RESOURCE_LOADERS.register_no_key
class WaymoLidarWholeStoragePoseLoader(_WaymoLidarLoaderBase):
    def __init__(self):
        super().__init__('LiDARPoseComponent', True, is_pair=False)

@ALL_RESOURCE_LOADERS.register_no_key
class WaymoLidarWholeStorageSegLoader(_WaymoLidarLoaderBase):
    def __init__(self):
        super().__init__('LiDARSegmentationLabelComponent', True)


def _read_all(scene_id: str, folder: str, tag: str) -> dd.DataFrame:
    path = Path(f"{folder}/{tag}/{scene_id}.parquet")
    return dd.read_parquet(path)


def _read_camera_metadata(scene_id: str, folder: str):
    path = Path(f"{folder}/camera_image/{scene_id}.parquet")

    metadata = dd.read_parquet(
        path,
        columns=[
            'key.frame_timestamp_micros', 'key.camera_name',
            'key.segment_context_name'
        ],
    ).compute()

    return metadata


def _read_lidar_metadata(scene_id: str, folder: str):
    path = Path(f"{folder}/lidar/{scene_id}.parquet")
    metadata = dd.read_parquet(
        path,
        columns=[
            'key.frame_timestamp_micros', 'key.laser_name',
            'key.segment_context_name'
        ],
    ).compute()

    return metadata


def _read_pose(scene_id: str, folder: str):
    path = Path(f"{folder}/vehicle_pose/{scene_id}.parquet")
    pose = dd.read_parquet(
        path,
        columns=[
            'key.frame_timestamp_micros',
            '[VehiclePoseComponent].world_from_vehicle.transform'
        ],
    ).compute()

    return pose


def _read_cam_calib(scene_id: str, folder: str):
    path = Path(f"{folder}/camera_calibration/{scene_id}.parquet")
    pose = dd.read_parquet(path, ).compute()
    return pose


def _read_lidar_calib(scene_id: str, folder: str):
    path = Path(f"{folder}/lidar_calibration/{scene_id}.parquet")
    pose = dd.read_parquet(path, ).compute()
    return pose

def _read_lidar_seg(scene_id: str, folder: str):
    path = Path(f"{folder}/lidar_segmentation/{scene_id}.parquet")
    pose = dd.read_parquet(path, ).compute()
    return pose

def _read_lidar(scene_id: str, folder: str):
    path = Path(f"{folder}/lidar/{scene_id}.parquet")
    pose = dd.read_parquet(path, ).compute()
    return pose

def load_scene(scene_id: str, folder: str, cache_whole_in_memory: bool = True):
    cam_img_path = Path(f"{folder}/camera_image/{scene_id}.parquet")
    cam_seg_path = Path(f"{folder}/camera_segmentation/{scene_id}.parquet")
    lidar_pc_path = Path(f"{folder}/lidar/{scene_id}.parquet")
    lidar_pc_pose_path = Path(f"{folder}/lidar_pose/{scene_id}.parquet")
    lidar_seg_path = Path(f"{folder}/lidar_segmentation/{scene_id}.parquet")

    cam_metadata = _read_camera_metadata(scene_id, folder)
    cam_pose = _read_cam_calib(scene_id, folder)
    lidar_metadata = _read_lidar_metadata(scene_id, folder)
    ego_poses = _read_pose(scene_id, folder)
    lidar_boxes = _read_all(scene_id, folder, 'lidar_box').compute()
    camera_boxes = _read_all(scene_id, folder, 'camera_box').compute()

    frame_id_to_frame: dict[str, BasicFrame] = {}
    for idx, row in ego_poses.iterrows():
        frame_id = row['key.frame_timestamp_micros']
        world_to_veh = row[
            '[VehiclePoseComponent].world_from_vehicle.transform'].reshape(4, 4)
        frame = BasicFrame(id=str(frame_id),
                      timestamp=frame_id,
                      pose=Pose(to_world=world_to_veh))

        frame_id_to_frame[frame_id] = frame
    cam_name_to_calib_row = {}
    for idx, row in cam_pose.iterrows():
        camera_name = row['key.camera_name']
        cam_name_to_calib_row[camera_name] = row
    lidar_name_to_calib_row = {}
    for idx, row in _read_lidar_calib(scene_id, folder).iterrows():
        laser_name = row['key.laser_name']
        lidar_name_to_calib_row[laser_name] = row
    for idx, row in cam_metadata.iterrows():
        frame_id = row['key.frame_timestamp_micros']
        frame = frame_id_to_frame[frame_id]
        camera_name = row['key.camera_name']
        calib_row = cam_name_to_calib_row[camera_name]

        sensor_to_vehicle = calib_row[
            '[CameraCalibrationComponent].extrinsic.transform'].reshape(4, 4)
        # sensor_to_vehicle = np.linalg.inv(vehicle_to_sensor)
        vehicle_to_world = frame.pose.to_world
        cam_to_world = vehicle_to_world @ sensor_to_vehicle
        fu = calib_row['[CameraCalibrationComponent].intrinsic.f_u']
        fv = calib_row['[CameraCalibrationComponent].intrinsic.f_v']
        cu = calib_row['[CameraCalibrationComponent].intrinsic.c_u']
        cv = calib_row['[CameraCalibrationComponent].intrinsic.c_v']
        k1 = calib_row['[CameraCalibrationComponent].intrinsic.k1']
        k2 = calib_row['[CameraCalibrationComponent].intrinsic.k2']
        p1 = calib_row['[CameraCalibrationComponent].intrinsic.p1']
        p2 = calib_row['[CameraCalibrationComponent].intrinsic.p2']
        k3 = calib_row['[CameraCalibrationComponent].intrinsic.k3']

        width = calib_row['[CameraCalibrationComponent].width']
        height = calib_row['[CameraCalibrationComponent].height']

        distortion = np.array([k1, k2, p1, p2, k3], np.float64)
        intrinsic = np.eye(4)
        intrinsic[0, 0] = fu
        intrinsic[1, 1] = fv
        intrinsic[0, 2] = cu
        intrinsic[1, 2] = cv
        seg_rc = WaymoImageResource(loader_type='WaymoImageSegLoader',
                                    image_name=CameraName(
                                        row['key.camera_name']),
                                    frame_timestamp_micros=frame_id,
                                    base_uri=str(cam_seg_path),
                                    uid=frame_id)
        image_rc = WaymoImageResource(loader_type='WaymoImageLoader',
                                         image_name=CameraName(
                                             row['key.camera_name']),
                                         frame_timestamp_micros=frame_id,
                                         base_uri=str(cam_img_path),
                                         uid=frame_id)
        frame.sensors.append(
            WaymoCamera(
                id=f"camera_{camera_name}",
                timestamp=frame_id,
                pose=Pose(to_world=cam_to_world, to_vehicle=sensor_to_vehicle),
                image_shape_wh=(width, height),
                intrinsic=intrinsic,
                distortion=distortion,
                image_rc=image_rc,
                seg_label_rc=seg_rc,
                # waymo camera x is front, z is up, use opencv cam (2, 0, 1) as reference.
                axes_front_u_v=(0, -1, -2),
            ))
    for idx, row in lidar_metadata.iterrows():
        frame_id = row['key.frame_timestamp_micros']
        frame = frame_id_to_frame[frame_id]
        laser_name = row['key.laser_name']
        calib_row = lidar_name_to_calib_row[laser_name]
        sensor_to_vehicle = calib_row[
            '[LiDARCalibrationComponent].extrinsic.transform'].reshape(4, 4)
        vehicle_to_world = frame.pose.to_world
        # sensor_to_vehicle = np.linalg.inv(vehicle_to_sensor)
        sensor_to_world = vehicle_to_world @ sensor_to_vehicle
        if cache_whole_in_memory:
            pc_loader = 'WaymoLidarWholeStorageLoader'
            pose_loader = 'WaymoLidarWholeStoragePoseLoader'
            seg_loader = 'WaymoLidarWholeStorageSegLoader'
            uid = ""
        else:
            pc_loader = 'WaymoLidarLoader'
            pose_loader = 'WaymoLidarPoseLoader'
            seg_loader = 'WaymoLidarSegLoader'
            uid = frame_id
        resource = WaymoLidarResource(loader_type=pc_loader,
                                      laser_name=LaserName(
                                          row['key.laser_name']),
                                      frame_timestamp_micros=frame_id,
                                      base_uri=str(lidar_pc_path),
                                      uid=uid)
        pose_resource = WaymoLidarResource(loader_type=pose_loader,
                                      laser_name=LaserName(
                                          row['key.laser_name']),
                                      frame_timestamp_micros=frame_id,
                                      base_uri=str(lidar_pc_pose_path),
                                      uid=uid)
        seg_resource = WaymoLidarResource(loader_type=seg_loader,
                                        laser_name=LaserName(
                                            row['key.laser_name']),
                                        frame_timestamp_micros=frame_id,
                                        base_uri=str(lidar_seg_path),
                                        uid=frame_id)
        beam_inclination = calib_row[
            '[LiDARCalibrationComponent].beam_inclination.values']
        beam_inclination_max = calib_row[
            '[LiDARCalibrationComponent].beam_inclination.max']
        beam_inclination_min = calib_row[
            '[LiDARCalibrationComponent].beam_inclination.min']
        if beam_inclination is None:
            beam_inclination = np.array([beam_inclination_min,
                                         beam_inclination_max], dtype=np.float32)
        else:
            beam_inclination = beam_inclination.astype(np.float32)
        beam_inclination = np.ascontiguousarray(beam_inclination[..., ::-1])
        frame.sensors.append(
            WaymoLidar(
                id=f"lidar_{laser_name}",
                timestamp=frame_id,
                pose=Pose(to_world=sensor_to_world, to_vehicle=sensor_to_vehicle),
                range_images_rc=resource,
                point_pose_rc=pose_resource,
                beam_inclination=beam_inclination,
                seg_label_rc=seg_resource,
                data_coord = CoordSystem.VEHICLE, # xyz loader load pc to vehicle

            ))
    for idx, row in lidar_boxes.iterrows():
        frame_id = row['key.frame_timestamp_micros']
        frame = frame_id_to_frame[frame_id]
        root_key = "[LiDARBoxComponent]"
        center_x = row[f"{root_key}.box.center.x"]
        center_y = row[f"{root_key}.box.center.y"]
        center_z = row[f"{root_key}.box.center.z"]
        size_x = row[f"{root_key}.box.size.x"]
        size_y = row[f"{root_key}.box.size.y"]
        size_z = row[f"{root_key}.box.size.z"]
        speed_x = row[f"{root_key}.speed.x"]
        speed_y = row[f"{root_key}.speed.y"]
        speed_z = row[f"{root_key}.speed.z"]
        acceleration_x = row[f"{root_key}.acceleration.x"]
        acceleration_y = row[f"{root_key}.acceleration.y"]
        acceleration_z = row[f"{root_key}.acceleration.z"]
        heading = row[f"{root_key}.box.heading"]
        type = row[f"{root_key}.type"]
        obj_id = row[f"key.laser_object_id"]
        box_to_veh = np.eye(4)
        rot = get_rotation_matrix_np(0, 0, heading)
        box_to_veh[:3, :3] = rot
        box_to_veh[:3, 3] = [center_x, center_y, center_z]
        veh_to_world = frame.pose.to_world
        box_to_world = veh_to_world @ box_to_veh
        pose = Pose(to_world=box_to_world, to_vehicle=box_to_veh,
            velocity=np.array([speed_x, speed_y, speed_z], dtype=np.float32),
            acceleration=np.array([acceleration_x, acceleration_y, acceleration_z], dtype=np.float32))
        o3d = Object3d(track_id=obj_id, type=type, pose=pose, size=np.array([size_x, size_y, size_z], dtype=np.float32))
        frame.objects.append(o3d)
    for idx, row in camera_boxes.iterrows():
        frame_id = row['key.frame_timestamp_micros']
        frame = frame_id_to_frame[frame_id]
        root_key = "[CameraBoxComponent]"
        center_x = row[f"{root_key}.box.center.x"]
        center_y = row[f"{root_key}.box.center.y"]
        size_x = row[f"{root_key}.box.size.x"]
        size_y = row[f"{root_key}.box.size.y"]
        type = row[f"{root_key}.type"]
        obj_id = row[f"key.camera_object_id"]
        cam_name = row[f"key.camera_name"]
        bbox_xywh = np.array([center_x - size_x / 2, center_y - size_y / 2, size_x, size_y], dtype=np.float32)
        obj_2d = Object2d(track_id=obj_id, type=type, bbox_xywh=bbox_xywh)
        cam = frame.get_camera_by_id(f"camera_{cam_name}")
        cam.objects.append(obj_2d)

    frames = list(frame_id_to_frame.values())
    scene: Scene[BasicFrame] = Scene(id=scene_id, frames=frames)
    return scene


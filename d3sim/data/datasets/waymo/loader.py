from ast import List
import enum
from pathlib import Path
import time
from typing import Any
from typing_extensions import override

import torch
import tqdm
from d3sim.constants import D3SIM_DEFAULT_DEVICE
import d3sim.core.dataclass_dispatch as dataclasses

from d3sim.data.scene_def.base import CoordSystem, Pose, Scene, Sensor, Resource, BaseFrame, ALL_RESOURCE_LOADERS, ResourceLoader
from d3sim.data.scene_def.camera import BasicCamera, BasicPinholeCamera
import dask.dataframe as dd
import numpy as np
from d3sim.data.datasets.waymo.ops import range_image_to_point_cloud
from d3sim.data.scene_def.frame import BasicFrame
from d3sim.data.scene_def.lidar import BasicLidar

import torchvision
import cv2

from d3sim.ops.points.downsample import downsample_indices  



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
        if self._point_load_cache is not None:
            return self._point_load_cache
        veh_to_world = torch.from_numpy(self.pose.vehicle_to_world.astype(np.float32)).to(D3SIM_DEFAULT_DEVICE)
        extrinsic = torch.from_numpy(self.pose.to_vehicle.astype(np.float32)).to(D3SIM_DEFAULT_DEVICE)
        ris = self.range_images_rc.data 
        inclination = torch.from_numpy(self.beam_inclination).to(D3SIM_DEFAULT_DEVICE)
        ri_0 = torch.from_numpy(ris[0]).to(D3SIM_DEFAULT_DEVICE)
        ri_1 = torch.from_numpy(ris[1]).to(D3SIM_DEFAULT_DEVICE)
        pixel_pose: torch.Tensor | None = None
        if self.id == f"lidar_{LaserName.TOP}": # TOP
            pixel_pose = torch.from_numpy(self.point_pose_rc.data).to(D3SIM_DEFAULT_DEVICE)
            # breakpoint()
        res_0, res_0_mask = range_image_to_point_cloud(ri_0, extrinsic, inclination, pixel_pose, veh_to_world, self._tmp_pixel_pose_transform)
        res_1, res_1_mask = range_image_to_point_cloud(ri_1, extrinsic, inclination, pixel_pose, veh_to_world, self._tmp_pixel_pose_transform)
        res = torch.cat([res_0, res_1], dim=0).cpu().numpy()
        inten_0 = ri_0[..., 1].reshape(-1)
        inten_1 = ri_1[..., 1].reshape(-1)
        inten = torch.cat([inten_0, inten_1], dim=0).cpu().numpy()
        mask = torch.cat([res_0_mask, res_1_mask], dim=0).cpu().numpy()
        res_all = res, inten, mask
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
        return self._load_xyz_inten_mask()[0]

    @property
    def intensity(self):
        return self._load_xyz_inten_mask()[1]

    @property
    def mask(self):
        return self._load_xyz_inten_mask()[2]

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
        img_decoded = cv2.imdecode(np.frombuffer(img_png_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        dividor = data.compute()['[CameraSegmentationLabelComponent].panoptic_label_divisor'].iloc[0]
        return img_decoded % dividor

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
    frames = list(frame_id_to_frame.values())
    scene: Scene[BasicFrame] = Scene(id=scene_id, frames=frames)
    return scene


def __read(scene_id: str, folder: str, tag: str) -> dd.DataFrame:
    path = Path(f"{folder}/{tag}/{scene_id}.parquet")

    return dd.read_parquet(path)

def _dev_lidar():
    scene_id = "13207915841618107559_2980_000_3000_000"
    folder = "/Users/yanyan/Downloads/waymo/training"
    "/Users/yanyan/Downloads/waymo/training/camera_calibration/13196796799137805454_3036_940_3056_940.parquet"
    # breakpoint()
    # print(cam_seg.columns, cam_seg.shape)
    # breakpoint()
    scene = load_scene(scene_id, folder)
    scene = scene.apply_world_transform_inplace(np.linalg.inv(scene.frames[0].pose.vehicle_to_world))

    frames = scene.frames
    # print(scene.frames[0].get_lidar_xyz("lidar_1", CoordSystem.VEHICLE))
    # scene = scene.apply_world_transform_inplace(np.linalg.inv(frames[0].pose.vehicle_to_world))
    all_pc = []
    for frame in tqdm.tqdm(scene.frames):
        all_pc.append(frame.get_lidar_xyz("lidar_1", CoordSystem.WORLD))
    all_pc = np.concatenate(all_pc)
    all_pc_th = torch.from_numpy(all_pc).to(D3SIM_DEFAULT_DEVICE)
    t = time.time()
    all_pc_down = downsample_indices(all_pc_th, 0.2)
    print(time.time() - t)
    print(all_pc.shape, all_pc_down.shape)
    return 
    lidar_obj = frames[0].get_sensors_by_type(WaymoLidar)[0]
    print(lidar_obj.id)
    print(lidar_obj.xyz.shape, lidar_obj.xyz.shape)
    for frame in frames:
        lidar = frame.get_sensors_by_type(WaymoLidar)[0] 
        pc = lidar.xyz 
        print("???", pc.shape)
        down_pc_inds = downsample_indices(torch.from_numpy(pc).to(D3SIM_DEFAULT_DEVICE), 0.2)
        print(pc.shape, down_pc_inds.shape)
        break
        # seg = frame.get_sensors_by_type(WaymoLidar)[0].segmentation
        # if seg is not None:
        #     print(seg.shape, seg.dtype)
        #     print(frame.get_sensors_by_type(WaymoLidar)[0].xyz.shape)
        #     break
        for cam in frame.get_sensors_by_type(WaymoCamera):
            breakpoint()
            if cam.segmentation is not None:
                print(frame.id, cam.id, cam.segmentation.shape, cam.segmentation.dtype)

def test_lidar():
    from d3sim.data.datasets.waymo import v2

    scene_id = "13207915841618107559_2980_000_3000_000"
    folder = "/Users/yanyan/Downloads/waymo/training"
    scene = load_scene(scene_id, folder)
    frames = scene.frames
    lidar_obj = frames[0].get_sensors_by_type(WaymoLidar)[0]
    pc_my = lidar_obj.xyz
    num_pc = pc_my.shape[0] // 2
    pc_my = pc_my[:num_pc][lidar_obj.mask[:num_pc]]
    lidar_calib_df = __read(scene_id, folder, 'lidar_calibration').compute()
    xyzm = lidar_obj.xyz_masked()
    print(pc_my[:5], xyzm)

    lidar_pose_df = __read(scene_id, folder, 'lidar_pose')
    lidar_df = __read(scene_id, folder, 'lidar')

    lidar_df = v2.merge(lidar_df, lidar_pose_df)
    vehicle_pose_df = __read(scene_id, folder, 'vehicle_pose')

    df = v2.merge(lidar_df, vehicle_pose_df)
    _, row = next(iter(df.iterrows()))
    print(row.keys())
    lidar = v2.LiDARComponent.from_dict(row)
    lidar_pose = v2.LiDARPoseComponent.from_dict(row)
    lidar_calib = v2.LiDARCalibrationComponent.from_dict(lidar_calib_df.iloc[4])
    frame_pose = v2.VehiclePoseComponent.from_dict(row)
    print(lidar.range_image_return1.shape)
    pc_ref_tf = v2.convert_range_image_to_point_cloud(lidar.range_image_return1, lidar_calib, lidar_pose.range_image_return1, frame_pose)
    pc_ref = pc_ref_tf.numpy()
    print(lidar_calib.key, lidar_pose.key)
    # print(lidar_obj.pose.vehicle_to_world, frame_pose.world_from_vehicle.transform)
    print(pc_ref[:5])
    print(np.linalg.norm(lidar_obj.point_pose_rc.data.reshape(-1) - lidar_pose.range_image_return1.values.reshape(-1)))
    print(np.linalg.norm(pc_my - pc_ref))

    breakpoint()
    print("?")

    # print(lidar.key, pc_ref.shape, pc_my.shape)
    # print(pc_my[:5])

if __name__ == "__main__":
    _dev_lidar()
import enum
from pathlib import Path
import time

import torch
from d3sim.constants import D3SIM_DEFAULT_DEVICE

from d3sim.data.scene_def.base import CoordSystem, Object2d, Object3d, Pose, Scene, Sensor, Resource, BaseFrame, ALL_RESOURCE_LOADERS, ResourceLoader
from d3sim.data.scene_def.camera import BasicCamera, BasicPinholeCamera
import dask.dataframe as dd
import numpy as np
from d3sim.data.scene_def.frame import BasicFrame
from d3sim.data.scene_def.lidar import BasicLidar
from cumm import tensorview as tv 
import tqdm 
import cv2

from d3sim.data.datasets.waymo.loader import load_scene, CameraSegType, WaymoLidar
from d3sim.ops.points.downsample import downsample_indices
from d3sim.ops.points.projection import depth_map_to_jet_rgb, get_depth_map_from_uvd

def __read(scene_id: str, folder: str, tag: str) -> dd.DataFrame:
    path = Path(f"{folder}/{tag}/{scene_id}.parquet")

    return dd.read_parquet(path)

def _dev_lidar():
    scene_id = "13207915841618107559_2980_000_3000_000"
    folder = "/Users/yanyan/Downloads/waymo/training"
    "/Users/yanyan/Downloads/waymo/training/camera_calibration/13196796799137805454_3036_940_3056_940.parquet"
    # breakpoint()
    data = __read(scene_id, folder, 'camera_box').compute()
    # print(data.columns, data.shape)
    # breakpoint()
    scene = load_scene(scene_id, folder)
    scene = scene.apply_world_transform_inplace(np.linalg.inv(scene.frames[0].pose.vehicle_to_world))

    frames = scene.frames
    # print(scene.frames[0].get_lidar_xyz("lidar_1", CoordSystem.VEHICLE))
    # scene = scene.apply_world_transform_inplace(np.linalg.inv(frames[0].pose.vehicle_to_world))
    all_pc = []
    for frame in tqdm.tqdm(scene.frames):
        # uvd = frame.get_projected_pc_in_cam("lidar_1", "camera_1")
        cam = frame.get_camera_by_id("camera_1")
        if cam.segmentation is not None:
            point_uvd = frame.get_projected_pc_in_cam("lidar_1", "camera_1")
            depth_map = get_depth_map_from_uvd(point_uvd, cam.image_shape_wh)
            depth_map_rgb = depth_map_to_jet_rgb(depth_map).cpu().numpy()
            res = frame.get_projected_image_in_cam_np("lidar_1", "camera_1")
            cam = frame.get_camera_by_id("camera_1")
            cam_seg_label = cam.segmentation
            bbox_mask_np = cam.get_bbox_mask_np()
            res.reshape(-1, 3)[cam_seg_label.reshape(-1) == CameraSegType.TYPE_SKY] = (70, 130, 25)
            res.reshape(-1, 3)[bbox_mask_np.reshape(-1)] = 0

            for obj in cam.objects:
                cv2.rectangle(depth_map_rgb, (int(obj.bbox_xywh[0]), int(obj.bbox_xywh[1])),
                    (int(obj.bbox_xywh[0] + obj.bbox_xywh[2]), int(obj.bbox_xywh[1] + obj.bbox_xywh[3])), (0, 255, 0), 2)
            cv2.imwrite("test.jpg", depth_map_rgb)
            breakpoint()
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
    num_pc = pc_my.shape[0] # 2
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
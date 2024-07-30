from typing import List, Literal
from pathlib import Path

from d3sim.constants import D3SIM_FOLDER_CREATE_MODE
import subprocess

WAYMO_V2_FOLDERS = [
    "camera_box",
    "camera_calibration",
    "camera_hkp",
    "camera_image",
    "camera_segmentation",
    "camera_to_lidar_box_association",
    "lidar_box",
    "lidar_calibration",
    "lidar_camera_projection",
    "lidar_camera_synced_box",
    "lidar_hkp",
    "lidar_pose",
    "lidar_segmentation",
    "lidar",
    "projected_lidar_box",
    "stats",
    "vehicle_pose",
]


def download_by_scene_id(scene_id: str,
                         target_folder: str,
                         folder: Literal["training", "testing", "validation"],
                         data_ver: str = "waymo_open_dataset_v_2_0_1"):
    cmds: List[List[str]] = []
    for sub in WAYMO_V2_FOLDERS:
        save_folder = f"{target_folder}/{folder}/{sub}/{scene_id}.parquet"
        Path(save_folder).parent.mkdir(exist_ok=True,
                                       parents=True,
                                       mode=D3SIM_FOLDER_CREATE_MODE)
        path = f"gs://{data_ver}/{folder}/{sub}/{scene_id}.parquet"
        cmd = ["gcloud", "storage", "cp", path, save_folder]
        cmds.append(cmd)
        subprocess.check_call(cmd)

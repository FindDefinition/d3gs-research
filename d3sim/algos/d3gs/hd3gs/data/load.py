from pathlib import Path
from d3sim.algos.d3gs.data_base import MultipleSceneDatasetBase
from d3sim.constants import IsAppleSiliconMacOs
from d3sim.data.scene_def.base import CameraFieldTypes, Pose, Resource, Scene
from d3sim.data.scene_def.camera import BasicPinholeCamera
import numpy as np 
import imagesize

from d3sim.data.scene_def.frame import BasicFrame
from d3sim.dev import load_secret_constants

def load_example_scene(root: str):
    root_p = Path(root)
    images_folder = root_p / "images"
    masks_folder = root_p / "masks"

    cam_names = [f"cam{i}" for i in range(1, 7)]
    frame_id_to_cam_dict: dict[str, list[BasicPinholeCamera]] = {}
    for cam_name in cam_names:
        cam_folder = images_folder / cam_name
        cam_mask_folder = masks_folder / cam_name
        cam_folder_files = list(cam_folder.glob("*.jpg"))
        cam_folder_files.sort()
        # mask_folder_files = cam_mask_folder.glob("*.png")
        for img_path in cam_folder_files:
            mask_img_path = cam_mask_folder / f"{img_path.stem}.png"
            frame_id = img_path.stem
            img_rc = BasicPinholeCamera.get_opencv_img_resource(str(img_path))
            width, height = imagesize.get(str(img_path))
            cam = BasicPinholeCamera(id=cam_name, timestamp=0, pose=Pose(), fields={
                CameraFieldTypes.IMAGE: img_rc
            }, intrinsic=np.eye(4), distortion=np.zeros([4]), image_shape_wh=(int(width), int(height)), objects=[])
            cam.fields[CameraFieldTypes.VALID_MASK] = BasicPinholeCamera.get_opencv_img_resource(mask_img_path)
            if frame_id not in frame_id_to_cam_dict:
                frame_id_to_cam_dict[frame_id] = []
            frame_id_to_cam_dict[frame_id].append(cam)
    all_frames: list[BasicFrame] = []
    for frame_id, cam_list in frame_id_to_cam_dict.items():
        frame = BasicFrame(id=frame_id, timestamp=0, pose=Pose(), sensors=[*cam_list], objects=[])
        all_frames.append(frame)
    # all_frames.sort(key=lambda x: x.id)
    # for frame in all_frames:
    #     print(frame.id)
    print(root_p)
    return Scene(id="example_dataset", frames=all_frames, uri=str(root_p))


def _load_debug_scene():
    path = load_secret_constants().h3dgs_example_dataset_path
    return load_example_scene(path)

if __name__ == "__main__":
    _load_debug_scene()
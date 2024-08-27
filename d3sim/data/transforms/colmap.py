from ast import Is
from pathlib import Path
from typing import Literal
from d3sim.constants import IsAppleSiliconMacOs
from d3sim.data.scene_def import Scene
from d3sim.data.scene_def.base import CameraFieldTypes, Resource, SingleFileLoader
from d3sim.data.scene_def.camera import BasicPinholeCamera
from d3sim.data.scene_def.frame import BasicFrame
from d3sim.data.scene_def.transform import SceneTransform
from d3sim.core import dataclass_dispatch as dataclasses
import shutil
import subprocess
import cv2 
import torch 

HAS_CUDA = torch.cuda.is_available()

@dataclasses.dataclass
class ColmapWorkDirs:
    colmap_dir: Path
    cameras_root: Path = Path()
    masks_root: Path = Path()
    sfm_out_root: Path = Path()
    sfm_prior_root: Path = Path()
    sfm_tri_root: Path = Path()
    sfm_ba_out_root: Path = Path()

    undistorted_img_root: Path = Path() 

    database_path: Path = Path()
    match_list_path: Path = Path()
    create: bool = False
    def __post_init__(self):
        if self.create:
            self.colmap_dir.mkdir(exist_ok=True, parents=True, mode=0o755)
        else:
            assert self.colmap_dir.exists(), "colmap dir must exist"
        self.cameras_root = self.colmap_dir / "cameras"
        self.masks_root = self.colmap_dir / "masks"
        self.sfm_out_root = self.colmap_dir / "camera_calibration/unrectified/sparse"
        self.sfm_prior_root = self.colmap_dir / "camera_calibration/manually/sparse"
        self.sfm_tri_root = self.colmap_dir / "camera_calibration/triangulated/sparse"
        self.sfm_ba_out_root = self.colmap_dir / "camera_calibration/ba/sparse"
        self.undistorted_img_root = self.colmap_dir / "undistorted_models"
        if self.create:
            self.cameras_root.mkdir(exist_ok=True, parents=True, mode=0o755)
            self.masks_root.mkdir(exist_ok=True, parents=True, mode=0o755)
            self.sfm_out_root.mkdir(exist_ok=True, parents=True, mode=0o755)
            self.sfm_prior_root.mkdir(exist_ok=True, parents=True, mode=0o755)
            self.sfm_tri_root.mkdir(exist_ok=True, parents=True, mode=0o755)
            self.sfm_ba_out_root.mkdir(exist_ok=True, parents=True, mode=0o755)
            self.undistorted_img_root.mkdir(exist_ok=True, parents=True, mode=0o755)

        else:
            assert self.cameras_root.exists(), "cameras root must exist"
            assert self.masks_root.exists(), "masks root must exist"
            assert self.sfm_out_root.exists(), "sfm out root must exist"
            assert self.sfm_prior_root.exists(), "sfm prior root must exist"
            assert self.sfm_tri_root.exists(), "sfm tri root must exist"
            assert self.sfm_ba_out_root.exists(), "sfm ba root must exist"
            assert self.undistorted_img_root.exists(), "undistorted img root must exist"
        self.database_path = self.colmap_dir / "database.db"
        self.match_list_path = self.colmap_dir / "matches.txt"

@dataclasses.dataclass
class RawColmapFeatureExtract(SceneTransform):
    """Do colmap calibration without any priors.
    """
    colmap_path: str = "colmap"
    default_focal_length_factor: float | None = 0.5 # TODO set to None later
    use_gpu: bool | None = None
    def __call__(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = Path(scene.uri) / "__colmap"
        if colmap_dir.exists():
            shutil.rmtree(colmap_dir)
        workdirs = ColmapWorkDirs(colmap_dir)
        cam_id_to_cams = scene.get_sensor_id_to_sensors(BasicPinholeCamera)

        for cam_id, cams in cam_id_to_cams.items():
            cam_root = workdirs.cameras_root / cam_id
            subfolder = cam_root
            subfolder.mkdir(exist_ok=True, parents=True, mode=0o755)
            mask_sub_folder = workdirs.masks_root / cam_id
            mask_sub_folder.mkdir(exist_ok=True, parents=True, mode=0o755)
            file_copy_pair = []
            for cam in cams:
                if isinstance(cam.image_rc.loader, SingleFileLoader) and cam.is_transform_empty:
                    # just copy the file
                    uri = cam.image_rc.base_uri
                    uri_suffix = Path(uri).suffix
                    img_path = subfolder / f"{cam.global_id}{uri_suffix}"
                    file_copy_pair.append((uri, img_path))
                else:
                    # read img, lazy transforms may applied.
                    # TODO parallel here
                    img = cam.image 
                    # save as jpg
                    img_path = subfolder / f"{cam.global_id}.jpg"
                    cv2.imwrite(str(img_path), img)
                if cam.has_field(CameraFieldTypes.VALID_MASK):
                    field_val = cam.fields[CameraFieldTypes.VALID_MASK]
                    if isinstance(field_val, Resource) and isinstance(field_val.loader, SingleFileLoader) and cam.is_transform_empty:
                        file_copy_pair.append((field_val.base_uri, mask_sub_folder / f"{img_path.name}.png"))
                    else:
                        mask = cam.get_field_np(CameraFieldTypes.VALID_MASK)
                        assert mask is not None 
                        mask_path = mask_sub_folder / f"{img_path.name}.png"
                        cv2.imwrite(str(mask_path), mask) 
            # run copy
            for src, dst in file_copy_pair:
                shutil.copy(src, dst)
        colmap_feature_extractor_args = [
            self.colmap_path, "feature_extractor",
                "--database_path", f"{workdirs.database_path}",
                "--image_path", f"{workdirs.cameras_root}",
                "--ImageReader.single_camera_per_folder", "1",
                "--ImageReader.camera_model", "OPENCV",
            ]
        if self.default_focal_length_factor is not None:
            colmap_feature_extractor_args.append("--ImageReader.default_focal_length_factor")
            colmap_feature_extractor_args.append(str(self.default_focal_length_factor))
        use_gpu = self.use_gpu
        if use_gpu is None:
            use_gpu = not IsAppleSiliconMacOs and HAS_CUDA
        if use_gpu:
            assert not IsAppleSiliconMacOs, "gpu is not supported on apple silicon"
            colmap_feature_extractor_args.append("--SiftExtraction.use_gpu")
            colmap_feature_extractor_args.append("1")
        subprocess.run(colmap_feature_extractor_args, check=True)

        return scene 


@dataclasses.dataclass
class ColmapCustomMatchGen(SceneTransform):
    num_sequence_match: int = 0
    num_quad_sequence_match: int = 10
    def __call__(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = Path(scene.uri) / "__colmap"
        workdirs = ColmapWorkDirs(colmap_dir, create=False)
        cam_id_to_cams = scene.get_sensor_id_to_sensors(BasicPinholeCamera)
        cam_ids = list(cam_id_to_cams.keys())
        # cam_ids = [f"cam{i}" for i in [2, 5, 4, 3, 6, 1]]
        # generate cam id match pairs
        cam_id_match_pairs = []
        cam_idx_match_pairs = []
        for i in range(len(cam_ids)):
            for j in range(i, len(cam_ids)):
                cam_id_match_pairs.append((cam_ids[i], cam_ids[j]))
                cam_idx_match_pairs.append((i, j))
        # print(cam_id_match_pairs)

        all_matchs: list[tuple[str, str]] = []
        for cam_idx1, cam_idx2 in cam_idx_match_pairs:
            cam_id1 = cam_ids[cam_idx1]
            cam_id2 = cam_ids[cam_idx2]
            # use relative
            cam_colmap_dir1 = Path(cam_id1)
            cam_colmap_dir2 = Path(cam_id2)

            for current_image_id, cam in enumerate(cam_id_to_cams[cam_id1]):
                pass 
                for frame_step in range(self.num_sequence_match):
                    matched_frame_id = current_image_id + frame_step
                    if matched_frame_id < len(cam_id_to_cams[cam_id2]):
                        cam2 = cam_id_to_cams[cam_id2][matched_frame_id]
                        all_matchs.append((str(cam_colmap_dir1 / f"{cam.global_id}.jpg"), str(cam_colmap_dir2 / f"{cam2.global_id}.jpg")))

                for match_id in range(self.num_quad_sequence_match):
                    matched_frame_id = current_image_id + int(2**match_id) - 1
                    if matched_frame_id < len(cam_id_to_cams[cam_id2]):
                        cam2 = cam_id_to_cams[cam_id2][matched_frame_id]
                        all_matchs.append((str(cam_colmap_dir1 / f"{cam.global_id}.jpg"), str(cam_colmap_dir2 / f"{cam2.global_id}.jpg")))

        all_matchs = list(dict.fromkeys(all_matchs))
        # remove reversed pair
        all_matchs_reversed = [(b, a) for a, b in all_matchs]
        all_matchs_reversed_set = set(all_matchs_reversed)
        all_matchs = list(filter(lambda x: x not in all_matchs_reversed_set, all_matchs))
        assert colmap_dir.exists(), "feature extract transform must be run first"
        with open(workdirs.match_list_path, "w") as f:
            for pair in all_matchs:
                f.write(f"{pair[0]} {pair[1]}\n")
        return scene 

@dataclasses.dataclass
class ColmapCustomMatch(SceneTransform):
    """Do colmap calibration without any priors.
    """
    colmap_path: str = "colmap"
    use_gpu: bool | None = None

    def __call__(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = Path(scene.uri) / "__colmap"
        workdirs = ColmapWorkDirs(colmap_dir, create=False)

        colmap_matches_importer_args = [
            self.colmap_path, "matches_importer",
            "--database_path", str(workdirs.database_path),
            "--match_list_path", str(workdirs.match_list_path),
        ]
        use_gpu = self.use_gpu
        if use_gpu is None:
            use_gpu = not IsAppleSiliconMacOs and HAS_CUDA
        if use_gpu:
            assert not IsAppleSiliconMacOs, "gpu is not supported on apple silicon"
            colmap_matches_importer_args.append("--SiftMatching.use_gpu")
            colmap_matches_importer_args.append("1")
        subprocess.run(colmap_matches_importer_args, check=True)
        return scene 

@dataclasses.dataclass
class ColmapBuiltinMatch(SceneTransform):
    """Do colmap calibration without any priors.
    """
    colmap_path: str = "colmap"
    use_gpu: bool | None = None
    matcher: Literal["exhaustive", "sequential", "transitive", "vocab_tree", "spatial"] = "exhaustive"

    def __call__(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = Path(scene.uri) / "__colmap"
        workdirs = ColmapWorkDirs(colmap_dir, create=False)
        
        colmap_matches_importer_args = [
            self.colmap_path, f"{self.matcher}_matcher",
            "--database_path", str(workdirs.database_path),
        ]
        use_gpu = self.use_gpu
        if use_gpu is None:
            use_gpu = not IsAppleSiliconMacOs and HAS_CUDA
        if use_gpu:
            assert not IsAppleSiliconMacOs, "gpu is not supported on apple silicon"
            colmap_matches_importer_args.append("--SiftMatching.use_gpu")
            colmap_matches_importer_args.append("1")
        subprocess.run(colmap_matches_importer_args, check=True)
        return scene 


@dataclasses.dataclass
class ColmapMapper(SceneTransform):
    """Do colmap calibration without any priors.
    """
    colmap_path: str = "colmap"
    is_hierarchical: bool = False
    ba_global_function_tolerance: float = 0.000001

    def __call__(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = Path(scene.uri) / "__colmap"
        workdirs = ColmapWorkDirs(colmap_dir, create=False)
        colmap_hierarchical_mapper_args = [
            self.colmap_path, "hierarchical_mapper" if self.is_hierarchical else "mapper",
            "--database_path", f"{workdirs.database_path}",
            "--image_path", f"{workdirs.cameras_root}",
            "--output_path", str(workdirs.sfm_out_root),
            "--Mapper.ba_global_function_tolerance", str(self.ba_global_function_tolerance)
            ]
        subprocess.run(colmap_hierarchical_mapper_args, check=True)
        scene.set_user_data("model_output", str(workdirs.sfm_out_root))
        return scene 

@dataclasses.dataclass
class ColmapCreateModelFromPrior(SceneTransform):
    colmap_path: str = "colmap"

@dataclasses.dataclass
class ColmapPointTriangulator(SceneTransform):
    colmap_path: str = "colmap"
    def __call__(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = Path(scene.uri) / "__colmap"
        workdirs = ColmapWorkDirs(colmap_dir, create=False)
        colmap_hierarchical_mapper_args = [
            self.colmap_path, "point_triangulator",
            "--database_path", f"{workdirs.database_path}",
            "--image_path", f"{workdirs.cameras_root}",
            "--input_path", f"{workdirs.sfm_prior_root}",
            "--output_path", str(workdirs.sfm_tri_root),
            ]
        subprocess.run(colmap_hierarchical_mapper_args, check=True)
        scene.set_user_data("model_output", str(workdirs.sfm_tri_root))
        return scene 

@dataclasses.dataclass
class ColmapBundleAdjustment(SceneTransform):
    """assume you run point_triangulator first. (get a complete sparse model)
    """
    colmap_path: str = "colmap"
    def __call__(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = Path(scene.uri) / "__colmap"
        workdirs = ColmapWorkDirs(colmap_dir, create=False)
        colmap_hierarchical_mapper_args = [
            self.colmap_path, "bundle_adjuster",
            "--database_path", f"{workdirs.database_path}",
            "--image_path", f"{workdirs.cameras_root}",
            "--input_path", f"{workdirs.sfm_prior_root}",
            "--output_path", str(workdirs.sfm_out_root),
            ]
        subprocess.run(colmap_hierarchical_mapper_args, check=True)
        scene.set_user_data("model_output", str(workdirs.sfm_ba_out_root))
        return scene 

@dataclasses.dataclass
class ColmapUndistort(SceneTransform):
    """assume you run all pipeline first. (get a complete sparse model)
    """
    colmap_path: str = "colmap"
    def __call__(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = Path(scene.uri) / "__colmap"
        workdirs = ColmapWorkDirs(colmap_dir, create=False)
        colmap_hierarchical_mapper_args = [
            self.colmap_path, "image_undistorter",
            "--image_path", f"{workdirs.cameras_root}",
            "--input_path", f"{scene.get_user_data_type_checked('model_output', str)}",
            "--output_path", str(workdirs.undistorted_img_root),
            "--output_type COLMAP"
            ]
        subprocess.run(colmap_hierarchical_mapper_args, check=True)
        scene.set_user_data("model_output", str(workdirs.sfm_ba_out_root))
        return scene 

# point_triangulator

def __main():
    from d3sim.algos.d3gs.hd3gs.data.load import _load_debug_scene
    scene = _load_debug_scene()
    # scene = RawColmapFeatureExtract()(scene)
    scene = ColmapCustomMatchGen()(scene)
    scene = ColmapCustomMatch()(scene)
    scene = ColmapMapper(is_hierarchical=True)(scene)

if __name__ == "__main__":
    __main()
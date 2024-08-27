from pathlib import Path
from typing import Literal
from d3sim.constants import IsAppleSiliconMacOs
from d3sim.data.scene_def import Scene
from d3sim.data.scene_def.base import CameraFieldTypes, DistortType, Resource, SingleFileLoader
from d3sim.data.scene_def.camera import BasicPinholeCamera
from d3sim.data.scene_def.frame import BasicFrame
from d3sim.data.scene_def.transform import SceneTransform, SceneTransformOfflineDisk
from d3sim.core import dataclass_dispatch as dataclasses
import shutil
import subprocess
import cv2 
import torch 
import d3sim.data.transforms.colmap_util as cmu
import os 
import numpy as np 
from sklearn.neighbors import NearestNeighbors
import scipy
import scipy.spatial
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
        self.sfm_out_root = self.colmap_dir / "cam_calib/unrectified/sparse"
        self.sfm_prior_root = self.colmap_dir / "cam_calib/manually/sparse"
        self.sfm_tri_root = self.colmap_dir / "cam_calib/triangulated/sparse"
        self.sfm_ba_out_root = self.colmap_dir / "cam_calib/ba/sparse"
        self.undistorted_img_root = self.colmap_dir / "cam_calib/undistorted"
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
class ColmapFeatureExtract(SceneTransformOfflineDisk):
    """Do colmap calibration without any priors.
    """
    colmap_path: str = "colmap"
    default_focal_length_factor: float | None = None
    use_gpu: bool | None = None
    def __call__(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = Path(scene.uri) / "__colmap"
        # if colmap_dir.exists():
        #     shutil.rmtree(colmap_dir)
        workdirs = ColmapWorkDirs(colmap_dir, create=True)
        if self.file_flag_exists(colmap_dir, f"{self.default_focal_length_factor}"):
            return scene
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
        self.write_file_flag(colmap_dir, f"{self.default_focal_length_factor}")

        return scene 

@dataclasses.dataclass
class ColmapCreatePoseAndIntrinsicPrior(SceneTransformOfflineDisk):
    """Add intrinsic priors and create extrinsic priors
    """
    add_intrinsic_prior: bool = True
    add_pose_prior: bool = True
    # let colmap use the prior focal length
    prior_focal_length: bool = True
    def __call__(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = Path(scene.uri) / "__colmap"
        if self.file_flag_exists(colmap_dir):
            return scene
        workdirs = ColmapWorkDirs(colmap_dir, create=False)
        db = cmu.COLMAPDatabase.connect(str(workdirs.database_path))
        cam_id_to_cams = scene.get_sensor_id_to_sensors(BasicPinholeCamera)
        # read db, get d3sim cam global id to colmap cam global id map
        cam_global_id_to_colmap_id = {}
        for row in  db.execute("SELECT * FROM images"):
            (image_id, cam_file_name, camera_id) = row
            cam_global_id = Path(cam_file_name).stem
            cam_global_id_to_colmap_id[cam_global_id] = (image_id, camera_id, cam_file_name)
        prior_dir = workdirs.sfm_prior_root
        # camera_id, model, width, height, params, prior = next(rows)
        # params = blob_to_array(params, np.float64)

        res_colmap_images = {}
        res_colmap_cams = {}
        for cam_id, cams in cam_id_to_cams.items():
            # use first cam as intrinsic prior
            # TODO convert intrinsic matrix and distort to flatten vector
            focal_length = cams[0].focal_length
            center = cams[0].principal_point
            distort = cams[0].distortion[:4]
            assert cams[0].distortion_type == DistortType.kOpencvPinhole or cams[0].distortion_type == DistortType.kOpencvPinholeWaymo
            params = [focal_length[0], focal_length[1], center[0], center[1], distort[0], distort[1], distort[2], distort[3]]
            colmap_img_id, colmap_cam_id, cam_file_name = cam_global_id_to_colmap_id[cams[0].global_id]
            if self.add_intrinsic_prior:
                db.add_camera("OPENCV", cams[0].image_shape_wh[0], cams[0].image_shape_wh[1], params, camera_id=colmap_cam_id, prior_focal_length=self.prior_focal_length)
            colmap_cam = cmu.Camera(id=colmap_cam_id, model="OPENCV", width=cams[0].image_shape_wh[0], height=cams[0].image_shape_wh[1], params=params)
            res_colmap_cams[colmap_cam_id] = colmap_cam
            if self.add_pose_prior:
                for cam in cams:
                    colmap_img_id, colmap_cam_id, cam_file_name = cam_global_id_to_colmap_id[cam.global_id]
                    assert not cam.pose.is_empty
                    world2cam = np.linalg.inv(cam.pose.to_world)
                    R = world2cam[:3, :3]
                    T = world2cam[:3, 3]
                    qvec = cmu.rotmat2qvec(R)
                    cmu_img = cmu.Image(id=colmap_img_id, qvec=qvec, tvec=T, camera_id=colmap_cam_id, name=cam_file_name, xys=[], point3D_ids=[])
                    res_colmap_images[colmap_img_id] = cmu_img
        if self.add_intrinsic_prior:
            db.commit()
        with open(prior_dir / "points3D.txt", "w") as f:
            f.write("")
        cmu.write_cameras_text(res_colmap_cams, str(prior_dir / "cameras.txt"))
        cmu.write_images_text(res_colmap_images, str(prior_dir / "images.txt"))
        self.write_file_flag(colmap_dir)
        return scene 

@dataclasses.dataclass
class ColmapCustomMatchGen(SceneTransformOfflineDisk):
    num_sequence_match: int = 0
    num_quad_sequence_match: int = 10
    def __call__(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        unique_key = f"{self.num_sequence_match}_{self.num_quad_sequence_match}"
        colmap_dir = Path(scene.uri) / "__colmap"
        if self.file_flag_exists(colmap_dir, unique_key):
            return scene
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
        self.write_file_flag(colmap_dir, unique_key)
        return scene 

@dataclasses.dataclass
class ColmapCustomMatch(SceneTransformOfflineDisk):
    """Do colmap calibration without any priors.
    """
    colmap_path: str = "colmap"
    use_gpu: bool | None = None

    def __call__(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = Path(scene.uri) / "__colmap"
        if self.file_flag_exists(colmap_dir):
            return scene

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
        self.write_file_flag(colmap_dir)
        return scene 

@dataclasses.dataclass
class ColmapBuiltinMatch(SceneTransformOfflineDisk):
    """Do colmap calibration without any priors.
    """
    colmap_path: str = "colmap"
    use_gpu: bool | None = None
    matcher: Literal["exhaustive", "sequential", "transitive", "vocab_tree", "spatial"] = "exhaustive"

    def __call__(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = Path(scene.uri) / "__colmap"
        unique_key = f"{self.matcher}"
        if self.file_flag_exists(colmap_dir, unique_key):
            return scene

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
        self.write_file_flag(colmap_dir, unique_key)
        return scene 


@dataclasses.dataclass
class ColmapMapper(SceneTransformOfflineDisk):
    """Do colmap calibration without any priors.
    """
    colmap_path: str = "colmap"
    is_hierarchical: bool = False
    ba_global_function_tolerance: float = 0.000001
    use_gpu: bool | None = None

    def __call__(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = Path(scene.uri) / "__colmap"
        workdirs = ColmapWorkDirs(colmap_dir, create=False)
        unique_key = f"{self.is_hierarchical}_{self.ba_global_function_tolerance}"
        if self.file_flag_exists(colmap_dir, unique_key):
            return scene
        this_work_dir = self.create_work_dir(colmap_dir, unique_key)
        colmap_args = [
            self.colmap_path, "hierarchical_mapper" if self.is_hierarchical else "mapper",
            "--database_path", f"{workdirs.database_path}",
            "--image_path", f"{workdirs.cameras_root}",
            "--output_path", str(this_work_dir),
            "--Mapper.ba_global_function_tolerance", str(self.ba_global_function_tolerance)
            ]
        use_gpu = self.use_gpu
        if use_gpu is None:
            use_gpu = not IsAppleSiliconMacOs and HAS_CUDA
        if use_gpu:
            assert not IsAppleSiliconMacOs, "gpu is not supported on apple silicon"
            colmap_args.append("--Mapper.ba_use_gpu")
            colmap_args.append("1")
        subprocess.run(colmap_args, check=True)
        this_work_dir_childs = list(this_work_dir.iterdir())
        print("Mapper Generated", len(this_work_dir_childs), "models")
        scene.set_user_data("model_output", str(this_work_dir / "0"))
        self.write_file_flag(colmap_dir, unique_key)
        return scene 

@dataclasses.dataclass
class ColmapCreateModelFromPrior(SceneTransform):
    colmap_path: str = "colmap"

@dataclasses.dataclass
class ColmapPointTriangulator(SceneTransformOfflineDisk):
    colmap_path: str = "colmap"
    def __call__(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = Path(scene.uri) / "__colmap"
        if self.file_flag_exists(colmap_dir):
            return scene

        workdirs = ColmapWorkDirs(colmap_dir, create=False)
        colmap_args = [
            self.colmap_path, "point_triangulator",
            "--database_path", f"{workdirs.database_path}",
            "--image_path", f"{workdirs.cameras_root}",
            "--input_path", f"{workdirs.sfm_prior_root}",
            "--output_path", str(workdirs.sfm_tri_root),
            ]
        subprocess.run(colmap_args, check=True)
        scene.set_user_data("model_output", str(workdirs.sfm_tri_root))
        self.write_file_flag(colmap_dir)
        return scene 

@dataclasses.dataclass
class ColmapBundleAdjustment(SceneTransformOfflineDisk):
    """assume you run point_triangulator first. (get a complete sparse model)
    """
    colmap_path: str = "colmap"
    use_gpu: bool | None = None
    def __call__(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = Path(scene.uri) / "__colmap"
        if self.file_flag_exists(colmap_dir):
            return scene
        workdirs = ColmapWorkDirs(colmap_dir, create=False)
        colmap_args = [
            self.colmap_path, "bundle_adjuster",
            "--database_path", f"{workdirs.database_path}",
            "--image_path", f"{workdirs.cameras_root}",
            "--input_path", f"{workdirs.sfm_prior_root}",
            "--output_path", str(workdirs.sfm_out_root),
            ]
        use_gpu = self.use_gpu
        if use_gpu is None:
            use_gpu = not IsAppleSiliconMacOs and HAS_CUDA
        if use_gpu:
            assert not IsAppleSiliconMacOs, "gpu is not supported on apple silicon"
            colmap_args.append("--BundleAdjustment.use_gpu")
            colmap_args.append("1")
        subprocess.run(colmap_args, check=True)
        scene.set_user_data("model_output", str(workdirs.sfm_out_root))
        self.write_file_flag(colmap_dir)
        return scene 

@dataclasses.dataclass
class ColmapUndistort(SceneTransformOfflineDisk):
    """assume you run all pipeline first. (get a complete sparse model)
    """
    colmap_path: str = "colmap"
    max_image_size: int = 2048
    def __call__(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = Path(scene.uri) / "__colmap"
        if self.file_flag_exists(colmap_dir):
            return scene
        workdirs = ColmapWorkDirs(colmap_dir, create=False)
        out_path = Path(scene.get_user_data_type_checked('model_output', str))
        colmap_args = [
            self.colmap_path, "image_undistorter",
            "--image_path", f"{workdirs.cameras_root}",
            "--input_path", f"{out_path}",
            "--output_path", str(workdirs.undistorted_img_root),
            "--output_type", "COLMAP",
            "--max_image_size", str(self.max_image_size),
        ]
        subprocess.run(colmap_args, check=True)
        scene.set_user_data("model_output", str(workdirs.undistorted_img_root / "sparse"))
        self.write_file_flag(colmap_dir)
        return scene 

@dataclasses.dataclass
class ColmapCollectResultModel(SceneTransformOfflineDisk):
    """save model from `model_output` in scene userdata to
    `workdirs.sfm_out_root`.
    """
    def __call__(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = Path(scene.uri) / "__colmap"
        if self.file_flag_exists(colmap_dir):
            return scene
        workdirs = ColmapWorkDirs(colmap_dir, create=False)
        out_path = Path(scene.get_user_data_type_checked('model_output', str))
        scene.set_user_data("model_output", str(workdirs.sfm_out_root))
        self.write_file_flag(colmap_dir)
        return scene 

def _copy_model(src: Path, dst: Path):
    img_bin0 = src / "images.bin"
    img_bin1 = dst / "images.bin"
    shutil.copy(img_bin0, img_bin1)
    cam_bin0 = src / "cameras.bin"
    cam_bin1 = dst / "cameras.bin"
    if cam_bin0.exists():
        shutil.copy(cam_bin0, cam_bin1)
    pts_bin0 = src / "points3D.bin"
    pts_bin1 = dst / "points3D.bin"
    if pts_bin0.exists():
        shutil.copy(pts_bin0, pts_bin1)

@dataclasses.dataclass
class ColmapFilterFloatingAndNoSFM(SceneTransformOfflineDisk):
    """remove floating cameras and cameras without sfm points
    """
    min_closest_dist_multipler: float = 10.0
    def __call__(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = Path(scene.uri) / "__colmap"
        unique_key = f"{self.min_closest_dist_multipler}"
        if self.file_flag_exists(colmap_dir, unique_key):
            return scene
        workdirs = ColmapWorkDirs(colmap_dir, create=False)
        out_path = Path(scene.get_user_data_type_checked('model_output', str))
        this_work_dir = self.create_work_dir(colmap_dir, unique_key)
        # copy model in out_path to this_work_dir
        # shutil.copytree(out_path, this_work_dir, dirs_exist_ok=True)
        _copy_model(out_path, this_work_dir)
        cam_extrinsics = cmu.read_images_binary(str(this_work_dir / "images.bin"))

        cam_centers = np.array([
            -cmu.qvec2rotmat(cam_extrinsics[key].qvec).T @ cam_extrinsics[key].tvec 
            for key in cam_extrinsics
        ])
        # TODO use cuda knn to replace these code 
        cam_knn = NearestNeighbors(n_neighbors=2).fit(cam_centers)        
        cam_to_closest_dist = []
        for key, cam_center in zip(cam_extrinsics, cam_centers):
            distances, indices = cam_knn.kneighbors(cam_center[None])
            cam_to_closest_dist.append(distances[0, -1])
        median_to_closest_dist = np.median(cam_to_closest_dist)
        res_cam_extrs = {}
        # TODO we need to filter intrinsics as well
        for key, second_min_distance in zip(cam_extrinsics, cam_to_closest_dist):
            cam_extr = cam_extrinsics[key]

            if len(cam_extr.point3D_ids) > 0 and second_min_distance <= self.min_closest_dist_multipler * median_to_closest_dist:
                valid_pts_mask = cam_extr.point3D_ids >= 0
                if valid_pts_mask.any():
                    res_cam_extrs[key] = cmu.Image(
                            id=cam_extr.id,
                            qvec=cam_extr.qvec,
                            tvec=cam_extr.tvec,
                            camera_id=cam_extr.camera_id,
                            name=cam_extr.name,
                            xys=cam_extr.xys[valid_pts_mask],
                            point3D_ids=cam_extr.point3D_ids[valid_pts_mask],
                        )
        print("Filtered out", len(cam_extrinsics), len(res_cam_extrs), "cameras")
        cmu.write_images_binary(res_cam_extrs, str(this_work_dir / "images.bin"))
        scene.set_user_data("model_output", str(this_work_dir))
        self.write_file_flag(colmap_dir)
        return scene 

@dataclasses.dataclass
class ColmapRefineRotAndScale(SceneTransformOfflineDisk):
    """refine cameras
    """
    target_closest_dist: float = 20.0

    def fit_plane_least_squares(self, points: np.ndarray):
        # TODO replace this with sklearn
        # Augment the point cloud with a column of ones
        A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
        B = points[:, 2]
        # Solve the least squares problem A * [a, b, c].T = B to get the plane equation z = a*x + b*y + c
        coefficients, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        
        # Plane coefficients: z = a*x + b*y + c
        a, b, c = coefficients
        
        # The normal vector is [a, b, -1]
        normal_vector = np.array([a, b, -1])
        normal_vector /= np.linalg.norm(normal_vector)  # Normalize the normal vector
        
        # An in-plane vector can be any vector orthogonal to the normal. One simple choice is:
        in_plane_vector = np.cross(normal_vector, np.array([0, 0, 1]))
        if np.linalg.norm(in_plane_vector) == 0:
            in_plane_vector = np.cross(normal_vector, np.array([0, 1, 0]))
        in_plane_vector /= np.linalg.norm(in_plane_vector)  # Normalize the in-plane vector
        
        return normal_vector, in_plane_vector, np.mean(points, axis=0)
    
    def rotate_camera(self, qvec, tvec, rot_matrix, upscale):
        # Assuming cameras have 'T' (translation) field
        R = cmu.qvec2rotmat(qvec)
        T = np.array(tvec)

        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R
        Rt[:3, 3] = T
        Rt[3, 3] = 1.0

        C2W = np.linalg.inv(Rt)
        cam_center = np.copy(C2W[:3, 3])
        cam_rot_orig = np.copy(C2W[:3, :3])
        cam_center = np.matmul(cam_center, rot_matrix)
        cam_rot = np.linalg.inv(rot_matrix) @ cam_rot_orig
        C2W[:3, 3] = upscale * cam_center
        C2W[:3, :3] = cam_rot
        Rt = np.linalg.inv(C2W)
        new_pos = Rt[:3, 3]
        new_rot = cmu.rotmat2qvec(Rt[:3, :3])
        return new_pos, new_rot

    def __call__(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = Path(scene.uri) / "__colmap"
        unique_key = f"{self.target_closest_dist}"
        if self.file_flag_exists(colmap_dir, unique_key):
            return scene
        workdirs = ColmapWorkDirs(colmap_dir, create=False)
        if scene.has_user_data("model_output"):
            out_path = Path(scene.get_user_data_type_checked('model_output', str))
        else:
            out_path = workdirs.sfm_out_root / "0"
        this_work_dir = self.create_work_dir(colmap_dir, unique_key)

        model = cmu.read_model(out_path)
        assert model is not None 
        cameras, images_metas_in, points3d_in = model
        median_distances = []
        for key in images_metas_in:
            image_meta = images_metas_in[key]
            cam_center = -cmu.qvec2rotmat(image_meta.qvec).astype(np.float32).T @ image_meta.tvec.astype(np.float32)
            
            median_distances.extend([
                np.linalg.norm(points3d_in[pt_idx].xyz - cam_center) for pt_idx in image_meta.point3D_ids if pt_idx != -1
            ])

        median_distance = np.median(np.array(median_distances))
        upscale = (self.target_closest_dist / median_distance)
        cam_centers = np.array([
            -cmu.qvec2rotmat(images_metas_in[key].qvec).T @ images_metas_in[key].tvec
            for key in images_metas_in
        ])
        candidates = cam_centers[scipy.spatial.ConvexHull(cam_centers).vertices]

        up, _, _ = self.fit_plane_least_squares(cam_centers)
        dist_mat = scipy.spatial.distance_matrix(candidates, candidates)

        # get indices of cameras that are furthest apart
        i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
        right = candidates[i] - candidates[j]
        right /= np.linalg.norm(right)
        
        up = torch.from_numpy(up).double()
        right = torch.from_numpy(right).double()

        forward = torch.cross(up, right)
        forward /= torch.norm(forward, p=2)

        right = torch.cross(forward, up)
        right /= torch.norm(right, p=2)

        # Stack the target axes as columns to form the rotation matrix
        rotation_matrix = torch.stack([right, forward, up], dim=1)
        positions = []
        print("Doing points")
        for key in points3d_in: 
            positions.append(points3d_in[key].xyz)
        
        positions = torch.from_numpy(np.array(positions))
        
        # Perform the rotation by matrix multiplication
        rotated_points = upscale * torch.matmul(positions, rotation_matrix)
        points3d_out = {}
        for key, rotated in zip(points3d_in, rotated_points):
            point3d_in = points3d_in[key]
            points3d_out[key] = cmu.Point3D(
                id=point3d_in.id,
                xyz=rotated,
                rgb=point3d_in.rgb,
                error=point3d_in.error,
                image_ids=point3d_in.image_ids,
                point2D_idxs=point3d_in.point2D_idxs,
            )
        images_metas_out = {} 
        for key in images_metas_in: 
            image_meta_in = images_metas_in[key]
            new_pos, new_rot = self.rotate_camera(image_meta_in.qvec, image_meta_in.tvec, rotation_matrix.double().numpy(), upscale)
            
            images_metas_out[key] = cmu.Image(
                id=image_meta_in.id,
                qvec=new_rot,
                tvec=new_pos,
                camera_id=image_meta_in.camera_id,
                name=image_meta_in.name,
                xys=image_meta_in.xys,
                point3D_ids=image_meta_in.point3D_ids,
            )

        cmu.write_model(cameras, images_metas_out, points3d_out, str(this_work_dir))
        scene.set_user_data("model_output", str(this_work_dir))
        self.write_file_flag(colmap_dir)
        return scene 

@dataclasses.dataclass
class ColmapFetchResult(SceneTransform):
    """assume you run all pipeline first. (get a complete sparse model)
    """
    remove_fail_cams: bool = True
    def __call__(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = Path(scene.uri) / "__colmap"
        workdirs = ColmapWorkDirs(colmap_dir, create=False)
        out_path = workdirs.sfm_out_root / "0"
        try:
            cameras_extrinsic_file = os.path.join(out_path, "images.bin")
            cameras_intrinsic_file = os.path.join(out_path, "cameras.bin")
            cam_extrinsics = cmu.read_images_binary(cameras_extrinsic_file)
            cam_intrinsics = cmu.read_cameras_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(out_path, "images.txt")
            cameras_intrinsic_file = os.path.join(out_path, "cameras.txt")
            cam_extrinsics = cmu.read_images_text(cameras_extrinsic_file)
            cam_intrinsics = cmu.read_cameras_text(cameras_intrinsic_file)
        gid_to_cams = scene.get_global_id_to_sensor(BasicPinholeCamera)
        cam_with_nonempty_pose_and_in_colmap: None | BasicPinholeCamera = None
        for idx, key in enumerate(cam_extrinsics):
            extr = cam_extrinsics[key]
            intr = cam_intrinsics[extr.camera_id]
            # our cam file name is always global id, so use this to set back to the original
            # camera.
            height = intr.height
            width = intr.width

            uid = intr.id
            R = cmu.qvec2rotmat(extr.qvec)
            T = np.array(extr.tvec)
            world2cam = np.eye(4)
            world2cam[:3, :3] = R
            world2cam[:3, 3] = T
            cam2world = np.linalg.inv(world2cam)
            distort = np.zeros(4)
            if intr.model=="SIMPLE_PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[0]
                cx = intr.params[1]
                cy = intr.params[2]
            elif intr.model=="PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                cx = intr.params[2]
                cy = intr.params[3]
            elif intr.model=="OPENCV":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                cx = intr.params[2]
                cy = intr.params[3]
                distort[0] = intr.params[4]
                distort[1] = intr.params[5]
                distort[2] = intr.params[6]
                distort[3] = intr.params[7]

            else:
                raise NotImplementedError("Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!")
            intrinsic = np.eye(4)
            intrinsic[0, 0] = focal_length_x
            intrinsic[1, 1] = focal_length_y
            intrinsic[0, 2] = cx
            intrinsic[1, 2] = cy
            cam_file_name = extr.name 
            cam_global_id = Path(cam_file_name).stem
            if cam_global_id in gid_to_cams:
                cam = gid_to_cams[cam_global_id]
                if not cam.pose.is_empty:
                    if cam_with_nonempty_pose_and_in_colmap is None:
                        cam_with_nonempty_pose_and_in_colmap = cam
                cam.pose.to_world = cam2world
                cam.intrinsic = intrinsic
                gid_to_cams.pop(cam_global_id)

            # breakpoint()
        if self.remove_fail_cams:
            scene.remove_sensors_inplace(list(gid_to_cams.values()))
            print(f"Removed {len(gid_to_cams)} cameras with no pose in colmap")
        breakpoint()
        return scene 

@dataclasses.dataclass
class ColmapFetchReferenceResult(SceneTransform):
    """assume you run all pipeline first. (get a complete sparse model)
    """
    remove_fail_cams: bool = True
    external_path: str | None = None
    def __call__(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        if self.external_path is not None:
            out_path = Path(self.external_path)
        else:
            colmap_dir = Path(scene.uri) / "__colmap"
            workdirs = ColmapWorkDirs(colmap_dir, create=False)
            out_path = workdirs.sfm_out_root / "0"
        try:
            cameras_extrinsic_file = os.path.join(out_path, "images.bin")
            cameras_intrinsic_file = os.path.join(out_path, "cameras.bin")
            cam_extrinsics = cmu.read_images_binary(cameras_extrinsic_file)
            cam_intrinsics = cmu.read_cameras_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(out_path, "images.txt")
            cameras_intrinsic_file = os.path.join(out_path, "cameras.txt")
            cam_extrinsics = cmu.read_images_text(cameras_extrinsic_file)
            cam_intrinsics = cmu.read_cameras_text(cameras_intrinsic_file)
        gid_to_cams = scene.get_global_id_to_sensor(BasicPinholeCamera)
        cam_with_nonempty_pose_and_in_colmap: None | BasicPinholeCamera = None
        for idx, key in enumerate(cam_extrinsics):
            extr = cam_extrinsics[key]
            intr = cam_intrinsics[extr.camera_id]
            # our cam file name is always global id, so use this to set back to the original
            # camera.
            height = intr.height
            width = intr.width

            uid = intr.id
            R = cmu.qvec2rotmat(extr.qvec)
            T = np.array(extr.tvec)
            world2cam = np.eye(4)
            world2cam[:3, :3] = R
            world2cam[:3, 3] = T
            cam2world = np.linalg.inv(world2cam)
            distort = np.zeros(4)
            if intr.model=="SIMPLE_PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[0]
                cx = intr.params[1]
                cy = intr.params[2]
            elif intr.model=="PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                cx = intr.params[2]
                cy = intr.params[3]
            elif intr.model=="OPENCV":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                cx = intr.params[2]
                cy = intr.params[3]
                distort[0] = intr.params[4]
                distort[1] = intr.params[5]
                distort[2] = intr.params[6]
                distort[3] = intr.params[7]

            else:
                raise NotImplementedError("Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!")
            intrinsic = np.eye(4)
            intrinsic[0, 0] = focal_length_x
            intrinsic[1, 1] = focal_length_y
            intrinsic[0, 2] = cx
            intrinsic[1, 2] = cy
            cam_file_name = extr.name 
            print(cam_file_name)
            breakpoint()
            cam_global_id = Path(cam_file_name).stem
            if cam_global_id in gid_to_cams:
                cam = gid_to_cams[cam_global_id]
                if not cam.pose.is_empty:
                    if cam_with_nonempty_pose_and_in_colmap is None:
                        cam_with_nonempty_pose_and_in_colmap = cam
                cam.pose.to_world = cam2world
                cam.intrinsic = intrinsic
                gid_to_cams.pop(cam_global_id)

            # breakpoint()
        if self.remove_fail_cams:
            scene.remove_sensors_inplace(list(gid_to_cams.values()))
            print(f"Removed {len(gid_to_cams)} cameras with no pose in colmap")
        breakpoint()
        return scene 

def __main_test_simpify():
    img_bin_folder = "/root/autodl-tmp/example_dataset/camera_calibration/unrectified"
    fake_scene = Scene("wtf", [])
    fake_scene.uri = img_bin_folder
    fake_scene.set_user_data("model_output", img_bin_folder)
    workdirs = ColmapWorkDirs(Path(img_bin_folder) / "__colmap", create=True)
    ColmapFilterFloatingAndNoSFM()(fake_scene)

def __main_test_refine():
    img_bin_folder = "/root/autodl-tmp/example_dataset/camera_calibration/rectified"
    fake_scene = Scene("wtf", [])
    fake_scene.uri = img_bin_folder
    fake_scene.set_user_data("model_output", str(Path(img_bin_folder) / "sparse"))
    workdirs = ColmapWorkDirs(Path(img_bin_folder) / "__colmap", create=True)
    ColmapRefineRotAndScale()(fake_scene)

def __main():
    from d3sim.algos.d3gs.hd3gs.data.load import _load_debug_scene
    scene = _load_debug_scene()
    scene = ColmapFeatureExtract(default_focal_length_factor=0.5)(scene)
    scene = ColmapCustomMatchGen()(scene)
    scene = ColmapCustomMatch()(scene)
    scene = ColmapMapper(is_hierarchical=True)(scene)
    scene = ColmapUndistort()(scene)
    # scene = ColmapFetchReferenceResult(external_path="/root/autodl-tmp/example_dataset/camera_calibration/rectified/sparse/")(scene)

if __name__ == "__main__":
    __main()
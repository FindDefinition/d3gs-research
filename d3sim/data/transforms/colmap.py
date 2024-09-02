from pathlib import Path
import tempfile
from typing import Literal
from d3sim import data
from d3sim.constants import IsAppleSiliconMacOs
from d3sim.core.unique_tree_id import UniqueTreeId
from d3sim.data.scene_def import Scene
from d3sim.data.scene_def.base import CameraFieldTypes, DistortType, Resource, SingleFileLoader
from d3sim.data.scene_def.camera import BasicPinholeCamera, OpencvImageFileLoader
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
import os 

class ColmapConstants:
    Meta = "__colmap_transform_meta"

def _run_colmap_subprocess(colmap_args: list[str]):
    print("[COLMAP]", " ".join(colmap_args))
    subprocess.run(colmap_args, check=True)

# disable colmap info logging
# os.environ["GLOG_minloglevel"] = "2"
@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class _SceneTransformMeta:
    colmap_model_path: str 
    colmap_image_root: str | None
    colmap_mask_root: str | None
    colmap_match_txt_path: str | None
    colmap_workdir_prefix: str = "__colmap" 

class ColmapMetaHandleBase:
    @staticmethod
    def get_colmap_workdir_root_from_scene(scene: Scene) -> Path:
        assert scene.uri is not None 
        meta = ColmapMetaHandleBase.get_colmap_meta_from_scene(scene)
        if meta is None:
            return Path(scene.uri) / "__colmap"
        else:
            return Path(scene.uri) / meta.colmap_workdir_prefix

    @staticmethod
    def get_colmap_meta_from_scene(scene: Scene) -> _SceneTransformMeta | None:
        if scene.has_user_data(ColmapConstants.Meta):
            return scene.get_user_data_type_checked(ColmapConstants.Meta, _SceneTransformMeta)
        return None

    @staticmethod
    def set_scene_meta(scene: Scene, meta: _SceneTransformMeta):
        return scene.set_user_data(ColmapConstants.Meta, meta)

    @staticmethod
    def get_colmap_meta_from_scene_required(scene: Scene) -> _SceneTransformMeta:
        if scene.has_user_data(ColmapConstants.Meta):
            return scene.get_user_data_type_checked(ColmapConstants.Meta, _SceneTransformMeta)
        raise ValueError("colmap meta not found")

    @staticmethod
    def store_colmap_model_path(scene: Scene, model_path: str):
        if not scene.has_user_data(ColmapConstants.Meta):
            scene.set_user_data(ColmapConstants.Meta, _SceneTransformMeta(model_path, None, None, None))
        else:
            meta = scene.get_user_data_type_checked(ColmapConstants.Meta, _SceneTransformMeta)
            meta = dataclasses.replace(meta, colmap_model_path=model_path)
            scene.set_user_data(ColmapConstants.Meta, meta)
        return scene

    @staticmethod
    def store_colmap_image_root(scene: Scene, image_root: str):
        if not scene.has_user_data(ColmapConstants.Meta):
            scene.set_user_data(ColmapConstants.Meta, _SceneTransformMeta("", image_root, None, None))
        else:
            meta = scene.get_user_data_type_checked(ColmapConstants.Meta, _SceneTransformMeta)
            meta = dataclasses.replace(meta, colmap_image_root=image_root)
            scene.set_user_data(ColmapConstants.Meta, meta)

        return scene

    @staticmethod
    def store_colmap_mask_root(scene: Scene, mask_root: str):
        if not scene.has_user_data(ColmapConstants.Meta):
            scene.set_user_data(ColmapConstants.Meta, _SceneTransformMeta("", None, mask_root, None))
        else:
            meta = scene.get_user_data_type_checked(ColmapConstants.Meta, _SceneTransformMeta)
            meta = dataclasses.replace(meta, colmap_mask_root=mask_root)
            scene.set_user_data(ColmapConstants.Meta, meta)

        return scene
    
    @staticmethod
    def store_colmap_match_txt_path(scene: Scene, match_txt_path: str):
        if not scene.has_user_data(ColmapConstants.Meta):
            scene.set_user_data(ColmapConstants.Meta, _SceneTransformMeta("", None, None, match_txt_path))
        else:
            meta = scene.get_user_data_type_checked(ColmapConstants.Meta, _SceneTransformMeta)
            meta = dataclasses.replace(meta, colmap_match_txt_path=match_txt_path)
            scene.set_user_data(ColmapConstants.Meta, meta)

        return scene

class ColmapSceneTransformOfflineDisk(ColmapMetaHandleBase, SceneTransformOfflineDisk):
    def __call__(self, scene: Scene) -> Scene:
        prev_colmap_meta = self.get_colmap_meta_from_scene(scene)
        if prev_colmap_meta is None:
            return self.forward(scene)
        res_scene = self.forward(scene)
        res_meta = self.get_colmap_meta_from_scene(res_scene)
        if res_meta is None:
            # set previous meta
            res_scene.set_user_data(ColmapConstants.Meta, prev_colmap_meta)
        else:
            # merge result
            merged = dataclasses.replace(prev_colmap_meta, **dataclasses.asdict(res_meta))
            res_scene.set_user_data(ColmapConstants.Meta, merged)
        return scene 

HAS_CUDA = torch.cuda.is_available()

@dataclasses.dataclass
class ColmapWorkDirs:
    colmap_dir: Path
    cameras_root: Path = Path()
    masks_root: Path = Path()
    sfm_out_root: Path = Path()

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
        self.undistorted_img_root = self.colmap_dir / "cam_calib/undistorted"
        self.undistorted_mask_root = self.colmap_dir / "cam_calib/undistorted_mask"
        self.database_path = self.colmap_dir / "database.db"
        self.match_list_path = self.colmap_dir / "matches.txt"

    def create_dir(self, path: Path):
        path.mkdir(exist_ok=True, parents=True, mode=0o755)
        return path

def _write_or_copy_cam_field(cam: BasicPinholeCamera, field_type: CameraFieldTypes, write_root: Path, suffix: str = ".jpg") -> tuple[str, Path] | None:
    assert field_type in [CameraFieldTypes.IMAGE, CameraFieldTypes.VALID_MASK]
    field_val = cam.fields[field_type]
    copy_pair = None
    if isinstance(field_val, Resource) and isinstance(field_val.loader, SingleFileLoader):
        loader_suffix = Path(field_val.base_uri).suffix
        copy_pair = (field_val.base_uri, write_root / f"{cam.global_id}{loader_suffix}")
        # shutil.copy(field_val.base_uri, write_root / f"{cam.global_id}{loader_suffix}")
    else:
        field_val = cam.get_field_np(field_type)
        assert field_val is not None 
        cv2.imwrite(str(write_root / f"{cam.global_id}{suffix}"), field_val)
    return copy_pair 

def _create_image_field_colmap_subfolder(cam_id: str, field_type: CameraFieldTypes, cameras: list[BasicPinholeCamera], cameras_root: Path, is_feature_ignore_mask: bool = False):
    subfolder = cameras_root / cam_id
    suffix = ".jpg"
    if field_type == CameraFieldTypes.VALID_MASK:
        if is_feature_ignore_mask:
            suffix = ".jpg.png"
        else:
            suffix = ".png"
    subfolder.mkdir(exist_ok=True, parents=True, mode=0o755)
    file_copy_pair = []
    for cam in cameras:
        pair = _write_or_copy_cam_field(cam, field_type, subfolder, suffix)
        if pair is not None:
            file_copy_pair.append(pair)
    for src, dst in file_copy_pair:
        shutil.copy(src, dst)
    return 

def _change_suffix_in_model_image_name(model_path: Path, dst_path: Path, suffix: str):
    dst_path.mkdir(exist_ok=True, parents=True, mode=0o755)
    _copy_model(model_path, dst_path)
    
    colmap_imgs = cmu.read_images_binary(model_path / "images.bin")
    out_images_metas = {}
    for key in colmap_imgs:
        in_image_meta = colmap_imgs[key]
        name_p = Path(in_image_meta.name)
        out_images_metas[key] = cmu.Image(
            id=key,
            qvec=in_image_meta.qvec,
            tvec=in_image_meta.tvec,
            camera_id=in_image_meta.camera_id,
            name=str(name_p.parent / f"{name_p.stem}{suffix}"),
            xys=in_image_meta.xys,
            point3D_ids=in_image_meta.point3D_ids,
        )
    cmu.write_images_binary(out_images_metas, str(dst_path / "images.bin"))
    return 


@dataclasses.dataclass
class ColmapFeatureExtract(ColmapSceneTransformOfflineDisk):
    colmap_path: str = "colmap"
    default_focal_length_factor: float | None = None
    use_mask_during_feature_extraction: bool = False
    from_existing_model: bool = False
    use_gpu: bool | None = None
    def get_unique_key(self) -> str:
        return f"{self.default_focal_length_factor}_{self.use_mask_during_feature_extraction}"

    def forward(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = self.get_colmap_workdir_root_from_scene(scene)
        # if colmap_dir.exists():
        #     shutil.rmtree(colmap_dir)
        workdirs = ColmapWorkDirs(colmap_dir, create=True)
        # if self.file_flag_exists(colmap_dir):
        #     return scene
        cam_id_to_cams = scene.get_sensor_id_to_sensors(BasicPinholeCamera)

        for cam_id, cams in cam_id_to_cams.items():
            _create_image_field_colmap_subfolder(cam_id, CameraFieldTypes.IMAGE, cams, workdirs.cameras_root)
            if cams[0].has_field(CameraFieldTypes.VALID_MASK):
                _create_image_field_colmap_subfolder(cam_id, CameraFieldTypes.VALID_MASK, cams, workdirs.masks_root, True)
        workdirs.create_dir(workdirs.cameras_root)
        workdirs.create_dir(workdirs.masks_root)

        colmap_feature_extractor_args = [
            self.colmap_path, "feature_extractor",
                "--database_path", f"{workdirs.database_path}",
                "--image_path", f"{workdirs.cameras_root}",
            ]
        if self.from_existing_model:
            assert workdirs.database_path.exists()
            meta = self.get_colmap_meta_from_scene_required(scene)
            assert Path(meta.colmap_model_path).exists()
            # assume cameras and intrinsics are already added to database.
            colmap_feature_extractor_args.append("--ImageReader.existing_camera_id")
            colmap_feature_extractor_args.append("1")
        else:
            colmap_feature_extractor_args.extend([
                "--ImageReader.single_camera_per_folder", "1",
                "--ImageReader.camera_model", "OPENCV",
            ])
        if self.use_mask_during_feature_extraction:
            colmap_feature_extractor_args.append("--mask_path")
            colmap_feature_extractor_args.append(str(workdirs.masks_root))
        if self.default_focal_length_factor is not None:
            colmap_feature_extractor_args.append("--ImageReader.default_focal_length_factor")
            colmap_feature_extractor_args.append(str(self.default_focal_length_factor))
        # use_gpu = self.use_gpu
        # if use_gpu is None:
        #     use_gpu = not IsAppleSiliconMacOs and HAS_CUDA
        # if use_gpu:
        #     assert not IsAppleSiliconMacOs, "gpu is not supported on apple silicon"
        #     colmap_feature_extractor_args.append("--SiftExtraction.use_gpu")
        #     colmap_feature_extractor_args.append("1")
        _run_colmap_subprocess(colmap_feature_extractor_args)
        # self.write_file_flag(colmap_dir)
        self.store_colmap_image_root(scene, str(workdirs.cameras_root))
        self.store_colmap_mask_root(scene, str(workdirs.masks_root))
        return scene 

@dataclasses.dataclass
class ColmapCreatePoseAndIntrinsicPrior(ColmapSceneTransformOfflineDisk):
    """Add intrinsic priors and create extrinsic priors
    """
    add_intrinsic_prior: bool = True
    add_pose_prior: bool = True
    # let colmap use the prior focal length
    prior_focal_length: bool = True
    def forward(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = self.get_colmap_workdir_root_from_scene(scene)
        # if self.file_flag_exists(colmap_dir):
        #     return scene
        workdirs = ColmapWorkDirs(colmap_dir, create=False)
        db = cmu.COLMAPDatabase.connect(str(workdirs.database_path))
        cam_id_to_cams = scene.get_sensor_id_to_sensors(BasicPinholeCamera)
        # read db, get d3sim cam global id to colmap cam global id map
        cam_global_id_to_colmap_id = {}
        for row in  db.execute("SELECT * FROM images"):
            (image_id, cam_file_name, camera_id) = row
            cam_global_id = Path(cam_file_name).stem
            cam_global_id_to_colmap_id[cam_global_id] = (image_id, camera_id, cam_file_name)
        this_work_dir = self.delete_and_create_work_dir(colmap_dir)

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
        with open(this_work_dir / "points3D.txt", "w") as f:
            f.write("")
        cmu.write_cameras_text(res_colmap_cams, str(this_work_dir / "cameras.txt"))
        cmu.write_images_text(res_colmap_images, str(this_work_dir / "images.txt"))
        self.store_colmap_model_path(scene, str(this_work_dir))
        self.write_file_flag(colmap_dir)
        return scene 

@dataclasses.dataclass
class ColmapCreatePriorFromModel(ColmapSceneTransformOfflineDisk):
    """Add priors from previous colmap model.
    """
    # let colmap use the prior focal length
    prior_focal_length: bool = False
    def forward(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = self.get_colmap_workdir_root_from_scene(scene)
        if not colmap_dir.exists():
            colmap_dir.mkdir(exist_ok=True, parents=True, mode=0o755)
        model_dir = colmap_dir / "prior_model"
        model_dir.mkdir(exist_ok=True, parents=True, mode=0o755)
        if (colmap_dir / "database.db").exists() and (model_dir / "images.bin").exists():
            return scene
        # database must be removed before we can add new cameras
        (colmap_dir / "database.db").unlink(missing_ok=True)
        model_output = Path(self.get_colmap_meta_from_scene_required(scene).colmap_model_path)
        print("???", colmap_dir, model_output)

        cam_intrinsics = cmu.read_cameras_binary(model_output / "cameras.bin")
        images_metas = cmu.read_images_binary(model_output / "images.bin")
        db = cmu.COLMAPDatabase.connect(str(colmap_dir / "database.db"))
        db.create_tables()

        for key in cam_intrinsics:
            cam = cam_intrinsics[key]
            db.add_camera(cmu.CAMERA_MODEL_NAMES[cam.model].model_id, cam.width, cam.height, cam.params, camera_id=key, prior_focal_length=self.prior_focal_length)
        new_img_metas = {}
        for key in images_metas:
            image_meta = images_metas[key]
            db.add_image(image_meta.name, image_meta.camera_id, image_id=key)
            # clear sfm point metas
            image_meta = image_meta._replace(point3D_ids=[])
            image_meta = image_meta._replace(xys=[])
            new_img_metas[key] = image_meta
        db.commit()
        shutil.copy(model_output / "cameras.bin", model_dir / "cameras.bin")
        cmu.write_images_binary(new_img_metas, str(model_dir / "images.bin"))
        cmu.write_points3D_binary({}, str(model_dir / "points3D.bin"))
        self.store_colmap_model_path(scene, str(model_dir))
        return scene 

@dataclasses.dataclass
class ColmapCustomMatchGen(ColmapSceneTransformOfflineDisk):
    num_sequence_match: int = 0
    num_quad_sequence_match: int = 10
    def get_unique_key(self) -> str:
        return f"{self.num_sequence_match}_{self.num_quad_sequence_match}"

    def forward(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = self.get_colmap_workdir_root_from_scene(scene)
        # if self.file_flag_exists(colmap_dir):
        #     return scene
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
        match_list_path = workdirs.match_list_path
        self.store_colmap_match_txt_path(scene, str(match_list_path))
        with open(match_list_path, "w") as f:
            for pair in all_matchs:
                f.write(f"{pair[0]} {pair[1]}\n")
        self.write_file_flag(colmap_dir)
        return scene 


@dataclasses.dataclass
class ColmapPriorModelCustomMatchKNN(ColmapSceneTransformOfflineDisk):
    k: int = 200
    def get_unique_key(self) -> str:
        return f"{self.k}"

    def forward(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = self.get_colmap_workdir_root_from_scene(scene)
        # if self.file_flag_exists(colmap_dir):
        #     return scene
        model_input = Path(self.get_colmap_meta_from_scene_required(scene).colmap_model_path)
        images_metas = cmu.read_images_binary(f"{model_input}/images.bin")

        cam_centers = np.array([
            -cmu.qvec2rotmat(images_metas[key].qvec).astype(np.float32).T @ images_metas[key].tvec.astype(np.float32) 
            for key in images_metas
        ])
        n_neighbours = min(self.k, len(cam_centers))
        cam_nbrs = NearestNeighbors(n_neighbors=n_neighbours).fit(cam_centers)
        all_matchs: list[tuple[str, str]] = []
        keys = list(images_metas.keys())
        for key, cam_center in zip(images_metas, cam_centers):
            _, indices = cam_nbrs.kneighbors(cam_center[None])
            for idx in indices[0, 1:]:
                all_matchs.append((images_metas[key].name, images_metas[keys[idx]].name))
        this_work_dir = self.delete_and_create_work_dir(colmap_dir)

        match_list_path = this_work_dir / "matches.txt"
        with open(match_list_path, "w") as f:
            for pair in all_matchs:
                f.write(f"{pair[0]} {pair[1]}\n")
        self.store_colmap_match_txt_path(scene, str(match_list_path))
        return scene 

@dataclasses.dataclass
class ColmapCopySubsetImageFromModel(ColmapSceneTransformOfflineDisk):
    def forward(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = self.get_colmap_workdir_root_from_scene(scene)
        workdirs = ColmapWorkDirs(colmap_dir, create=False)
        workdirs_img_root = workdirs.cameras_root
        print(workdirs_img_root, "workdirs_img_root")
        workdirs.create_dir(workdirs_img_root)
        # if self.file_flag_exists(colmap_dir):
        #     return scene
        model_input = Path(self.get_colmap_meta_from_scene_required(scene).colmap_model_path)
        images_metas = cmu.read_images_binary(f"{model_input}/images.bin")
        img_root = self.get_colmap_meta_from_scene_required(scene).colmap_image_root
        assert img_root is not None, "image root must be set"
        img_root_p = Path(img_root)
        for key in images_metas:
            name_p = Path(images_metas[key].name)
            path = workdirs_img_root / name_p
            if not path.parent.exists():
                path.parent.mkdir(exist_ok=True, parents=True, mode=0o755)
            shutil.copy(img_root_p / name_p, path)
        
        self.store_colmap_image_root(scene, str(workdirs_img_root))
        return scene 

@dataclasses.dataclass
class ColmapCustomMatch(ColmapSceneTransformOfflineDisk):
    """Do colmap calibration without any priors.
    """
    colmap_path: str = "colmap"
    use_gpu: bool | None = None

    def forward(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = self.get_colmap_workdir_root_from_scene(scene)
        # if self.file_flag_exists(colmap_dir):
        #     return scene
        workdirs = ColmapWorkDirs(colmap_dir, create=False)
        meta = self.get_colmap_meta_from_scene(scene)
        if meta is None or meta.colmap_match_txt_path is None:
            match_list_path = str(workdirs.match_list_path)
        else:
            match_list_path = meta.colmap_match_txt_path
        colmap_matches_importer_args = [
            self.colmap_path, "matches_importer",
            "--database_path", str(workdirs.database_path),
            "--match_list_path", str(match_list_path),
        ]
        # use_gpu = self.use_gpu
        # if use_gpu is None:
        #     use_gpu = not IsAppleSiliconMacOs and HAS_CUDA
        # if use_gpu:
        #     assert not IsAppleSiliconMacOs, "gpu is not supported on apple silicon"
        #     colmap_matches_importer_args.append("--SiftMatching.use_gpu")
        #     colmap_matches_importer_args.append("1")
        _run_colmap_subprocess(colmap_matches_importer_args)
        self.write_file_flag(colmap_dir)
        return scene 

@dataclasses.dataclass(kw_only=True)
class ColmapCopyResultToFolder(ColmapSceneTransformOfflineDisk):
    """copy colmap pipeline to a folder
    """
    subfolder: str
    def forward(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = Path(scene.uri) / self.subfolder
        meta = self.get_colmap_meta_from_scene(scene)
        assert meta is not None 
        out_path = Path(meta.colmap_model_path)
        colmap_dir.mkdir(exist_ok=True, parents=True, mode=0o755)
        _copy_model(out_path, colmap_dir)
        return scene

@dataclasses.dataclass
class ColmapBuiltinMatch(ColmapSceneTransformOfflineDisk):
    """Do colmap calibration without any priors.
    """
    colmap_path: str = "colmap"
    use_gpu: bool | None = None
    matcher: Literal["exhaustive", "sequential", "transitive", "vocab_tree", "spatial"] = "exhaustive"
    def get_unique_key(self) -> str:
        return f"{self.matcher}"

    def forward(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = self.get_colmap_workdir_root_from_scene(scene)
        unique_key = f"{self.matcher}"
        # if self.file_flag_exists(colmap_dir):
        #     return scene

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
        _run_colmap_subprocess(colmap_matches_importer_args)
        self.write_file_flag(colmap_dir)
        return scene 


@dataclasses.dataclass
class ColmapMapper(ColmapSceneTransformOfflineDisk):
    """Do colmap calibration without any priors.
    """
    colmap_path: str = "colmap"
    is_hierarchical: bool = False
    ba_global_function_tolerance: float = 0.000001
    use_gpu: bool | None = None
    def get_unique_key(self) -> str:
        return f"{self.is_hierarchical}_{self.ba_global_function_tolerance}"

    def forward(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = self.get_colmap_workdir_root_from_scene(scene)
        workdirs = ColmapWorkDirs(colmap_dir, create=False)
        this_work_dir = self.get_work_dir_path(colmap_dir)
        self.store_colmap_model_path(scene, str(this_work_dir / "0"))
        # scene.set_user_data(ColmapConstants.ModelOutput, str(this_work_dir / "0"))
        # _copy_model(this_work_dir / "0", workdirs.sfm_out_root)
        workdirs.create_dir(workdirs.sfm_out_root)
        # if self.file_flag_exists(colmap_dir):
        #     return scene
        self.delete_and_create_work_dir(colmap_dir)
        # workdir save immediate results
        colmap_args = [
            self.colmap_path, "hierarchical_mapper" if self.is_hierarchical else "mapper",
            "--database_path", f"{workdirs.database_path}",
            "--image_path", f"{workdirs.cameras_root}",
            "--output_path", str(this_work_dir),
            "--Mapper.ba_global_function_tolerance", str(self.ba_global_function_tolerance)
            ]
        # use_gpu = self.use_gpu
        # if use_gpu is None:
        #     use_gpu = not IsAppleSiliconMacOs and HAS_CUDA
        # if use_gpu:
        #     assert not IsAppleSiliconMacOs, "gpu is not supported on apple silicon"
        #     colmap_args.append("--Mapper.ba_use_gpu")
        #     colmap_args.append("1")
        _run_colmap_subprocess(colmap_args)
        this_work_dir_childs = list(this_work_dir.iterdir())
        print("Mapper Generated", len(this_work_dir_childs) - 1, "models")
        # copy the first model to the root
        self.store_colmap_model_path(scene, str(this_work_dir / "0"))
        _copy_model(this_work_dir / "0", workdirs.sfm_out_root)
        self.write_file_flag(colmap_dir)
        return scene 

@dataclasses.dataclass
class ColmapCreateModelFromPrior(SceneTransform):
    colmap_path: str = "colmap"

"""
    if args.skip_bundle_adjustment:
        subprocess.run([colmap_exe, "point_triangulator",
            "--Mapper.ba_global_max_num_iterations", "5",
            "--Mapper.ba_global_max_refinements", "1", 
            "--database_path", f"{bundle_adj_chunk}/database.db",
            "--image_path", f"{bundle_adj_chunk}/images",
            "--input_path", f"{bundle_adj_chunk}/sparse/o",
            "--output_path", f"{bundle_adj_chunk}/sparse/0",
            ], check=True)
    else:
        colmap_point_triangulator_args = [
            colmap_exe, "point_triangulator",
            "--Mapper.ba_global_function_tolerance", "0.000001",
            "--Mapper.ba_global_max_num_iterations", "30",
            "--Mapper.ba_global_max_refinements", "3",
            ]

        colmap_bundle_adjuster_args = [
            colmap_exe, "bundle_adjuster",
            "--BundleAdjustment.refine_extra_params", "0",
            "--BundleAdjustment.function_tolerance", "0.000001",
            "--BundleAdjustment.max_linear_solver_iterations", "100",
            "--BundleAdjustment.max_num_iterations", "50", 
            "--BundleAdjustment.refine_focal_length", "0"
            ]


"""

@dataclasses.dataclass
class ColmapPointTriangulator(ColmapSceneTransformOfflineDisk):
    colmap_path: str = "colmap"
    ba_global_max_num_iterations: int = 30
    ba_global_max_refinements: int = 3
    ba_global_function_tolerance: float | None = None
    external_output_path: str | None = None
    def forward(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = self.get_colmap_workdir_root_from_scene(scene)
        # if self.file_flag_exists(colmap_dir):
        #     return scene
        model_input = self.get_colmap_meta_from_scene_required(scene).colmap_model_path
        workdirs = ColmapWorkDirs(colmap_dir, create=False)
        if self.external_output_path is not None:
            this_work_dir = Path(self.external_output_path)
            assert this_work_dir.exists(), "external output path must exist"
        else:
            this_work_dir = self.delete_and_create_work_dir(colmap_dir)
        colmap_args = [
            self.colmap_path, "point_triangulator",
            "--database_path", f"{workdirs.database_path}",
            "--image_path", f"{workdirs.cameras_root}",
            "--input_path", f"{model_input}",
            "--output_path", str(this_work_dir),
            ]
        colmap_args.extend([
            "--Mapper.ba_global_max_num_iterations", str(self.ba_global_max_num_iterations),
            "--Mapper.ba_global_max_refinements", str(self.ba_global_max_refinements),
        ])
        if self.ba_global_function_tolerance is not None:
            colmap_args.append("--Mapper.ba_global_function_tolerance")
            colmap_args.append(str(self.ba_global_function_tolerance))
        _run_colmap_subprocess(colmap_args)
        self.store_colmap_model_path(scene, str(this_work_dir))
        self.write_file_flag(colmap_dir)
        return scene 


@dataclasses.dataclass
class ColmapBundleAdjustment(ColmapSceneTransformOfflineDisk):
    """assume you run point_triangulator first. (get a complete sparse model)
    """
    colmap_path: str = "colmap"
    use_gpu: bool | None = None
    refine_extra_params: bool = False
    function_tolerance: float = 0.000001
    max_linear_solver_iterations: int = 100
    max_num_iterations: int = 50
    refine_focal_length: bool = False
    external_output_path: str | None = None

    def forward(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = self.get_colmap_workdir_root_from_scene(scene)
        # if self.file_flag_exists(colmap_dir):
        #     return scene
        model_input = self.get_colmap_meta_from_scene_required(scene).colmap_model_path
        if self.external_output_path is not None:
            this_work_dir = Path(self.external_output_path)
            assert this_work_dir.exists(), "external output path must exist"
        else:
            this_work_dir = self.delete_and_create_work_dir(colmap_dir)
        colmap_args = [
            self.colmap_path, "bundle_adjuster",
            "--input_path", f"{model_input}",
            "--output_path", str(this_work_dir),
            ]
        colmap_args.extend([
            "--BundleAdjustment.refine_extra_params", str(int(self.refine_extra_params)),
            "--BundleAdjustment.function_tolerance", str(self.function_tolerance),
            "--BundleAdjustment.max_linear_solver_iterations", str(self.max_linear_solver_iterations),
            "--BundleAdjustment.max_num_iterations", str(self.max_num_iterations), 
            "--BundleAdjustment.refine_focal_length", str(int(self.refine_focal_length)),
        ])
        _run_colmap_subprocess(colmap_args)
        self.store_colmap_model_path(scene, str(this_work_dir))
        self.write_file_flag(colmap_dir)
        return scene 

@dataclasses.dataclass(kw_only=True)
class ColmapTBALoop(ColmapSceneTransformOfflineDisk):
    """assume you run point_triangulator first. (get a complete sparse model)
    """
    colmap_path: str = "colmap"
    num_loop: int = 2
    point_triangulator: ColmapPointTriangulator
    bundle_adjustment: ColmapBundleAdjustment

    def forward(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = self.get_colmap_workdir_root_from_scene(scene)
        this_work_dir = self.delete_and_create_work_dir(colmap_dir)
        scene_in_loop = scene
        for j in range(self.num_loop):
            pt_out_tmp = this_work_dir / f"pt_tmp_{j}"
            pt_out_tmp.mkdir(exist_ok=True, parents=True, mode=0o755)
            ba_out_tmp = this_work_dir / f"ba_tmp_{j}"
            ba_out_tmp.mkdir(exist_ok=True, parents=True, mode=0o755)
            point_triangulator = dataclasses.replace(self.point_triangulator, external_output_path=str(pt_out_tmp))
            bundle_adjustment = dataclasses.replace(self.bundle_adjustment, external_output_path=str(ba_out_tmp))
            scene_in_loop = point_triangulator(scene_in_loop) 
            scene_in_loop = bundle_adjustment(scene_in_loop)
        out_path = this_work_dir / "tba_model"
        out_path.mkdir(exist_ok=True, parents=True, mode=0o755)
        out_model_path = self.get_colmap_meta_from_scene_required(scene_in_loop).colmap_model_path
        _copy_model(Path(out_model_path), out_path)
        self.store_colmap_model_path(scene, str(out_path))
        return scene

@dataclasses.dataclass
class OpencvUndistort(ColmapSceneTransformOfflineDisk):
    """TODO
    """

@dataclasses.dataclass
class ColmapUndistort(ColmapSceneTransformOfflineDisk):
    """assume fetch result is runned first. means intr and extr of scene cameras 
    are created/refined.
    """
    colmap_path: str = "colmap"
    max_image_size: int = 2048
    def get_unique_key(self) -> str:
        return f"{self.max_image_size}"

    def forward(self, scene: Scene[BasicFrame]):
        # TODO use opencv remap for segmentation (undistort without interpolation)
        assert scene.uri is not None 
        colmap_dir = self.get_colmap_workdir_root_from_scene(scene)
        assert scene.has_user_data("colmap_result_fetched"), "colmap result must be fetched first"
        workdirs = ColmapWorkDirs(colmap_dir, create=False)
        gid_to_cams = scene.get_global_id_to_sensor(BasicPinholeCamera)
        this_work_dir = self.delete_and_create_work_dir(colmap_dir)
        has_mask = next(iter(gid_to_cams.values())).has_field(CameraFieldTypes.VALID_MASK)
        if not self.file_flag_exists(colmap_dir) or True:
            out_path = Path(self.get_colmap_meta_from_scene_required(scene).colmap_model_path)
            cam_id_to_cams = scene.get_sensor_id_to_sensors(BasicPinholeCamera)
            workdirs.create_dir(workdirs.undistorted_img_root)
            if not workdirs.cameras_root.exists():
                for cam_id, cams in cam_id_to_cams.items():
                    _create_image_field_colmap_subfolder(cam_id, CameraFieldTypes.IMAGE, cams, workdirs.cameras_root)
            colmap_args = [
                self.colmap_path, "image_undistorter",
                "--image_path", f"{workdirs.cameras_root}",
                "--input_path", f"{out_path}",
                "--output_path", str(this_work_dir),
                "--output_type", "COLMAP",
                "--max_image_size", str(self.max_image_size),
            ]
            _run_colmap_subprocess(colmap_args)

            if has_mask:
                with tempfile.TemporaryDirectory() as tmpdir:
                    mask_root = Path(this_work_dir) / "masks"
                    for cam_id, cams in cam_id_to_cams.items():
                        _create_image_field_colmap_subfolder(cam_id, CameraFieldTypes.VALID_MASK, cams, mask_root)
                    _change_suffix_in_model_image_name(out_path, Path(this_work_dir / "masks"), ".png")
                    colmap_args = [
                        self.colmap_path, "image_undistorter",
                        "--image_path", f"{mask_root}",
                        "--input_path", f"{this_work_dir / "masks"}",
                        "--output_path", str(this_work_dir / "masks_model"),
                        "--output_type", "COLMAP",
                        "--max_image_size", str(self.max_image_size),
                    ]
                _run_colmap_subprocess(colmap_args)
        remain_gid_to_cams, _ = fetch_colmap_camera_result_and_assign_to_scene(str(this_work_dir / "sparse"), scene, assign_pose=False)
        assert len(remain_gid_to_cams) == 0
        for gid, cam in gid_to_cams.items():
            assert not cam.has_field(CameraFieldTypes.SEGMENTATION), "colmap undistort don't support segmentation"
            distort_img_path = this_work_dir / "images" / cam.id / f"{gid}.jpg"
            cam.set_field_np(CameraFieldTypes.IMAGE, Resource(
                base_uri=str(distort_img_path),
                loader_type="OpencvImageFileLoader",
                loader=OpencvImageFileLoader(),
            ))
            if has_mask:
                distort_mask_path = this_work_dir / "images" / cam.id / f"{gid}.png"
                cam.set_field_np(CameraFieldTypes.VALID_MASK, Resource(
                    base_uri=str(distort_mask_path),
                    loader_type="OpencvImageFileLoader",
                    loader=OpencvImageFileLoader(),
                ))
            cam.distortion[:] = 0
        # copy the first model to the root
        # _copy_model(workdirs.sfm_out_root, workdirs.undistorted_img_root / "sparse")
        self.store_colmap_model_path(scene, str(this_work_dir / "sparse"))
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
class ColmapFilterFloatingAndNoSFM(ColmapSceneTransformOfflineDisk):
    """remove floating cameras and cameras without sfm points
    """
    min_closest_dist_multipler: float = 10.0
    def get_unique_key(self) -> str:
        return f"{self.min_closest_dist_multipler}"

    def forward(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = self.get_colmap_workdir_root_from_scene(scene)
        meta = self.get_colmap_meta_from_scene(scene)
        assert meta is not None 
        out_path = Path(meta.colmap_model_path)
        this_work_dir = self.delete_and_create_work_dir(colmap_dir)
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
        self.store_colmap_model_path(scene, str(this_work_dir))
        self.write_file_flag(colmap_dir)
        return scene 

@dataclasses.dataclass
class ColmapRefineRotAndScale(ColmapSceneTransformOfflineDisk):
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

    def get_unique_key(self) -> str:
        return f"{self.target_closest_dist}"

    def forward(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = self.get_colmap_workdir_root_from_scene(scene)
        this_work_dir = self.get_work_dir_path(colmap_dir)
        workdirs = ColmapWorkDirs(colmap_dir, create=False)
        meta = self.get_colmap_meta_from_scene(scene)
        if meta is not None:
            out_path = Path(meta.colmap_model_path)
        else:
            out_path = workdirs.sfm_out_root
        print("!!!", out_path)
        self.store_colmap_model_path(scene, str(this_work_dir))

        # if self.file_flag_exists(colmap_dir):

        #     return scene
        self.delete_and_create_work_dir(colmap_dir)

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
        # _copy_model(this_work_dir, workdirs.sfm_out_root)
        self.write_file_flag(colmap_dir)
        return scene 

def fetch_colmap_camera_result_and_assign_to_scene(model_path: str, scene: Scene, assign_pose: bool = True):
    try:
        cameras_extrinsic_file = os.path.join(model_path, "images.bin")
        cameras_intrinsic_file = os.path.join(model_path, "cameras.bin")
        cam_extrinsics = cmu.read_images_binary(cameras_extrinsic_file)
        cam_intrinsics = cmu.read_cameras_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(model_path, "images.txt")
        cameras_intrinsic_file = os.path.join(model_path, "cameras.txt")
        cam_extrinsics = cmu.read_images_text(cameras_extrinsic_file)
        cam_intrinsics = cmu.read_cameras_text(cameras_intrinsic_file)
    gid_to_cams = scene.get_global_id_to_sensor(BasicPinholeCamera)
    # assert len(gid_to_cams) == len(cam_extrinsics), "you must use fetch result to filter invalid cameras first"
    colmap_world_to_origin: None | np.ndarray = None
    for idx, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        # our cam file name is always global id, so use this to set back to the original
        # camera.
        height = intr.height
        width = intr.width

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
            cam.image_shape_wh = (width, height)
            if assign_pose:
                prev_to_world = cam.pose.to_world
                cam.pose.to_world = cam2world
                if not cam.pose.is_empty:
                    if colmap_world_to_origin is None:
                        colmap_world_to_origin = prev_to_world @ np.linalg.inv(cam2world)
            cam.intrinsic = intrinsic
            gid_to_cams.pop(cam_global_id)
    return gid_to_cams, colmap_world_to_origin

@dataclasses.dataclass
class ColmapFetchResult(SceneTransform):
    """Fetch camera results from colmap model.
    assume you run all pipeline first. (get a complete sparse model)
    """
    remove_fail_cams: bool = True
    external_model_path: None | str = None
    def forward(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        assert not scene.has_user_data("colmap_result_fetched"), "you can't fetch result twice"
        colmap_dir = ColmapMetaHandleBase.get_colmap_workdir_root_from_scene(scene)
        workdirs = ColmapWorkDirs(colmap_dir, create=False)
        if self.external_model_path is not None:
            out_path = Path(self.external_model_path)
        else:
            meta = ColmapMetaHandleBase.get_colmap_meta_from_scene(scene)
            if meta is not None:
                out_path = Path(meta.colmap_model_path)
            else:
                out_path = workdirs.sfm_out_root
        gid_to_cams, colmap_world_to_origin = fetch_colmap_camera_result_and_assign_to_scene(str(out_path), scene)

        if self.remove_fail_cams:
            scene.remove_sensors_inplace(list(gid_to_cams.values()))
            print(f"Removed {len(gid_to_cams)} cameras with no pose in colmap")
        if colmap_world_to_origin is not None:
            scene.apply_world_transform_inplace(colmap_world_to_origin)
        # breakpoint()
        scene.set_user_data("colmap_result_fetched", True)
        print("!!!", out_path)
        ColmapMetaHandleBase.store_colmap_model_path(scene, str(out_path))
        # scene.set_user_data(ColmapConstants.ModelOutput, str(out_path))
        return scene 

@dataclasses.dataclass
class ColmapFromRefUnrectifiedResult(ColmapSceneTransformOfflineDisk):
    """convert camera results from h3dgs result to our scene format
    """
    ref_unrectified_result_path: str
    def forward(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        assert not scene.has_user_data("colmap_result_fetched"), "you can't fetch result twice"
        colmap_dir = self.get_colmap_workdir_root_from_scene(scene)
        workdirs = ColmapWorkDirs(colmap_dir, create=False)
        this_work_dir = self.delete_and_create_work_dir(colmap_dir)
        ref_path = Path(self.ref_unrectified_result_path)
        cameras_extrinsic_file = os.path.join(self.ref_unrectified_result_path, "images_heavy.bin")
        cam_extrinsics = cmu.read_images_binary(cameras_extrinsic_file)
        new_cam_extrinsics = {}
        gid_to_cameras = scene.get_global_id_to_sensor(BasicPinholeCamera)
        for idx, key in enumerate(cam_extrinsics):
            extr = cam_extrinsics[key]
            name = extr.name # format: cam_id/frame_id
            cam_id = Path(name).parts[0]
            frame_id = Path(name).stem 
            suffix = Path(name).suffix
            global_id = UniqueTreeId.from_parts([scene.id, frame_id, cam_id])
            assert global_id.uid_encoded in gid_to_cameras, f"camera {global_id.uid_encoded} not found in scene"
            our_path = Path(cam_id) / f"{global_id.uid_encoded}{suffix}"
            # print(type(extr))
            extr = extr._replace(name=str(our_path))
            new_cam_extrinsics[key] = extr
        cmu.write_images_binary(new_cam_extrinsics, str(this_work_dir / "images.bin"))
        shutil.copy(ref_path / "cameras.bin", this_work_dir / "cameras.bin")
        shutil.copy(ref_path / "points3D.bin", this_work_dir / "points3D.bin")
        self.store_colmap_model_path(scene, str(this_work_dir))
        # scene.set_user_data(ColmapConstants.ModelOutput, str(this_work_dir))
        return scene 

@dataclasses.dataclass
class ColmapToRefUnrectifiedResult(ColmapSceneTransformOfflineDisk):
    """Only exists for debug purpose
    """
    def forward(self, scene: Scene[BasicFrame]):
        assert scene.uri is not None 
        colmap_dir = self.get_colmap_workdir_root_from_scene(scene)
        workdirs = ColmapWorkDirs(colmap_dir, create=False)
        this_work_dir = self.delete_and_create_work_dir(colmap_dir)
        out_path = Path(self.get_colmap_meta_from_scene_required(scene).colmap_model_path)
        # out_path = Path(scene.get_user_data_type_checked(ColmapConstants.ModelOutput, str))

        cameras_extrinsic_file = os.path.join(out_path, "images.bin")
        cam_extrinsics = cmu.read_images_binary(cameras_extrinsic_file)
        new_cam_extrinsics = {}
        for idx, key in enumerate(cam_extrinsics):
            extr = cam_extrinsics[key]
            name = extr.name # format: cam_id/global_id
            cam_id = Path(name).parts[0]
            global_id = Path(name).stem 
            global_id_parts = UniqueTreeId(global_id).parts
            frame_id = global_id_parts[1]
            suffix = Path(name).suffix
            our_path = Path(cam_id) / f"{frame_id}{suffix}"
            # print(type(extr))
            extr = extr._replace(name=str(our_path))
            new_cam_extrinsics[key] = extr
        cmu.write_images_binary(new_cam_extrinsics, str(this_work_dir / "images.bin"))
        shutil.copy(out_path / "cameras.bin", this_work_dir / "cameras.bin")
        shutil.copy(out_path / "points3D.bin", this_work_dir / "points3D.bin")
        self.store_colmap_model_path(scene, str(this_work_dir))
        return scene 


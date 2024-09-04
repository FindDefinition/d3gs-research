from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import random
from typing import Annotated, Any

import tqdm
from d3sim.core import dataclass_dispatch as dataclasses
from d3sim.core.pytorch.hmt import HomogeneousTensor
from d3sim.data.scene_def.base import Scene
from d3sim.data.scene_def.frame import BasicFrame
from d3sim.data.transforms.colmap import ColmapConstants, ColmapMetaHandleBase
import d3sim.data.transforms.colmap_util as cmu
import torch
import numpy as np
import cv2
from functools import partial
import os
from d3sim.core import arrcheck
import hashlib

def get_nb_pts(image_metas):
    n_pts = 0
    for key in image_metas:
        pts_idx = image_metas[key].point3D_ids
        if (len(pts_idx) > 5):
            n_pts = max(n_pts, np.max(pts_idx))

    return n_pts + 1


def get_var_of_laplacian(image_id, images_dir, image_metas):
    image = cv2.imread(os.path.join(images_dir, image_metas[image_id].name))
    if image is not None:
        gray = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_32F).var(), image_id
    else:
        return 0, image_id


@dataclasses.dataclass
class ColmapPoint3d(HomogeneousTensor):
    xyz: Annotated[torch.Tensor, arrcheck.ArrayCheck(["N", 3], arrcheck.F32)]
    rgb: Annotated[torch.Tensor, arrcheck.ArrayCheck(["N", 3], arrcheck.U8)]
    error: Annotated[torch.Tensor, arrcheck.ArrayCheck(["N"], arrcheck.F32)]
    indice: Annotated[torch.Tensor, arrcheck.ArrayCheck(["N"], arrcheck.I32)]
    num_image_touch: Annotated[torch.Tensor,
                               arrcheck.ArrayCheck(["N"], arrcheck.I32)]

    @classmethod
    def empty(cls, num: int):
        return cls(xyz=torch.empty([num, 3], dtype=torch.float32),
                   rgb=torch.empty([num, 3], dtype=torch.uint8),
                   error=torch.empty([num], dtype=torch.float32),
                   indice=torch.empty([num], dtype=torch.int32),
                   num_image_touch=torch.empty([num], dtype=torch.int32))


@dataclasses.dataclass
class ColmapSceneSplitChunks:
    chunk_size: float = 100.0
    global_box_pad_factor: float = 0.2
    filter_image_laplacian_var_threshold: float = 1.0
    max_num_camera_per_chunk: int = 1500
    min_num_camera_per_chunk: int = 100
    add_random_far_cam_to_chunk: bool = True
    chunk_box_extend_factor: float = 2.0
    output_path: str | None = None
    debug: bool = False

    def _get_laplacian_vars(self, images_dir, image_metas):
        with ThreadPoolExecutor() as executor:
            res = []
            for res_item in tqdm.tqdm(executor.map(
                    partial(get_var_of_laplacian,
                            images_dir=images_dir,
                            image_metas=image_metas), image_metas),
                                      total=len(image_metas)):
                res.append(res_item)
        return {image_id: var for var, image_id in res}

    def __call__(self, scene: Scene[BasicFrame]) -> list[Scene[BasicFrame]]:
        meta = ColmapMetaHandleBase.get_colmap_meta_from_scene_required(scene)
        model_path = meta.colmap_model_path
        output_path = self.output_path
        if output_path is None:
            assert scene.uri is not None
            output_path = str(Path(scene.uri) / "h3dgs_raw_chunks")
        assert meta.colmap_image_root is not None
        res = cmu.read_model(model_path)
        assert res is not None
        cam_intrinsics, images_metas, points3d = res
        laplacian_vars = self._get_laplacian_vars(meta.colmap_image_root,
                                                  images_metas)

        cam_centers = np.array([
            -cmu.qvec2rotmat(images_metas[key].qvec).astype(np.float32).T
            @ images_metas[key].tvec.astype(np.float32) for key in images_metas
        ])
        random.seed(50051)

        n_pts = get_nb_pts(images_metas)

        # xyzs = np.zeros([n_pts, 3], np.float32)
        # errors = np.zeros([n_pts], np.float32) + 9e9
        # indices = np.zeros([n_pts], np.int64)
        # n_images = np.zeros([n_pts], np.int64)
        # colors = np.zeros([n_pts, 3], np.float32)
        colmap_point3d = ColmapPoint3d.empty(n_pts)
        colmap_point3d.error.fill_(9e9)
        idx = 0
        for key in points3d:
            colmap_point3d.xyz[idx] = torch.from_numpy(points3d[key].xyz)
            colmap_point3d.indice[idx] = points3d[key].id
            colmap_point3d.error[idx] = float(points3d[key].error)
            colmap_point3d.rgb[idx] = torch.from_numpy(points3d[key].rgb)
            colmap_point3d.num_image_touch[idx] = len(points3d[key].image_ids)
            idx += 1
        max_num_points = int(colmap_point3d.indice.max().item()) + 1
        valid_point_mask = (colmap_point3d.error < 1e1).numpy()
        all_mask = np.zeros([max_num_points], np.bool_)
        all_mask[colmap_point3d.indice] = valid_point_mask
        print(max_num_points, colmap_point3d.xyz.shape)
        # all_mask = colmap_point3d.error < 1e1
        colmap_point3d = colmap_point3d[valid_point_mask]
        # xyzsC, colorsC, errorsC, indicesC, n_imagesC = xyzs[mask], colors[mask], errors[mask], indices[mask], n_images[mask]

        points3d_ordered = np.zeros([colmap_point3d.indice.max() + 1, 3])
        points3d_ordered[
            colmap_point3d.indice.numpy()] = colmap_point3d.xyz.numpy()
        images_points3d = {}

        for key in images_metas:
            pts_idx = images_metas[key].point3D_ids
            mask_my = all_mask[images_metas[key].point3D_ids]
            mask = pts_idx >= 0
            mask *= pts_idx < len(points3d_ordered)
            pts_idx = pts_idx[mask]
            if len(pts_idx) > 0:
                image_points3d = points3d_ordered[pts_idx]
                mask = (image_points3d != 0).sum(axis=-1)
                # images_metas[key]["points3d"] = image_points3d[mask>0]
                # print(np.linalg.norm((mask>0).astype(np.float32) - mask_my.astype(np.float32)))
                images_points3d[key] = image_points3d[mask_my]
            else:
                # images_metas[key]["points3d"] = np.array([])
                images_points3d[key] = np.array([])
        global_bbox = np.stack(
            [cam_centers.min(axis=0)[:2],
             cam_centers.max(axis=0)[:2]])
        global_bbox[0, :2] -= self.global_box_pad_factor * self.chunk_size
        global_bbox[1, :2] += self.global_box_pad_factor * self.chunk_size
        extent = global_bbox[1] - global_bbox[0]
        chunk_width = int(np.ceil(extent[0] / self.chunk_size))
        chunk_height = int(np.ceil(extent[1] / self.chunk_size))
        padd = np.array([
            chunk_width * self.chunk_size - extent[0],
            chunk_height * self.chunk_size - extent[1]
        ])

        padd_ref = np.array([
            self.chunk_size - extent[0] % self.chunk_size,
            self.chunk_size - extent[1] % self.chunk_size
        ])
        global_bbox[0, :2] -= padd / 2
        global_bbox[1, :2] += padd / 2

        make_chunk_func = partial(
            self.make_chunk,
            n_width=chunk_width,
            n_height=chunk_height,
            global_bbox=global_bbox,
            colmap_point3d=colmap_point3d,
            images_metas=images_metas,
            images_points3d=images_points3d,
            cam_centers=cam_centers,
            laplacians_dict=laplacian_vars,
            output_path=output_path,
            cam_intrinsics=cam_intrinsics)
        cur_meta = ColmapMetaHandleBase.get_colmap_meta_from_scene_required(scene)
        res_scenes: list[Scene[BasicFrame]] = []
        scene_uri = scene.uri
        assert scene_uri is not None
        for i in range(chunk_width):
            for j in range(chunk_height):
                out_model_path_if_success = make_chunk_func(i, j)
                if out_model_path_if_success is not None:
                    new_meta = dataclasses.replace(cur_meta,
                                                    colmap_model_path=out_model_path_if_success)
                    new_scene = scene.copy()
                    new_scene.uri = str(Path(scene_uri) / f"h3dgs_raw_chunks/{i}_{j}")
                    ColmapMetaHandleBase.set_scene_meta(new_scene, new_meta)
                    res_scenes.append(new_scene)
        return res_scenes
        # breakpoint()
        # global_bbox[0, 2] = -1e12
        # global_bbox[1, 2] = 1e12

    def make_chunk(self, i: int, j: int, n_width: int, n_height: int,
                   global_bbox: np.ndarray, colmap_point3d: ColmapPoint3d,
                   images_metas: dict, images_points3d: dict,
                   cam_centers: np.ndarray, laplacians_dict: dict,
                   output_path: str, cam_intrinsics: Any):
        # in_path = f"{args.base_dir}/chunk_{i}_{j}"
        # if os.path.exists(in_path):
        print(f"chunk {i}_{j}")
        # corner_min, corner_max = bboxes[i, j, :, 0], bboxes[i, j, :, 1]
        corner_min = global_bbox[0] + np.array(
            [i * self.chunk_size, j * self.chunk_size])
        corner_max = global_bbox[0] + np.array([(i + 1) * self.chunk_size,
                                                (j + 1) * self.chunk_size])
        # corner_min[2] = -1e12
        # corner_max[2] = 1e12

        corner_min_for_pts = corner_min.copy()
        corner_max_for_pts = corner_max.copy()
        if i == 0:
            corner_min_for_pts[0] = -1e12
        if j == 0:
            corner_min_for_pts[1] = -1e12
        if i == n_width - 1:
            corner_max_for_pts[0] = 1e12
        if j == n_height - 1:
            corner_max_for_pts[1] = 1e12
        mask_lower = np.all(colmap_point3d.xyz[:, :2].numpy()
                            < corner_max_for_pts,
                            axis=-1)
        mask_upper = np.all(colmap_point3d.xyz[:, :2].numpy()
                            > corner_min_for_pts,
                            axis=-1)
        mask = np.logical_and(mask_lower, mask_upper)
        colmap_point3d = colmap_point3d[torch.from_numpy(mask)]

        valid_cam = np.logical_and(
            np.all(cam_centers[:, :2] < corner_max, axis=-1),
            np.all(cam_centers[:, :2] > corner_min, axis=-1))

        box_center = (corner_max + corner_min) / 2
        extent = (corner_max - corner_min) / 2
        extended_corner_min = box_center - self.chunk_box_extend_factor * extent
        extended_corner_max = box_center + self.chunk_box_extend_factor * extent

        for cam_idx, key in enumerate(images_metas):
            # if not valid_cam[cam_idx]:
            image_points3d = images_points3d[key]
            n_pts = (np.all(image_points3d[:, :2] < corner_max_for_pts, axis=-1) *
                     np.all(image_points3d[:, :2] > corner_min_for_pts, axis=-1)
                     ).sum() if len(image_points3d) > 0 else 0
            # If within chunk
            if np.all(cam_centers[cam_idx, :2] < corner_max) and np.all(
                    cam_centers[cam_idx, :2] > corner_min):
                valid_cam[cam_idx] = n_pts > 50
            # If within 2x of the chunk
            elif np.all(cam_centers[cam_idx, :2] < extended_corner_max) and np.all(
                    cam_centers[cam_idx, :2] > extended_corner_min):
                valid_cam[cam_idx] = n_pts > 50 and random.uniform(0, 1) > 0.5
            # All distances
            if (not valid_cam[cam_idx]
                ) and n_pts > 10 and self.add_random_far_cam_to_chunk:
                valid_cam[cam_idx] = random.uniform(
                    0, 0.5) < (float(n_pts) / len(image_points3d))

        print(
            f"{valid_cam.sum()} valid cameras after visibility-base selection")
        if self.filter_image_laplacian_var_threshold > 0:
            chunk_laplacians = np.array([
                laplacians_dict[key]
                for cam_idx, key in enumerate(images_metas)
                if valid_cam[cam_idx]
            ])
            laplacian_mean = chunk_laplacians.mean()
            laplacian_std_dev = chunk_laplacians.std()
            for cam_idx, key in enumerate(images_metas):
                if valid_cam[cam_idx] and laplacians_dict[key] < (
                        laplacian_mean -
                        self.filter_image_laplacian_var_threshold *
                        laplacian_std_dev):
                    # image = cv2.imread(f"{args.base_dir}/images_masked/{images_metas[key]['name']}")
                    # cv2.imshow("blurry", image)
                    # cv2.waitKey(0)
                    valid_cam[cam_idx] = False

            print(f"{valid_cam.sum()} after Laplacian")

        if valid_cam.sum() > self.max_num_camera_per_chunk:
            for _ in range(valid_cam.sum() - self.max_num_camera_per_chunk):
                remove_idx = random.randint(0, valid_cam.sum() - 1)
                remove_idx_glob = np.arange(
                    len(valid_cam))[valid_cam][remove_idx]
                valid_cam[remove_idx_glob] = False

            print(f"{valid_cam.sum()} after random removal")

        valid_keys = [
            key for idx, key in enumerate(images_metas) if valid_cam[idx]
        ]

        if valid_cam.sum(
        ) > self.min_num_camera_per_chunk:  # or init_valid_cam.sum() > 0:
            out_path = os.path.join(output_path, f"{i}_{j}")
            out_colmap = os.path.join(out_path, "sparse", "0")
            if not self.debug:
                os.makedirs(out_colmap, exist_ok=True)

            # must remove sfm points to use colmap triangulator in following steps
            images_out = {}
            for key in valid_keys:
                image_meta = images_metas[key]
                images_out[key] = cmu.Image(id=key,
                                            qvec=image_meta.qvec,
                                            tvec=image_meta.tvec,
                                            camera_id=image_meta.camera_id,
                                            name=image_meta.name,
                                            xys=[],
                                            point3D_ids=[])

                # if os.path.exists(test_file) and image_meta.name in blending_dict:
                #     n_pts = np.isin(image_meta.point3D_ids, new_indices).sum()
                #     blending_dict[image_meta.name][f"{i}_{j}"] = str(n_pts)
            new_indices = colmap_point3d.indice.numpy()
            new_xyzs = colmap_point3d.xyz.numpy()
            new_colors = colmap_point3d.rgb.numpy()
            new_errors = colmap_point3d.error.numpy()

            points_out = {
                new_indices[idx]:
                cmu.Point3D(id=new_indices[idx],
                            xyz=new_xyzs[idx],
                            rgb=new_colors[idx],
                            error=new_errors[idx],
                            image_ids=np.array([]),
                            point2D_idxs=np.array([]))
                for idx in range(len(new_xyzs))
            }
            if not self.debug:
                cmu.write_model(cam_intrinsics, images_out, points_out,
                                out_colmap)
                chunk_meta_data = {
                    "center": ((corner_min + corner_max) / 2).tolist(),
                    "extent": (corner_max - corner_min).tolist()
                }
                with open(os.path.join(out_path, "meta.json"), 'w') as f:
                    f.write(json.dumps(chunk_meta_data))
                # with open(os.path.join(out_path, "extent.txt"), 'w') as f:
                #     f.write(' '.join(map(str, corner_max - corner_min)))
            else:
                bio = io.BytesIO()
                cmu.write_points3D_binary(points_out, bio)
                data_md5 = hashlib.md5(bio.getvalue()).hexdigest()
                print("Point MD5", data_md5)
                bio = io.BytesIO()
                cmu.write_images_binary(images_out, bio)
                data_md5 = hashlib.md5(bio.getvalue()).hexdigest()
                print("Images MD5", data_md5)
                bio = io.BytesIO()
                cmu.write_cameras_binary(cam_intrinsics, bio)
                data_md5 = hashlib.md5(bio.getvalue()).hexdigest()
                print("Intrinsics MD5", data_md5)

            return out_colmap
        else:
            print(f"Chunk {i},{j} excluded")
            return None


import enum
import math
from typing import Annotated, Self
import torch

from d3sim.core import arrcheck
from d3sim.core.arrcheck.tensor import ArrayCheck
from d3sim.core.thtools import np_to_torch_dev
from d3sim.data.scene_def.base import CameraFieldTypes
from .base import BaseCamera, DistortType, Object2d, Sensor, Resource, ALL_RESOURCE_LOADERS, ResourceLoader
import d3sim.core.dataclass_dispatch as dataclasses 
import numpy as np 
import abc 
import cv2
from d3sim.ops.image.bbox import create_image_bbox_mask
import torchvision
from torchvision.transforms import functional as Fv

@ALL_RESOURCE_LOADERS.register_no_key
class OpencvImageFileLoader(ResourceLoader):
    def read(self, resource: Resource[np.ndarray]) -> np.ndarray:
        return cv2.imread(resource.base_uri)

@ALL_RESOURCE_LOADERS.register_no_key
class PILImageFileLoader(ResourceLoader):
    def read(self, resource: Resource[np.ndarray]) -> np.ndarray:
        from PIL import Image
        return np.array(Image.open(resource.base_uri))

@dataclasses.dataclass
class CropParam:
    x: int
    y: int
    w: int 
    h: int 

@dataclasses.dataclass
class ResizeParam:
    scale_wh: tuple[float, float]
    target_size_wh: tuple[int, int]

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class UndistortParam:
    target_size_wh: tuple[int, int]
    distort: np.ndarray
    cam_matrix: np.ndarray
    new_cam_matrix: np.ndarray
    roi: list[int]
    distort_type: DistortType

@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class BasicCamera(BaseCamera, abc.ABC):
    image_shape_wh: tuple[int, int]
    image_rc: Resource[np.ndarray]
    intrinsic: Annotated[np.ndarray, ArrayCheck([4, 4], arrcheck.Float32 | arrcheck.Float64)] = dataclasses.field(default_factory=lambda: np.eye(4))
    distortion: np.ndarray
    distortion_type: DistortType = DistortType.kOpencvPinhole
    _lazy_image_transform: list[CropParam | ResizeParam | UndistortParam] = dataclasses.field(default_factory=list)

    def get_field_np(self, field: CameraFieldTypes | str) -> np.ndarray | None:
        if field == CameraFieldTypes.IMAGE:
            self.homogeneous_fields[CameraFieldTypes.IMAGE] = self.image_rc.data
            return self.image_rc.data
        return super().get_field_np(field)

    def get_objects_by_source(self, source: str = "") -> list[Object2d]:
        res: list[Object2d] = []
        if source == "":
            return self.objects
        for obj in self.objects:
            if obj.source == source:
                res.append(obj)
        return res

    def _crop_objects(self, objects: list[Object2d], crop_xy: tuple[int, int], crop_wh: tuple[int, int]) -> list[Object2d]:
        objects = self.get_objects_by_source()
        all_bbox_xywh = self.get_bbox_xywh()
        new_bbox_xy = np.maximum(all_bbox_xywh[:, :2], np.array(crop_xy))
        new_xy_max = np.minimum(all_bbox_xywh[:, :2] + all_bbox_xywh[:, 2:], np.array(crop_xy) + np.array(crop_wh))
        new_bbox_wh = new_xy_max - new_bbox_xy
        new_bbox_xywh = np.concatenate([new_bbox_xy - np.array(crop_xy), new_bbox_wh], axis=1)
        mask = np.all(new_bbox_wh > 0, axis=1)
        new_objs: list[Object2d] = []
        for i, obj in enumerate(objects):
            valid = mask[i]
            if valid:
                new_objs.append(dataclasses.replace(obj, bbox_xywh=new_bbox_xywh[i]))
        return new_objs
    
    def crop(self, crop_xy: tuple[int, int], crop_wh: tuple[int, int]) -> Self:
        x, y = crop_xy
        w_valid = min(crop_wh[0], self.image_shape_wh[0] - x)
        h_valid = min(crop_wh[1], self.image_shape_wh[1] - y)
        crop_wh = (w_valid, h_valid)
        w, h = crop_wh
        assert x >= 0 and y >= 0 and x + w <= self.image_shape_wh[0] and y + h <= self.image_shape_wh[1]
        objects = self.get_objects_by_source()
        new_objs: list[Object2d] = self._crop_objects(objects, crop_xy, crop_wh)
        return dataclasses.replace(self, image_shape_wh=(w, h),
            intrinsic=self.get_cropped_intrinsic(crop_xy),
            objects=new_objs,
            _lazy_image_transform=self._lazy_image_transform + [CropParam(x, y, w, h)])

    def resize(self, scale: float) -> Self:
        target_size_wh = (int(round(self.image_shape_wh[0] * scale)), int(round(self.image_shape_wh[1] * scale)))
        real_scale_wh = (target_size_wh[0] / self.image_shape_wh[0], target_size_wh[1] / self.image_shape_wh[1])
        
        objects = self.get_objects_by_source()
        new_objs: list[Object2d] = []
        for i, obj in enumerate(objects):            
            new_objs.append(dataclasses.replace(obj, bbox_xywh=obj.bbox_xywh * scale))
        return dataclasses.replace(self, image_shape_wh=target_size_wh, 
            intrinsic=self.get_resized_intrinsic(scale),
             objects=new_objs,
             _lazy_image_transform=self._lazy_image_transform + [ResizeParam(real_scale_wh, target_size_wh)])

    def undistort(self, alpha: float = 1.0) -> Self:
        new_cam_matrix, roi = cv2.getOptimalNewCameraMatrix(self.intrinsic[:3, :3], self.distortion, self.image_shape_wh, alpha)
        crop_xy = (roi[0], roi[1])
        crop_wh = (roi[2], roi[3])
        objects = self.get_objects_by_source()
        new_objs: list[Object2d] = self._crop_objects(objects, crop_xy, crop_wh)

        new_intrinsic = np.eye(4, dtype=np.float32)
        new_intrinsic[:3, :3] = new_cam_matrix
        new_intrinsic[0, 2] -= crop_xy[0]
        new_intrinsic[1, 2] -= crop_xy[1]

        undistort_param = UndistortParam(crop_wh, self.distortion, self.intrinsic[:3, :3], new_cam_matrix, list(roi), self.distortion_type)
        distort = self.distortion.copy()
        distort[:] = 0
        # print(roi, new_cam_matrix)
        return dataclasses.replace(self, objects=new_objs, distortion=distort, distortion_type=DistortType.kNone, image_shape_wh=crop_wh, intrinsic=new_intrinsic,  _lazy_image_transform=self._lazy_image_transform + [undistort_param])

    def get_cropped_intrinsic(self, crop_xy: tuple[int, int]) -> np.ndarray:
        x, y = crop_xy
        intrinsic = self.intrinsic.copy()
        intrinsic[0, 2] -= x
        intrinsic[1, 2] -= y
        return intrinsic

    def get_resized_intrinsic(self, scale: float) -> np.ndarray:
        intrinsic = self.intrinsic.copy()
        intrinsic[0, 0] *= scale
        intrinsic[1, 1] *= scale
        intrinsic[0, 2] *= scale
        intrinsic[1, 2] *= scale
        return intrinsic

    @property 
    def focal_length(self) -> np.ndarray:
        return np.array([self.intrinsic[0, 0], self.intrinsic[1, 1]], np.float32)

    @property 
    def principal_point(self) -> np.ndarray:
        return np.array([self.intrinsic[0, 2], self.intrinsic[1, 2]], np.float32)

    @property 
    def principal_point_unified(self) -> np.ndarray:
        return self.principal_point / np.array(self.image_shape_wh, np.float32)

    @property 
    def image(self) -> np.ndarray:
        """Image RGB Uint8 array."""
        return self._apply_image_transform(self.get_field_np_required(CameraFieldTypes.IMAGE))

    def _apply_image_transform(self, img: np.ndarray, img_is_seg: bool = False) -> np.ndarray:
        if self._lazy_image_transform:
            for transform in self._lazy_image_transform:
                if isinstance(transform, CropParam):
                    x, y, w, h = transform.x, transform.y, transform.w, transform.h
                    img = img[y:y+h, x:x+w]
                elif isinstance(transform, ResizeParam):
                    mode = cv2.INTER_NEAREST if img_is_seg else cv2.INTER_LINEAR
                    img = cv2.resize(img, transform.target_size_wh, interpolation=mode)
                elif isinstance(transform, UndistortParam):
                    assert transform.distort_type == DistortType.kOpencvPinhole or transform.distort_type == DistortType.kOpencvPinholeWaymo
                    img = cv2.undistort(img, transform.cam_matrix, transform.distort, newCameraMatrix=transform.new_cam_matrix)
                    img = img[transform.roi[1]:transform.roi[1] + transform.roi[3], transform.roi[0]:transform.roi[0] + transform.roi[2]]
                    assert img.shape[1] == transform.target_size_wh[0]
                    assert img.shape[0] == transform.target_size_wh[1]
            return img
        return img

    def _apply_image_transform_torch(self, img_th: torch.Tensor, output_is_chw: bool = False, img_is_seg: bool = False) -> torch.Tensor:
        if self._lazy_image_transform:
            ndim = img_th.ndim
            if ndim > 2:
                img_th_chw = img_th.permute(2, 0, 1)
            else:
                img_th_chw = img_th.unsqueeze(0)
            # use torchvision to apply transforms
            for transform in self._lazy_image_transform:
                if isinstance(transform, CropParam):
                    x, y, w, h = transform.x, transform.y, transform.w, transform.h
                    img_th_chw = Fv.crop(img_th_chw, y, x, h, w)
                elif isinstance(transform, ResizeParam):
                    assert img_th.dtype in [torch.float32, torch.uint8]
                    if img_is_seg:
                        mode = Fv.InterpolationMode.NEAREST
                    else:
                        mode = Fv.InterpolationMode.BILINEAR
                    target_size_wh = transform.target_size_wh
                    img_th_chw = Fv.resize(img_th_chw, [target_size_wh[1], target_size_wh[0]], interpolation=mode)
                else:
                    raise NotImplementedError
            if output_is_chw:
                return img_th_chw
            else:
                if ndim == 2:
                    return img_th_chw[0]
                return img_th_chw.permute(1, 2, 0)
        return img_th

    def get_image_torch(self, output_is_chw: bool = False, device: torch.device | None = None) -> torch.Tensor:
        if self.has_field_torch(CameraFieldTypes.IMAGE):
            res = self.get_field_torch(CameraFieldTypes.IMAGE)
            assert res is not None 
            return self._apply_image_transform_torch(res, output_is_chw)
        return self._apply_image_transform_torch(np_to_torch_dev(self.image, device), output_is_chw)

    @property 
    def segmentation(self) -> np.ndarray | None:
        data = self.get_field_np(CameraFieldTypes.SEGMENTATION)
        if data is not None:
            return self._apply_image_transform(data)
        return None

    @property 
    def mask(self) -> np.ndarray | None:
        data = self.get_field_np(CameraFieldTypes.VALID_MASK)
        if data is not None:
            return self._apply_image_transform(data)
        return None

    def get_bbox_xywh(self, source: str = ""):
        objects = self.get_objects_by_source(source)
        bbox_xywh = np.empty((len(objects), 4), dtype=np.float32)
        for i, obj in enumerate(objects):
            bbox_xywh[i] = obj.bbox_xywh
        return bbox_xywh

    def get_bbox_mask(self, bbox_xywh: np.ndarray | None = None, bbox_pad: np.ndarray | None = None, bbox_source: str = "") -> torch.Tensor:
        if bbox_xywh is None:
            bbox_xywh = self.get_bbox_xywh(bbox_source)
        bbox_xywh_th = np_to_torch_dev(bbox_xywh)
        return create_image_bbox_mask(bbox_xywh_th, self.image_shape_wh, bbox_pad)

    def get_bbox_mask_np(self, bbox_xywh: np.ndarray | None = None, bbox_pad: np.ndarray | None = None, bbox_source: str = "") -> np.ndarray:
        return self.get_bbox_mask(bbox_xywh, bbox_pad, bbox_source).cpu().numpy()


@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class BasicPinholeCamera(BasicCamera):
    @property 
    def fov_xy(self) -> np.ndarray:
        focal_length = self.focal_length
        return 2 * np.arctan(0.5 * np.array(self.image_shape_wh, np.float32) / focal_length)



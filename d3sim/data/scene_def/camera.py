from typing import Annotated
import torch

from d3sim.core import arrcheck
from d3sim.core.arrcheck.tensor import ArrayCheck
from d3sim.core.thtools import np_to_torch_dev
from .base import Object2d, Sensor, Resource, ALL_RESOURCE_LOADERS, ResourceLoader
import d3sim.core.dataclass_dispatch as dataclasses 
import numpy as np 
import abc 
import cv2
from d3sim.ops.image.bbox import create_image_bbox_mask

@ALL_RESOURCE_LOADERS.register_no_key
class OpencvImageFileLoader(ResourceLoader):
    def read(self, resource: Resource[np.ndarray]) -> np.ndarray:
        return cv2.imread(resource.base_uri)

@ALL_RESOURCE_LOADERS.register_no_key
class PILImageFileLoader(ResourceLoader):
    def read(self, resource: Resource[np.ndarray]) -> np.ndarray:
        from PIL import Image
        return np.array(Image.open(resource.base_uri))

@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class BasicCamera(Sensor, abc.ABC):
    image_shape_wh: tuple[int, int]
    image_rc: Resource[np.ndarray]
    intrinsic: Annotated[np.ndarray, ArrayCheck([4, 4], arrcheck.Float32 | arrcheck.Float64)] = dataclasses.field(default_factory=lambda: np.eye(4))
    distortion: np.ndarray
    axes_front_u_v: tuple[int, int, int] = (2, 0, 1)
    objects: list[Object2d] = dataclasses.field(default_factory=list)

    def get_objects_by_source(self, source: str = "") -> list[Object2d]:
        res: list[Object2d] = []
        for obj in self.objects:
            if obj.source == source:
                res.append(obj)
        return res

    @property 
    def focal_length(self) -> np.ndarray:
        return np.array([self.intrinsic[0, 0], self.intrinsic[1, 1]], np.float32)

    @property 
    def principal_point(self) -> np.ndarray:
        return np.array([self.intrinsic[0, 2], self.intrinsic[1, 2]], np.float32)

    @property 
    def principal_point_unified(self) -> np.ndarray:
        return np.array([self.intrinsic[0, 2], self.intrinsic[1, 2]], np.float32) / np.array(self.image_shape_wh, np.float32)

    @property 
    def image(self) -> np.ndarray:
        """Image RGB Uint8 array."""
        return self.image_rc.data
    
    @property 
    def segmentation(self) -> np.ndarray | None:
        return None

    @property 
    def mask(self) -> np.ndarray | None:
        return None

    def get_bbox_xywh(self, source: str = ""):
        objects = self.get_objects_by_source(source)
        bbox_xywh = np.empty((len(objects), 4), dtype=np.float32)
        for i, obj in enumerate(objects):
            bbox_xywh[i] = obj.bbox_xywh
        return bbox_xywh

    def get_bbox_mask(self, bbox_pad: np.ndarray | None = None, bbox_source: str = "") -> torch.Tensor:
        bbox_xywh = self.get_bbox_xywh(bbox_source)
        bbox_xywh_th = np_to_torch_dev(bbox_xywh)
        return create_image_bbox_mask(bbox_xywh_th, self.image_shape_wh, bbox_pad)

    def get_bbox_mask_np(self, bbox_pad: np.ndarray | None = None, bbox_source: str = "") -> np.ndarray:
        return self.get_bbox_mask(bbox_pad, bbox_source).cpu().numpy()


@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class BasicPinholeCamera(BasicCamera):
    @property 
    def fov_xy(self) -> np.ndarray:
        focal_length = self.focal_length
        return 2 * np.arctan(0.5 * np.array(self.image_shape_wh, np.float32) / focal_length)



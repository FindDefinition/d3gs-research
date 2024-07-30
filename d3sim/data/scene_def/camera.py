from .base import Sensor, Resource, ALL_RESOURCE_LOADERS, ResourceLoader
import d3sim.core.dataclass_dispatch as dataclasses 
import numpy as np 
import abc 
import cv2

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
    intrinsic: np.ndarray = dataclasses.field(default_factory=lambda: np.eye(4))
    
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

@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class BasicPinholeCamera(BasicCamera):
    distortion: np.ndarray
import abc
from .base import BaseLidar, Sensor, Resource, ALL_RESOURCE_LOADERS, ResourceLoader, LidarFieldTypes
import d3sim.core.dataclass_dispatch as dataclasses 
import numpy as np 

@dataclasses.dataclass(kw_only=True)
class BasicLidar(BaseLidar, abc.ABC):

    @property 
    def xyz(self) -> np.ndarray:
        """Per-point xyz."""
        return self.get_field_np_required(LidarFieldTypes.POINT_CLOUD) 

    @property 
    def segmentation(self) -> np.ndarray | None:
        """Per-point segmentation."""
        return self.get_field_np(LidarFieldTypes.SEGMENTATION) 

    @property 
    def timestamp_per_point(self) -> np.ndarray | None:
        """Per-point timestamp."""
        return self.get_field_np(LidarFieldTypes.TIMESTAMP) 

    @property 
    def intensity(self) -> np.ndarray | None:
        """Per-point intensity."""
        return self.get_field_np(LidarFieldTypes.INTENSITY) 

    @property 
    def mask(self) -> np.ndarray | None:
        """Per-point mask, used to keep raw point shape."""
        return self.get_field_np(LidarFieldTypes.VALID_MASK) 

    def xyz_masked(self) -> np.ndarray:
        """Per-point xyz."""
        if self.mask is None:
            return self.xyz
        # print(self.xyz, self.xyz[self.mask])
        # print(self.xyz, self.xyz[self.mask])
        # print(self.xyz, self.xyz[self.mask])

        return self.xyz[self.mask]

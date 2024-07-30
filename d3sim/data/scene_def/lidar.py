import abc
from .base import Sensor, Resource, ALL_RESOURCE_LOADERS, ResourceLoader
import d3sim.core.dataclass_dispatch as dataclasses 
import numpy as np 

@dataclasses.dataclass(kw_only=True)
class BasicLidar(Sensor, abc.ABC):

    @property 
    @abc.abstractmethod
    def xyz(self) -> np.ndarray:
        """Per-point xyz."""
        raise NotImplementedError

    @property 
    def segmentation(self) -> np.ndarray | None:
        """Per-point segmentation."""
        return None 

    @property 
    def timestamp_per_point(self) -> np.ndarray | None:
        """Per-point timestamp."""
        return None 

    @property 
    def intensity(self) -> np.ndarray | None:
        """Per-point intensity."""
        return None 

    @property 
    def mask(self) -> np.ndarray | None:
        """Per-point mask, used to keep raw point shape."""
        return None 

    def xyz_masked(self) -> np.ndarray:
        """Per-point xyz."""
        if self.mask is None:
            return self.xyz
        # print(self.xyz, self.xyz[self.mask])
        # print(self.xyz, self.xyz[self.mask])
        # print(self.xyz, self.xyz[self.mask])

        return self.xyz[self.mask]

import abc
from typing import Literal 
from d3sim.core import dataclass_dispatch as dataclasses
import numpy as np

from d3sim.data.scene_def.camera import BasicPinholeCamera 

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class DatasetPointCloud:
    xyz: np.ndarray
    rgb: np.ndarray | None = None 
    # assume relative value
    timestamp: np.ndarray | None = None
    # assume uint8 (0-255)
    intensity: np.ndarray | None = None
    # d3sim standard segmentation (unknown, ground, vehicle, pedestrian, etc)
    segmentation: np.ndarray | None = None

class D3simDataset(abc.ABC):

    @property 
    @abc.abstractmethod
    def dataset_point_cloud(self) -> DatasetPointCloud:
        raise NotImplementedError

    @property 
    @abc.abstractmethod
    def extent(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def get_cameras(self, split: Literal["train", "test"]) -> list[BasicPinholeCamera]:
        raise NotImplementedError


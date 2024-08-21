import abc
from typing import Generic, Self

from d3sim.algos.d3gs.base import GaussianModelBase, T
import torch.amp.grad_scaler

from d3sim.algos.d3gs.data_base import D3simDataset
from d3sim.core import dataclass_dispatch as dataclasses
from torch.optim.optimizer import Optimizer

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class OptimizerObject:
    optimizer: Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler | None


class GaussianOptimizerBase(Generic[T], abc.ABC):
    
    @abc.abstractmethod
    def init_optimizer(self, model: T, dataset: D3simDataset, batch_size: int, total_step: int): ...

    @abc.abstractmethod
    def get_optimizer_obj_dict(self) -> dict[str, OptimizerObject]: ...

    def get_optimizer_dict(self) -> dict[str, Optimizer]:
        optims = self.get_optimizer_obj_dict()
        return {k: v.optimizer for k, v in optims.items()}

    def zero_grad(self, set_to_none: bool = False):
        optims = self.get_optimizer_obj_dict()
        for optim_obj in optims.values():
            optim_obj.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, scaler: torch.amp.grad_scaler.GradScaler | None = None):
        optims = self.get_optimizer_obj_dict()
        for optim_obj in optims.values():
            if scaler is not None:
                scaler.step(optim_obj.optimizer)
            else:
                optim_obj.optimizer.step()
            if optim_obj.scheduler is not None:
                optim_obj.scheduler.step()

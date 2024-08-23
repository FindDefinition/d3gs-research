
import abc
from d3sim.core import dataclass_dispatch as dataclasses
from d3sim.core.arrcheck.dcbase import DataClassWithArrayCheck
from typing import Annotated, Callable, Literal, Mapping
from d3sim.core.pytorch.hmt import HomogeneousTensor
from d3sim.algos.d3gs.base import GaussianModelBase
import d3sim.core.arrcheck as ac
import torch
from d3sim.csrc.inliner import INLINER
import pccm 
from d3sim.algos.d3gs.render import GaussianSplatOutput 
from d3sim.algos.d3gs import config_def
from d3sim.core.debugtools import get_from_store
from torch.optim.optimizer import Optimizer

@torch.no_grad()
def _update_optimizer_only(
    param_fn: Callable[[str, torch.Tensor], torch.nn.Parameter],
    optimizer_fn: Callable[[str, torch.Tensor], torch.Tensor],
    optimizers: Mapping[str, Optimizer] | Optimizer,
    names: list[str],
):
    """Update the parameters and the state in the optimizers with defined functions.

    Args:
        param_fn: A function that takes the name of the parameter and the parameter itself,
            and returns the new parameter.
        optimizer_fn: A function that takes the key of the optimizer state and the state value,
            and returns the new state value.
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        names: A list of key names to update. If None, update all. Default: None.
    """
    if isinstance(optimizers, Optimizer):
        for i, param_group in enumerate(optimizers.param_groups):
            p = param_group["params"][0]
            name = param_group["name"]
            if name not in names:
                continue 
            p_state = optimizers.state[p]
            del optimizers.state[p]
            for key in p_state.keys():
                if key != "step":
                    v = p_state[key]
                    if v is not None:
                        p_state[key] = optimizer_fn(key, v)
            p_new = param_fn(name, p)
            optimizers.param_groups[i]["params"] = [p_new]
            optimizers.state[p_new] = p_state
    else:
        for name in names:
            optimizer = optimizers[name]
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        v = p_state[key]
                        if v is not None:
                            p_state[key] = optimizer_fn(key, v)
                p_new = param_fn(name, p)
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state

def compare_optimizer_params(
    optimizers: Mapping[str, Optimizer],
    ref_optimizers: Mapping[str, Optimizer],
):

    for name in optimizers.keys():
        optimizer = optimizers[name]
        ref_optimizer = ref_optimizers[name]
        for i, param_group in enumerate(optimizer.param_groups):
            p = param_group["params"][0]
            ref_p = ref_optimizer.param_groups[i]["params"][0]
            p_state = optimizer.state[p]
            ref_p_state = ref_optimizer.state[ref_p]
            for key in p_state.keys():
                if key != "step":
                    v = p_state[key]
                    ref_v = ref_p_state[key]
                    print(f'{param_group["name"]} State-{key}', torch.linalg.norm(v - ref_v))
            print(param_group["name"], torch.linalg.norm(p - ref_p))

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class GaussianTrainState(HomogeneousTensor):
    # duv_ndc = duv * (0.5 * shape_wh)
    # duv_ndc_length = torch.norm(duv_ndc, dim=1)
    # i.e. l2 norm of duv_ndc
    duv_ndc_length: Annotated[torch.Tensor, ac.ArrayCheck(["N"], ac.F32)]
    count: Annotated[torch.Tensor, ac.ArrayCheck(["N"], ac.F32)]
    max_radii: Annotated[torch.Tensor, ac.ArrayCheck(["N"], ac.F32)]
    scene_scale: float = 1.0

    @classmethod 
    def create(cls, num_gaussian: int, scene_scale: float = 1.0, device=None):
        return cls(
            duv_ndc_length=torch.zeros(num_gaussian, dtype=torch.float32, device=device),
            count=torch.zeros(num_gaussian, dtype=torch.float32, device=device),
            max_radii=torch.zeros(num_gaussian, dtype=torch.float32, device=device),
            scene_scale=scene_scale,
        )

    def reset(self):
        self.duv_ndc_length.zero_()
        self.count.zero_()
        self.max_radii.zero_()


class GaussianStrategyBase(abc.ABC):
    @abc.abstractmethod
    def update(self, state: GaussianTrainState, out: GaussianSplatOutput, duv: torch.Tensor): ...

    @abc.abstractmethod
    def refine_gs(
        self,
        model: GaussianModelBase,
        optimizers: Mapping[str, Optimizer],
        state: GaussianTrainState,
        step: int,
    ): ...

    @abc.abstractmethod
    def rescale_steps_by_batch_size(self, batch_size: int):
        raise NotImplementedError


@torch.no_grad()
def select_gs_optimizer_by_mask(
    model: GaussianModelBase,
    optimizers: Mapping[str, Optimizer],
    inds: torch.Tensor,
):
    """Inplace remove the Gaussian with the given mask.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to remove the Gaussians.
    """

    def param_fn(name: str, p: torch.Tensor) -> torch.nn.Parameter:
        res = getattr(model, name)
        assert isinstance(res, torch.nn.Parameter)
        return res

    def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
        return v[inds]
    # update the parameters and the state in the optimizers
    _update_optimizer_only(param_fn, optimizer_fn, optimizers, list(model.get_all_tensor_fields().keys()))


@torch.no_grad()
def zero_optimizer_by_range(
    model: GaussianModelBase,
    optimizers: Mapping[str, Optimizer],
    interval: tuple[int, int],
):
    """Inplace remove the Gaussian with the given mask.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to remove the Gaussians.
    """

    def param_fn(name: str, p: torch.Tensor) -> torch.nn.Parameter:
        return getattr(model, name)

    def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
        v[interval[0]:interval[1]].zero_()
        return v
    # update the parameters and the state in the optimizers
    _update_optimizer_only(param_fn, optimizer_fn, optimizers, list(model.get_all_tensor_fields().keys()))


@torch.no_grad()
def reset_param_in_optim(
    model: GaussianModelBase,
    optimizers: Mapping[str, Optimizer],
    name: str,
):
    """Inplace remove the Gaussian with the given mask.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to remove the Gaussians.
    """

    def param_fn(name: str, p: torch.Tensor) -> torch.nn.Parameter:
        return getattr(model, name)

    def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(v)
    # update the parameters and the state in the optimizers
    _update_optimizer_only(param_fn, optimizer_fn, optimizers, [name])

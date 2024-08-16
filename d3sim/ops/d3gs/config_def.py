from typing import Literal
import numpy as np
import torch 
from d3sim.constants import IsAppleSiliconMacOs
from d3sim.core import dataclass_dispatch as dataclasses

@dataclasses.dataclass
class Strategy:
    reset_opacity_thresh: float = 0.01
    prune_opacity_thresh: float = 0.005
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    reset_opacity_every: int = 3000
    refine_every: int = 100
    absgrad: bool = False
    revised_opacity: bool = False
    verbose: bool = False

    def rescale_steps_by_batch_size(self, batch_size: int):
        self.refine_start_iter = self.refine_start_iter // batch_size
        self.refine_stop_iter = self.refine_stop_iter // batch_size
        self.refine_every = self.refine_every // batch_size
        self.reset_opacity_every = self.reset_opacity_every // batch_size

@dataclasses.dataclass
class Optimizer:
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001

    position_lr_max_steps: int = 30_000
    def rescale_steps_by_batch_size(self, batch_size: int):
        self.position_lr_max_steps = self.position_lr_max_steps // batch_size

@dataclasses.dataclass
class Train:
    optim: Optimizer = dataclasses.field(default_factory=Optimizer)
    strategy: Strategy = dataclasses.field(default_factory=Strategy)
    iterations: int = 30_000
    lambda_dssim: float = 0.2
    batch_size: int = 1

    def rescale_steps_by_batch_size(self, batch_size: int):
        self.iterations = self.iterations // batch_size
        self.optim.rescale_steps_by_batch_size(batch_size)
        self.strategy.rescale_steps_by_batch_size(batch_size)

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class GaussianSplatConfig:
    tile_size: tuple[int, int] = (16, 16)
    eps: float = 1e-6
    cov2d_radii_eigen_eps: float = 0.1
    projected_clamp_factor: float = 1.3
    gaussian_std_sigma: float = 3.0
    gaussian_lowpass_filter: float = 0.3
    alpha_eps: float = 1.0 / 255.0
    transmittance_eps: float = 0.0001
    depth_32bit_prec: float = 0.001

    enable_32bit_sort: bool = False
    render_depth: bool = False
    bg_color: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros((3), np.float32))
    scale_global: float = 1.0
    use_nchw: bool = True
    render_rgba: bool = False

    transmittance_is_double = False
    backward_reduction: Literal["none", "warp", "block"] = "warp"
    verbose: bool = False

    recalc_cov3d_in_bwd: bool = True
    enable_device_asserts: bool = False

    use_cub_sort: bool = False if IsAppleSiliconMacOs else True

    use_int64_tile_touched: bool = False

    _bg_color_device: torch.Tensor | None = None

    @property 
    def num_channels(self):
        return 4 if self.render_rgba else 3

    @property
    def block_size(self):
        return self.tile_size[0] * self.tile_size[1]

    def get_bg_color_device(self, device: torch.device):
        if self._bg_color_device is None:
            self._bg_color_device = torch.from_numpy(self.bg_color).to(device)
        return self._bg_color_device


@dataclasses.dataclass
class Model:
    op: GaussianSplatConfig

@dataclasses.dataclass
class Config:
    train: Train
    model: Model


def get_lite_gsplat_cfg_for_apple():
    cfg = GaussianSplatConfig()
    cfg.enable_32bit_sort = True 
    cfg.gaussian_std_sigma = 2.0
    return cfg 
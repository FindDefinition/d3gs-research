import enum
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

class EarlyFilterAlgo(enum.IntEnum):
    NONE = 0
    # use aabb overlap. fast enough in most cases.
    AABB = 1
    # Ellipse: reference filter, use ellipse-aabb overlap, very slow, only for evaluation.
    ELLIPSE = 2
    # OBB: use OBB-aabb overlap. number of filterd close to ellipse overlap, 
    # slightly slower than aabb solution
    OBB = 3
    # dual fast voxel traversal to find obb-grid overlap in O(1) space and O(M + N) time, 
    # best for large scene 
    OBB_DFVT = 4

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class GaussianSplatConfig:
    tile_size: tuple[int, int] = (16, 16)
    warp_size: tuple[int, int] = (8, 4)
    eps: float = 1e-6
    cov2d_radii_eigen_eps: float = 0.1
    projected_clamp_factor: float = 1.3
    gaussian_std_sigma: float = 3.0
    gaussian_lowpass_filter: float = 0.3
    alpha_eps: float = 1.0 / 255.0
    transmittance_eps: float = 0.0001
    # TODO use far instead of prec for depth sort
    depth_32bit_prec: float = 0.001

    enable_32bit_sort: bool = False
    render_depth: bool = False
    render_inv_depth: bool = False
    enable_anti_aliasing: bool = False
    bg_color: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros((3), np.float32))
    scale_global: float = 1.0
    # TODO nchw is deprecated. remove it in the future.
    use_nchw: bool = False
    render_rgba: bool = False

    transmittance_is_double = False
    backward_reduction: Literal["none", "warp", "block"] = "warp"
    verbose: bool = False

    recalc_cov3d_in_bwd: bool = True
    enable_device_asserts: bool = False

    use_cub_sort: bool = False if IsAppleSiliconMacOs else True
    # TODO use_int64_tile_touched should be True by default,
    # False should only used for performance benchmark.
    use_int64_tile_touched: bool = False

    use_proxy_model = True

    # if set to True, you must use custom features instead of rgb
    # the out.color will only have 1 (if render alpha) or 0 channels
    # should be used when you want to use complex color model
    # such as hash grid or embedding.

    # builtin rgb only support simple color model such as sh.
    # you can use proxy to change the simple color model.
    disable_builtin_rgb = False

    move_opacity_in_grad_to_prep: bool = False

    measure_atomic_add_count: bool = False

    # lossless gaussian filter in prep by more precise method
    # this will use gaussian 2d sample formula and 
    # alpha_eps (1 / 255) in rasterize kernel to construct a 
    # precise ellipse instead of original coarse 3-sigma ellipse in prep,
    # then use bounding box (aabb or obb) of that ellipse to do filter.

    # this can increase performance of sort and rasterize.
    # forward 30%, backward 5% in garden scene.
    # up to 500% performance increase in large scene.
    # TODO completely remove original 3-sigma ellipse in the future.
    early_filter_algo: EarlyFilterAlgo = EarlyFilterAlgo.OBB_DFVT

    @property 
    def num_channels(self):
        if self.disable_builtin_rgb:
            return 1 if self.render_rgba else 0
        return 4 if self.render_rgba else 3

    @property
    def block_size(self):
        return self.tile_size[0] * self.tile_size[1]


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
import random

import tqdm
from d3sim.constants import D3SIM_DEFAULT_DEVICE
from d3sim.core.train import BasicTrainEngine, StepEvent, TrainEvent, TrainEventType
from d3sim.ops.d3gs.base import GaussianModelBase, GaussianModelOriginFused
from d3sim.ops.d3gs import config_def
from d3sim.core import dataclass_dispatch as dataclasses

from d3sim.ops.d3gs.data.load import load_scene_info_and_first_cam, Scene, original_cam_to_d3sim_cam
from d3sim.ops.d3gs.render import GaussianSplatOutput, rasterize_gaussians
from d3sim.ops.d3gs.strategy import GaussianTrainState, GaussianStrategyBase
from d3sim.ops.d3gs.losses import l1_loss, ssim

from d3sim.ops.d3gs.tools import create_origin_3dgs_optimizer, init_original_3dgs_model
import contextlib 
import torch 
@dataclasses.dataclass
class StepCtx:
    output: GaussianSplatOutput | None = None

class Trainer:
    def __init__(self, model: GaussianModelBase, cfg: config_def.Config, scene_extent: float) -> None:
        self.engine = BasicTrainEngine()
        self.model = model
        optim, scheduler = create_origin_3dgs_optimizer(model, cfg.train.optim, scene_extent)
        self.optim = optim 
        self.scheduler = scheduler
        self.cfg = cfg
        self.train_state = GaussianTrainState.create(model.xyz.shape[0])
        self.strategy = GaussianStrategyBase(cfg.train.strategy)
        self.engine.register_period_event("log", 1000, self._on_log)
        self.engine.register_period_event("up_sh", 1000, self._on_up_sh_degree_log)
        self.engine.register_base_event(TrainEventType.BeginIteration, self._on_iter_start)

    def train(self, scene: Scene):
        viewpoint_stack = None

        for ev in tqdm.tqdm(self.engine.train_step_generator(0, range(self.cfg.train.iterations), total_step=self.cfg.train.iterations, step_ctx_creator=self.train_step_ctx)):
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack)-1))
            
            uv_grad_holder = torch.empty([self.model.xyz.shape[0], 2], dtype=torch.float32, device=self.model.xyz.device)
            uv_grad_holder.requires_grad = True
            cam = original_cam_to_d3sim_cam(viewpoint_cam)
            out = rasterize_gaussians(self.model, [cam], gaussian_cfg=self.cfg.model.op, training=True, uv_grad_holder=uv_grad_holder)
            assert ev.step_ctx_value is not None 
            ev.step_ctx_value.output = out
            gt_image = viewpoint_cam.original_image.to(D3SIM_DEFAULT_DEVICE)
            Ll1 = l1_loss(out.color, gt_image)
            loss = (1.0 - self.cfg.train.lambda_dssim) * Ll1 + self.cfg.train.lambda_dssim * (1.0 - ssim(out.color, gt_image))
            loss.backward()
            self.optim.step()
            self.optim.zero_grad(set_to_none = True)



    def _on_iter_start(self, ev: TrainEvent):
        for param_group in self.optim.param_groups:
            if param_group["name"] == "xyz":
                lr = self.scheduler(ev.cur_step)
                param_group['lr'] = lr
                return lr


    def _on_log(self, ev: TrainEvent):
        print(ev.cur_step)
        pass 

    def _on_up_sh_degree_log(self, ev: TrainEvent):
        if self.model.cur_sh_degree < 3:
            self.model.cur_sh_degree += 1
            print(f"Up SH degree to {self.model.cur_sh_degree}")


    @contextlib.contextmanager
    def train_step_ctx(self):
        yield StepCtx()


def __main():
    scene_info, first_cam = load_scene_info_and_first_cam("/Users/yanyan/Downloads/360_v2/garden")
    points = scene_info.point_cloud.points
    cfg = config_def.Config(
        model=config_def.Model(config_def.GaussianSplatConfig()),
        train=config_def.Train()
    )
    model = GaussianModelOriginFused.empty_parameter(points.shape[0], 3, True, device=torch.device(D3SIM_DEFAULT_DEVICE)) 
    init_original_3dgs_model(model, points, scene_info.point_cloud.colors)
    trainer = Trainer(model, cfg, scene_info.nerf_normalization["radius"])
    scene = Scene(scene_info, True)

    trainer.train(scene)

if __name__ == "__main__":
    __main() 
import random

import tqdm
from d3sim.constants import D3SIM_DEFAULT_DEVICE, PACKAGE_ROOT, IsAppleSiliconMacOs
from d3sim.core.train import BasicTrainEngine, StepEvent, TrainEvent, TrainEventType
from d3sim.csrc.inliner import INLINER
from d3sim.data.scene_def.camera import BasicPinholeCamera
from d3sim.algos.d3gs.base import GaussianModelBase, GaussianModelOrigin, GaussianModelOriginFused
from d3sim.algos.d3gs import config_def
from d3sim.core import dataclass_dispatch as dataclasses

from d3sim.algos.d3gs.data.load import load_scene_info_and_first_cam, Scene, original_cam_to_d3sim_cam
from d3sim.algos.d3gs.render import CameraBundle, GaussianSplatOp, GaussianSplatOutput, rasterize_gaussians, rasterize_gaussians_dynamic
from d3sim.algos.d3gs.strategy import GaussianTrainState, GaussianStrategyBase, reset_param_in_optim
from d3sim.algos.d3gs.losses import SSimLoss, SSimLossV2, SSimLossV3, l1_loss, ssim
from cumm import tensorview as tv
from d3sim.algos.d3gs.tools import create_origin_3dgs_optimizers, init_original_3dgs_model
import contextlib 
import pccm 

import torch 
from rich.progress import Progress, TextColumn, TaskProgressColumn, TimeRemainingColumn, BarColumn, MofNCompleteColumn
from d3sim.algos.d3gs.gsplat import DefaultStrategy as StrategyRef
from d3sim.core.debugtools import create_enter_debug_store
import numpy as np 
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from cumm.inliner import measure_and_print_torch
def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def get_used_gpu_mem_GB():
    if IsAppleSiliconMacOs:
        return torch.mps.current_allocated_memory() / 1024 ** 3
    return torch.cuda.max_memory_allocated() / 1024**3

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class StepCtx:
    output: GaussianSplatOutput | None = None
    loss: torch.Tensor | None = None

def cams_to_cam_bundle(cams: list[BasicPinholeCamera]):
    cam2worlds = [cam.pose.to_world for cam in cams]
    cam2world_Ts = [cam2world[:3].T for cam2world in cam2worlds]
    cam2world_Ts_onearray = np.stack(cam2world_Ts, axis=0).reshape(-1, 4, 3)
    cam2world_Ts_th = torch.from_numpy(np.ascontiguousarray(cam2world_Ts_onearray)).to(D3SIM_DEFAULT_DEVICE)

    focal_xys = np.stack([cam.focal_length for cam in cams], axis=0).reshape(-1, 2)
    ppoints = np.stack([cam.principal_point for cam in cams], axis=0).reshape(-1, 2)
    focal_xy_ppoints = np.concatenate([focal_xys, ppoints], axis=1)
    focal_xy_ppoints_th = torch.from_numpy(np.ascontiguousarray(focal_xy_ppoints)).to(D3SIM_DEFAULT_DEVICE)
    frame_local_ids = None 
    if cams[0].frame_local_id is not None:
        for cam in cams:
            assert cam.frame_local_id is not None
        frame_local_ids = torch.tensor([cam.frame_local_id for cam in cams], dtype=torch.int32, device=D3SIM_DEFAULT_DEVICE)
    cam_bundle = CameraBundle(focal_xy_ppoints_th, cams[0].image_shape_wh, frame_local_ids, cam2world_Ts_th)
    return cam_bundle

class Trainer:
    def __init__(self, model: GaussianModelBase, cfg: config_def.Config, scene_extent: float) -> None:
        self.engine = BasicTrainEngine()
        self.model = model
        cfg.train.rescale_steps_by_batch_size(cfg.train.batch_size)
        self.device = torch.device(D3SIM_DEFAULT_DEVICE)
        optim, scheduler = create_origin_3dgs_optimizers(model, cfg.train.optim, cfg.train.batch_size, scene_extent)
        self.optims = optim 
        self.scheduler = scheduler
        self.cfg = cfg
        self.train_state = GaussianTrainState.create(model.xyz.shape[0], device=model.xyz.device)
        # self.train_state_ref = GaussianTrainState.create(model.xyz.shape[0], device=model.xyz.device)
        self.train_state.scene_scale = scene_extent
        self.train_state_ref = GaussianTrainState.create(model.xyz.shape[0], device=model.xyz.device)
        # self.train_state_ref = GaussianTrainState.create(model.xyz.shape[0], device=model.xyz.device)
        self.train_state_ref.scene_scale = scene_extent

        self.strategy = GaussianStrategyBase(cfg.train.strategy)
        self.strategy_ref = StrategyRef(verbose=True)

        self.gauss_op = GaussianSplatOp(cfg.model.op)
        
        self.use_strategy_ref = False
        self.engine.register_period_event("log", 1000, self._on_log)
        self.engine.register_period_event("up_sh", 1000 // cfg.train.batch_size, self._on_up_sh_degree_log)
        self.engine.register_base_event(TrainEventType.BeginIteration, self._on_iter_start)
        if not self.use_strategy_ref:
            self.engine.register_period_event("update_gs", cfg.train.strategy.refine_every, self._on_refine_gs)
            self.engine.register_period_event("reset_opacity", cfg.train.strategy.reset_opacity_every, self._on_reset_opacity)
        # self.engine.register_period_event("do_eval", 2000, self._on_reset_opacity)


    def _on_eval(self, step: int, scene: Scene):
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
            self.device
        )
        metrics = {"psnr": [], "ssim": [], "lpips": []}

        for cam_raw in scene.getTestCameras():
            cam = original_cam_to_d3sim_cam(cam_raw)
            out = rasterize_gaussians(self.model, cam, op=self.gauss_op, training=False)
            gt_image = cam_raw.original_image.to(self.device)
            color = out.color
            if not self.cfg.model.op.use_nchw:
                color = out.color.permute(0, 3, 1, 2)

            metrics["psnr"].append(psnr(color, gt_image[None]))
            metrics["ssim"].append(ssim(color, gt_image[None]))
            metrics["lpips"].append(lpips(color, gt_image[None]))

        psnr_val = torch.stack(metrics["psnr"]).mean()
        ssim_val = torch.stack(metrics["ssim"]).mean()
        lpips_val = torch.stack(metrics["lpips"]).mean()
        print(
            f"PSNR: {psnr_val.item():.3f}, SSIM: {ssim_val.item():.4f}, LPIPS: {lpips_val.item():.3f} "
            f"Number of GS: {len(self.model)}"
        )

    def train(self, scene: Scene):
        viewpoint_stack = None
        uv_grad_holder = torch.empty([self.cfg.train.batch_size, self.model.xyz.shape[0], 2], dtype=torch.float32, device=self.model.xyz.device)
        uv_grad_holder.requires_grad = True
        ema_loss_for_log = 0.0
        prog = Progress(
            TextColumn("[progress.description]{task.description}"),
            MofNCompleteColumn(),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),

        )
        ssim_loss_custom = SSimLoss(11, 3)
        ssim_loss_custom_v2 = SSimLossV3(11, 3)

        ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).to(D3SIM_DEFAULT_DEVICE)
        verbose = False
        total = self.cfg.train.iterations
        # vs gsplat
        # on 7000 step (4500000 gaussians):
        # gsplat forward: 8ms
        # gsplat backward: 17ms
        # our forward: 6ms
        # our backward: 10ms
        # our impl is 55% faster than gsplat.
        with contextlib.nullcontext():
        # with prog as progress:
            # task1 = progress.add_task("[red]Training...", total=self.cfg.train.iterations)
            for ev in tqdm.tqdm((self.engine.train_step_generator(0, range(total), total_step=total, 
                    step_ctx_creator=self.train_step_ctx)), total=total):
                # verbose = ev.cur_step > 6500
                with measure_and_print_torch("prep", enable=False):

            # for ev in (self.engine.train_step_generator(0, range(self.cfg.train.iterations), total_step=self.cfg.train.iterations, step_ctx_creator=self.train_step_ctx)):
                    cams = []
                    gt_imgs = []
                    for j in range(self.cfg.train.batch_size):
                        if not viewpoint_stack:
                            viewpoint_stack = scene.getTrainCameras().copy()
                        viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack)-1))
                        cam = original_cam_to_d3sim_cam(viewpoint_cam)
                        gt_image = viewpoint_cam.original_image.to(D3SIM_DEFAULT_DEVICE)
                        cams.append(cam)
                        gt_imgs.append(gt_image)
                    gt_image = torch.stack(gt_imgs, dim=0)
                    if len(cams) == 1:
                        cam_bundle = cams[0]
                    else:
                        cam_bundle = cams_to_cam_bundle(cams)
                    if uv_grad_holder.shape[0] != self.model.xyz.shape[0]:
                        uv_grad_holder = torch.empty([self.cfg.train.batch_size, self.model.xyz.shape[0], 2], dtype=torch.float32, device=self.model.xyz.device)
                        uv_grad_holder.requires_grad_(True)
                with measure_and_print_torch(f"fwd-{self.model.xyz.shape[0]}", enable=verbose):
                    # with tv.measure_and_print("rasterize"):
                    out = rasterize_gaussians_dynamic(self.model, cam_bundle, op=self.gauss_op, training=True, uv_grad_holder=uv_grad_holder)
                with measure_and_print_torch(f"fwd-loss-{self.model.xyz.shape[0]}", enable=verbose):

                    assert ev.step_ctx_value is not None 
                    ev.step_ctx_value.output = out
                    color = out.color
                    if not self.cfg.model.op.use_nchw:
                        color = out.color.permute(0, 3, 1, 2)
                    Ll1 = l1_loss(color, gt_image)
                    # WARNING: ssim loss in mps is very slow. 40ms (batch_size=8) in m3.
                    loss = (1.0 - self.cfg.train.lambda_dssim) * Ll1 + self.cfg.train.lambda_dssim * (1.0 - ssim(color, gt_image))
                    # with tv.measure_and_print("rasterize-bwd"):
                with measure_and_print_torch(f"bwd-{self.model.xyz.shape[0]}", enable=verbose):

                    loss.backward()
                duv = uv_grad_holder.grad
                ev.step_ctx_value.loss = loss
                assert duv is not None 
                with measure_and_print_torch("optim", enable=verbose):

                    for optim in self.optims.values():
                        optim.step()
                        # optim.zero_grad(set_to_none = True)
                # with measure_and_print_torch("optim zero_grad", enable=verbose):

                    for optim in self.optims.values():
                        # optim.step()
                        optim.zero_grad(set_to_none = True)
                # with measure_and_print_torch("remain", enable=verbose):

                if (ev.cur_step + 1) < self.cfg.train.strategy.refine_stop_iter:
                    if not self.use_strategy_ref:
                        # print(uv_grad_holder.shape, )
                        # for i in range(self.cfg.train.batch_size):
                        #     self.strategy.update_v2(self.train_state, out.select_batch(i), duv[i], batch_size=self.cfg.train.batch_size)
                        self.strategy.update_v2_batch(self.train_state, out, duv)
                        # print(torch.linalg.norm(self.train_state.duv_ndc_length - self.train_state_ref.duv_ndc_length))
                        # breakpoint()
                    # self.strategy.update(self.train_state_ref, out, duv)
                    # print(torch.linalg.norm(self.train_state_ref.duv_ndc_length - self.train_state.duv_ndc_length))
                    # print(torch.linalg.norm(self.train_state_ref.count - self.train_state.count))
                    # breakpoint()
                if self.use_strategy_ref:

                    params = self.model.get_all_tensor_fields()
                    state_dict = {
                        **self.train_state_ref.get_all_tensor_fields(),
                        "scene_scale": self.train_state_ref.scene_scale
                    }
                    self.strategy_ref.step_post_backward(
                        params,
                        self.optims,
                        state_dict,
                        ev.cur_step + 1,
                        {
                            "duv": uv_grad_holder,
                            "width": out.color.shape[2],
                            "height": out.color.shape[1],
                            "radii": out.radii,
                        }
                    )
                    # if ev.cur_step + 1 == 600:
                    #     print(torch.linalg.norm(self.train_state_ref.duv_ndc_length - self.train_state.duv_ndc_length))
                    #     print(torch.linalg.norm(self.train_state_ref.count - self.train_state.count))
                    #     self.strategy.refine_gs(self.model, self.optims, self.train_state, ev.cur_step + 1) 
                    #     breakpoint()
                    for k, p in params.items():
                        setattr(self.model, k, p)
                    for k, p in state_dict.items():
                        setattr(self.train_state_ref, k, p)
                uv_grad_holder.grad = None

                save_root = PACKAGE_ROOT / "build/debug"
                with torch.no_grad():
                    if (ev.cur_step + 1) % (1000 // self.cfg.train.batch_size) == 0:
                        out_color = out.color[0]
                        if not self.cfg.model.op.use_nchw:
                            out_color = out_color.permute(2, 0, 1)
                        out_color_u8 = out_color.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                        out_color_u8 = out_color_u8[..., ::-1]
                        gt_image_u8 = gt_image[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                        gt_image_u8 = gt_image_u8[..., ::-1]
                        out_color_u8 = np.concatenate([out_color_u8, gt_image_u8], axis=0)
                        # out_color_u8 = (out_color.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
                        import cv2 
                        cv2.imwrite(str(save_root / f"test_train_{ev.cur_step}.png"), out_color_u8)

                with torch.no_grad():
                    if (ev.cur_step + 1) % (1000 // self.cfg.train.batch_size) == 0:
                        self._on_eval(ev.cur_step + 1, scene)
                    # progress.print(f"LosVals: {loss.item():.{7}f}", self.model.xyz.shape[0])

                    ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                    if (ev.cur_step + 1) % (100 // self.cfg.train.batch_size) == 0:
                        print(f"Loss: {ema_loss_for_log:.{7}f}, Mem: {get_used_gpu_mem_GB():.3f} GB")
                    # progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                # if ev.cur_step >= 10:
                #     raise NotImplementedError
                # progress.update(task1, advance=1)

    def _on_iter_start(self, ev: TrainEvent):
        for param_group in self.optims["xyz"].param_groups:
            if param_group["name"] == "xyz":
                lr = self.scheduler(ev.cur_step + 1)
                param_group['lr'] = lr
                break

    def _on_log(self, ev: TrainEvent):
        assert ev.step_ctx_value is not None
        assert ev.step_ctx_value.loss is not None
        ema_loss_for_log = 0.0

    def _on_up_sh_degree_log(self, ev: TrainEvent):
        if self.model.cur_sh_degree < 3:
            self.model.cur_sh_degree += 1
            print(f"Up SH degree to {self.model.cur_sh_degree}")

    def _on_refine_gs(self, ev: TrainEvent):
        if self.use_strategy_ref:
            return 
        if ev.cur_step + 1 < self.cfg.train.strategy.refine_stop_iter:
            if (ev.cur_step + 1) > self.cfg.train.strategy.refine_start_iter:
            # if (ev.cur_step + 1) > 200:
                self.strategy.refine_gs(self.model, self.optims, self.train_state, ev.cur_step + 1) 
                # breakpoint()
                torch.cuda.empty_cache()
                print("REFINE GS!!!", self.model.xyz.shape[0])

    def _on_reset_opacity(self, ev: TrainEvent):
        if self.use_strategy_ref:
            return 

        if ev.cur_step + 1 < self.cfg.train.strategy.refine_stop_iter:
            print("RESET GS OPACITY!!!")
            origin_opacity = self.model.opacity
            opacities_new = torch.empty_like(self.model.opacity)
            opacity_thresh = self.cfg.train.strategy.reset_opacity_thresh
            code = pccm.code()
            code.raw(f"""
            namespace op = tv::arrayops;
            using math_op_t = tv::arrayops::MathScalarOp<float>;
            auto opacity_val = $origin_opacity[i];
            """)
            if self.model.fused_opacity_act_op is not None:
                code.raw(f"""
                opacity_val = tv::array<float, 1>{{opacity_val}}.op<op::{self.model.fused_opacity_act_op[0]}>()[0];
                """)
            code.raw(f"""
            opacity_val = math_op_t::min(opacity_val, $opacity_thresh);
            opacity_val = math_op_t::log(opacity_val / (1.0f - opacity_val));
            $opacities_new[i] = opacity_val;
            """)
            INLINER.kernel_1d("reset_opacity", opacities_new.shape[0], 0, code)
            # opacities_new = inverse_sigmoid(torch.min(self.model.opacity, torch.ones_like(self.model.opacity)*self.cfg.train.strategy.reset_opacity_thresh))
            opacities_new.requires_grad_(True)
            opacities_new_param = torch.nn.Parameter(opacities_new)
            self.model.opacity = opacities_new_param
            reset_param_in_optim(self.model, self.optims, "opacity")

    @contextlib.contextmanager
    def train_step_ctx(self):
        yield StepCtx()


def __main():
    if IsAppleSiliconMacOs:
        path = "/Users/yanyan/Downloads/360_v2/garden"
    else:
        path = "/root/autodl-tmp/garden_scene/garden"
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    scene_info, first_cam = load_scene_info_and_first_cam(path, image_folder="images_8" if IsAppleSiliconMacOs else "images_4")
    scene = Scene(scene_info, True)
    print("Load Dataset GPU Memory", get_used_gpu_mem_GB())
    points = scene_info.point_cloud.points
    if IsAppleSiliconMacOs:
        cfg = config_def.Config(
            model=config_def.Model(config_def.GaussianSplatConfig(enable_32bit_sort=True, gaussian_std_sigma=3.0, verbose=False)),
            train=config_def.Train(iterations=7000, batch_size=8)
        )
    else:
        cfg = config_def.Config(
            model=config_def.Model(config_def.GaussianSplatConfig(enable_32bit_sort=False, gaussian_std_sigma=3.0)),
            train=config_def.Train(iterations=7000, batch_size=8)
        )
    model = GaussianModelOriginFused.empty_parameter(points.shape[0], 3, True, device=torch.device(D3SIM_DEFAULT_DEVICE)) 
    init_original_3dgs_model(model, points, scene_info.point_cloud.colors)
    print("Init Model GPU Memory", get_used_gpu_mem_GB())

    # (_, xyz, feat_dc, feat_rest, scale, rot, opac, *_) = torch.load("/root/Projects/3dgs/gaussian-splatting/init_model.pth")

    # print(torch.linalg.norm(xyz - model.xyz))
    # print(torch.linalg.norm(feat_dc.reshape(-1) - model.color_sh_base.reshape(-1)))
    # print(torch.linalg.norm(feat_rest.reshape(-1) - model.color_sh.reshape(-1)))
    # print(torch.linalg.norm(scale.reshape(-1) - model.scale.reshape(-1)))
    # print(torch.linalg.norm(opac.reshape(-1) - model.opacity.reshape(-1)))

    # breakpoint()
    trainer = Trainer(model, cfg, scene_info.nerf_normalization["radius"])
    with create_enter_debug_store():

        trainer.train(scene)

if __name__ == "__main__":
    __main() 
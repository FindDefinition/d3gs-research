
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from d3sim.algos.d3gs.base import GaussianModelBase
from d3sim.algos.d3gs.config_def import GaussianSplatConfig
from d3sim.algos.d3gs.data_base import D3simDataset
from d3sim.algos.d3gs.render import GaussianSplatOp, rasterize_gaussians
from d3sim.algos.d3gs.tools import load_3dgs_origin_model
from d3sim.constants import D3SIM_DEFAULT_DEVICE, IsAppleSiliconMacOs
import torch 


def eval_dataset(model: GaussianModelBase, ds: D3simDataset, op: GaussianSplatOp):
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(D3SIM_DEFAULT_DEVICE)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(D3SIM_DEFAULT_DEVICE)
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
        D3SIM_DEFAULT_DEVICE
    )
    metrics = {"psnr": [], "ssim": [], "lpips": []}

    for cam_raw in ds.get_cameras("test"):
        cam = cam_raw
        out = rasterize_gaussians(model, cam, op=op, training=False)
        gt_image = cam_raw.get_image_torch(device=D3SIM_DEFAULT_DEVICE)
        color = out.color
        if not op.is_nchw:
            color = out.color.permute(0, 3, 1, 2)

        metrics["psnr"].append(psnr(color, gt_image[None]))
        metrics["ssim"].append(ssim(color, gt_image[None]))
        metrics["lpips"].append(lpips(color, gt_image[None]))

    psnr_val = torch.stack(metrics["psnr"]).mean()
    ssim_val = torch.stack(metrics["ssim"]).mean()
    lpips_val = torch.stack(metrics["lpips"]).mean()
    print(
        f"PSNR: {psnr_val.item():.3f}, SSIM: {ssim_val.item():.4f}, LPIPS: {lpips_val.item():.3f} "
        f"Number of GS: {len(model)}"
    )

def __main():
    from d3sim.algos.d3gs.origin.data.load import OriginDataset
    
    if IsAppleSiliconMacOs:
        path = "/Users/yanyan/Downloads/360_v2/garden"
        ckpt_path = "/Users/yanyan/Downloads/models/garden/point_cloud/iteration_30000/point_cloud.ply"
    else:
        path = "/root/autodl-tmp/garden_scene/garden"
        ckpt_path = "/root/autodl-tmp/garden_model/garden/point_cloud/iteration_30000/point_cloud.ply"
        ckpt_path = "/root/Projects/3dgs/gaussian-splatting/output/61637b5a-5/point_cloud/iteration_7000/point_cloud.ply"
    mod = load_3dgs_origin_model(ckpt_path, True, True)

    ds = OriginDataset(path, "images_8" if IsAppleSiliconMacOs else "images_4", True, [1.0])
    op = GaussianSplatOp(GaussianSplatConfig())
    eval_dataset(mod, ds, op)

if __name__ == "__main__":
    __main()
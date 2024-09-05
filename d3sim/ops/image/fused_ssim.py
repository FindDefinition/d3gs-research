from numpy import imag
from d3sim.constants import D3SIM_DEFAULT_DEVICE, IsAppleSiliconMacOs
from d3sim.csrc.inliner import INLINER
import torch
import pccm 
from cumm import tensorview as tv 
from math import exp
import torch.nn.functional as F
from cumm.inliner.sympy_codegen import VectorSymOperator, Scalar, Vector, VectorExpr
from torch.autograd.function import once_differentiable
from cumm.inliner import measure_and_print_torch

class SSIMOperator(VectorSymOperator):
    def forward(self, mu1, mu2, mu11, mu22, mu12) -> dict[str, VectorExpr]:
        mu1_sq = mu1 * mu1 
        mu2_sq = mu2 * mu2 
        mu1_mu2 = mu1 * mu2 
        sigma1_sq = mu11 - mu1_sq
        sigma2_sq = mu22 - mu2_sq 
        sigma12 = mu12 - mu1_mu2
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return {
            "ssim_map": ssim_map,
        }


_CACHED_CODES: dict[str, pccm.FunctionCode] = {}

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = (_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window.to(D3SIM_DEFAULT_DEVICE)

def ssim_loss(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    # if img1.is_cuda:
    #     window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def ssim_map(img1, img2, window_size=11):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    # if img1.is_cuda:
    #     window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return get_ssim_map(img1, img2, window, window_size, channel)


def get_ssim_map(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    # print(mu1)
    mu1_sq = mu1.square()
    mu2_sq = mu2.square()
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / 
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)))

    return ssim_map

# get_ssim_map_compiled = torch.compile(get_ssim_map, backend="inductor")

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    ssim_map = get_ssim_map(img1, img2, window, window_size, channel)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim_forward(images_x: torch.Tensor, images_y: torch.Tensor, window_size: int, training: bool = True, imgx_is_nhwc: bool = False):
    # images_x: NCHW
    assert images_x.is_contiguous() and images_y.is_contiguous()
    assert images_x.ndim == 4 and images_y.ndim == 4
    num_channels = images_x.size(-3 if not imgx_is_nhwc else -1)
    assert num_channels == images_y.size(-3)
    batch_size = images_x.shape[0]
    width = images_y.shape[3]
    height = images_y.shape[2]
    ssim_res = torch.empty_like(images_y)
    tile_size_x = 16
    tile_size_y = 16
    block_size = tile_size_x * tile_size_y
    tile_num_x = tv.div_up(width, tile_size_x)
    tile_num_y = tv.div_up(height, tile_size_y)
    assert window_size % 2 == 1
    padding = window_size // 2
    kernel_unique_name = f"fused_ssim_{window_size}_{num_channels}_{training}_{imgx_is_nhwc}"
    dmu1 = None 
    dmu11 = None
    dmu12 = None
    if training:
        dmu1 = torch.empty_like(images_y)
        dmu11 = torch.empty_like(images_y)
        dmu12 = torch.empty_like(images_y)
    if kernel_unique_name in _CACHED_CODES:
        code = _CACHED_CODES[kernel_unique_name]
    else:
        window = create_window(window_size, num_channels).cpu().numpy()[0][0]
        code = pccm.code()
        if IsAppleSiliconMacOs:
            code.raw(f"""
            tv::array<uint32_t, 2> pixel_idx_xy{{threadgroupPositionInGrid.x * {tile_size_x} + threadPositionInThreadgroup.x, 
                                                 threadgroupPositionInGrid.y * {tile_size_y} + threadPositionInThreadgroup.y}};
            tv::array<uint32_t, 2> tile_idx_xy{{threadgroupPositionInGrid.x, threadgroupPositionInGrid.y}};
            tv::array<uint32_t, 2> thread_idx_xy{{threadPositionInThreadgroup.x, threadPositionInThreadgroup.y}};

            uint thread_rank = threadPositionInThreadgroup.y * {tile_size_x} + threadPositionInThreadgroup.x;
            int batch_idx = threadgroupPositionInGrid.z;
            """)
        else:
            code.raw(f"""
            tv::array<uint32_t, 2> pixel_idx_xy{{blockIdx.x * {tile_size_x} + threadIdx.x,
                                                    blockIdx.y * {tile_size_y} + threadIdx.y}};
            
            tv::array<uint32_t, 2> tile_idx_xy{{blockIdx.x, blockIdx.y}};
            tv::array<uint32_t, 2> thread_idx_xy{{threadIdx.x, threadIdx.y}};
            uint32_t thread_rank = threadIdx.y * {tile_size_x} + threadIdx.x;
            int batch_idx = blockIdx.z;
            """)

        code.raw(f"""
        tv::array<uint32_t, 2> pixel_idx_xy_base{{tile_idx_xy[0] * {tile_size_x},
                                                tile_idx_xy[1] * {tile_size_y}}};

        bool pixel_valid = pixel_idx_xy[0] < $width && pixel_idx_xy[1] < $height;
        int CHW_offset = batch_idx * {num_channels} * $width * $height;
        auto image_x_ptr = $images_x + batch_idx * {num_channels} * $width * $height;
        auto image_y_ptr = $images_y + batch_idx * {num_channels} * $width * $height;
        auto ssim_res_ptr = $ssim_res + batch_idx * {num_channels} * $width * $height;
        constexpr int kPaddedTileSizeX = {tile_size_x} + {2 * padding};
        constexpr int kPaddedTileSizeY = {tile_size_y} + {2 * padding};

        constexpr int kLoadCount = kPaddedTileSizeX * kPaddedTileSizeY;
        constexpr int kLoadIters = (kLoadCount + {block_size} - 1) / {block_size};

        TV_SHARED_MEMORY float buf_x[{tile_size_y} + {2 * padding}][{tile_size_x} + {2 * padding}];
        TV_SHARED_MEMORY float buf_y[{tile_size_y} + {2 * padding}][{tile_size_x} + {2 * padding}];
        """)
        with code.for_(f"int i = 0; i < {num_channels}; ++i"):
            img_x_index_str = "i * $width * $height + pixel_idx_y * $width + pixel_idx_x" if not imgx_is_nhwc else "pixel_idx_y * $width * $num_channels + pixel_idx_x * $num_channels + i"
            code.raw(f"""
            tv::parallel::block_sync_shared_io();

            // 32x32 block load (32 + 2 * padding) x (32 + 2 * padding) data
            TV_PRAGMA_UNROLL
            for (int j = 0; j < kLoadIters; ++j){{
                int load_idx = thread_rank + j * {block_size};
                int load_idx_x = load_idx % kPaddedTileSizeX;
                int load_idx_y = load_idx / kPaddedTileSizeX;
                int pixel_idx_x = pixel_idx_xy_base[0] + load_idx_x - {padding};
                int pixel_idx_y = pixel_idx_xy_base[1] + load_idx_y - {padding};
                bool valid = pixel_idx_x >= 0 && pixel_idx_x < $width && pixel_idx_y >= 0 && pixel_idx_y < $height;
                if (load_idx < kLoadCount){{
                    buf_x[load_idx_y][load_idx_x] = valid ? image_x_ptr[{img_x_index_str}] : 0.0f;
                    buf_y[load_idx_y][load_idx_x] = valid ? image_y_ptr[i * $width * $height + pixel_idx_y * $width + pixel_idx_x] : 0.0f;
                }}
            }}
            tv::parallel::block_sync_shared_io();
            float val_x = 0.0f;
            float val_y = 0.0f;
            float val_xx = 0.0f;
            float val_yy = 0.0f;
            float val_xy = 0.0f;
            int local_x = thread_idx_xy[0];
            int local_y = thread_idx_xy[1];
            """)
            for i in range(window_size):
                for j in range(window_size):
                    gauss_val = float(window[i, j].item())
                    code.raw(f"""
                    val_x += buf_x[local_y + {i}][local_x + {j}] * {gauss_val}f;
                    val_y += buf_y[local_y + {i}][local_x + {j}] * {gauss_val}f;
                    val_xx += buf_x[local_y + {i}][local_x + {j}] * buf_x[local_y + {i}][local_x + {j}] * {gauss_val}f;
                    val_yy += buf_y[local_y + {i}][local_x + {j}] * buf_y[local_y + {i}][local_x + {j}] * {gauss_val}f;
                    val_xy += buf_x[local_y + {i}][local_x + {j}] * buf_y[local_y + {i}][local_x + {j}] * {gauss_val}f;
                    """)
            code.raw(f"""

            float mu1_sq = val_x * val_x;
            float mu2_sq = val_y * val_y;
            float mu1_mu2 = val_x * val_y;
            float sigma1_sq = val_xx - mu1_sq;
            float sigma2_sq = val_yy - mu2_sq;
            float sigma12 = val_xy - mu1_mu2;
            float C1 = 0.01f * 0.01f;
            float C2 = 0.03f * 0.03f;
            float ssim_val = ((2.0f * mu1_mu2 + C1) * (2.0f * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2));

            """)
            with code.if_("pixel_valid"):
                code.raw(f"""
                ssim_res_ptr[i * $width * $height + pixel_idx_xy[1] * $width + pixel_idx_xy[0]] = ssim_val;
                """)
                if training:
                    code.raw(f"""
                    auto mu1 = val_x;
                    auto mu2 = val_y;
                    auto mu11 = val_xx;
                    auto mu22 = val_yy;
                    auto mu12 = val_xy;
                    
                    auto dmu1_val = 2*(mu1*(2*mu1*mu2 + 0.0001F)*(((mu1)*(mu1)) + ((mu2)*(mu2)) + 0.0001F)*(-2*mu1*mu2 + 2*mu12 + 0.0009F) - mu1*(2*mu1*mu2 + 0.0001F)*(-2*mu1*mu2 + 2*mu12 + 0.0009F)*(-((mu1)*(mu1)) + mu11 - ((mu2)*(mu2)) + mu22 + 0.0009F) + mu2*(((mu1)*(mu1)) + ((mu2)*(mu2)) + 0.0001F)*(-4*mu1*mu2 + 2*mu12 + 0.0008F)*(-((mu1)*(mu1)) + mu11 - ((mu2)*(mu2)) + mu22 + 0.0009F))/(((((mu1)*(mu1)) + ((mu2)*(mu2)) + 0.0001F)*(((mu1)*(mu1)) + ((mu2)*(mu2)) + 0.0001F))*((-((mu1)*(mu1)) + mu11 - ((mu2)*(mu2)) + mu22 + 0.0009F)*(-((mu1)*(mu1)) + mu11 - ((mu2)*(mu2)) + mu22 + 0.0009F)));
                    auto dmu11_val = -(2*mu1*mu2 + 0.0001F)*(-2*mu1*mu2 + 2*mu12 + 0.0009F)/((((mu1)*(mu1)) + ((mu2)*(mu2)) + 0.0001F)*((-((mu1)*(mu1)) + mu11 - ((mu2)*(mu2)) + mu22 + 0.0009F)*(-((mu1)*(mu1)) + mu11 - ((mu2)*(mu2)) + mu22 + 0.0009F)));
                    auto dmu12_val = 2*(2*mu1*mu2 + 0.0001F)/((((mu1)*(mu1)) + ((mu2)*(mu2)) + 0.0001F)*(-((mu1)*(mu1)) + mu11 - ((mu2)*(mu2)) + mu22 + 0.0009F));
                    $dmu1[CHW_offset + i * $width * $height + pixel_idx_xy[1] * $width + pixel_idx_xy[0]] = dmu1_val;
                    $dmu11[CHW_offset + i * $width * $height + pixel_idx_xy[1] * $width + pixel_idx_xy[0]] = dmu11_val;
                    $dmu12[CHW_offset + i * $width * $height + pixel_idx_xy[1] * $width + pixel_idx_xy[0]] = dmu12_val;

                    """)
                    """
                    $mu1[CHW_offset + i * $width * $height + pixel_idx_xy[1] * $width + pixel_idx_xy[0]] = val_x;
                    $mu2[CHW_offset + i * $width * $height + pixel_idx_xy[1] * $width + pixel_idx_xy[0]] = val_y;
                    $mu11[CHW_offset + i * $width * $height + pixel_idx_xy[1] * $width + pixel_idx_xy[0]] = val_xx;
                    $mu22[CHW_offset + i * $width * $height + pixel_idx_xy[1] * $width + pixel_idx_xy[0]] = val_yy;
                    $mu12[CHW_offset + i * $width * $height + pixel_idx_xy[1] * $width + pixel_idx_xy[0]] = val_xy;

                    """
        _CACHED_CODES[kernel_unique_name] = code
    launch_param = tv.LaunchParam(
        (tile_num_x, tile_num_y, batch_size),
        (tile_size_x, tile_size_y, 1))
    INLINER.kernel_raw(kernel_unique_name, launch_param, code)
    return ssim_res, (dmu1, dmu11, dmu12)

def ssim_backward(dssim_map: torch.Tensor, mu_tensors: tuple[torch.Tensor, ...], images_x: torch.Tensor, images_y: torch.Tensor, window_size: int, imgx_is_nhwc: bool = False):
    # dssim_map: NCHW
    assert images_x.ndim == 4 and images_y.ndim == 4
    num_channels = images_x.size(-3 if not imgx_is_nhwc else -1)
    assert num_channels == images_y.size(-3)
    batch_size = images_x.shape[0]
    width = images_y.shape[3]
    height = images_y.shape[2]
    dimages_x = torch.empty_like(images_x)
    tile_size_x = 32
    tile_size_y = 16
    block_size = tile_size_x * tile_size_y
    tile_num_x = tv.div_up(width, tile_size_x)
    tile_num_y = tv.div_up(height, tile_size_y)
    assert window_size % 2 == 1
    padding = window_size // 2

    dmu1 = mu_tensors[0]
    dmu11 = mu_tensors[1]
    dmu12 = mu_tensors[2]
    # mu22_ten = mu_tensors[3]
    # mu12_ten = mu_tensors[4]
    # calc three conv grad
    # with tv.measure_and_print("ssim_bwd_prep"):

    #     INLINER.kernel_1d("ssim_bwd_prep", images_x.numel(), 0, f"""
    #     auto dssim_map_val = $dssim_map[i];
    #     $dmu1[i] *= dssim_map_val;
    #     $dmu11[i] *= dssim_map_val;
    #     $dmu12[i] *= dssim_map_val;
    #     """)
    # here we only need to run three gaussian convs.
    kernel_unique_name = f"fused_ssim_bwd_{window_size}_{num_channels}_{imgx_is_nhwc}"
    if kernel_unique_name in _CACHED_CODES:
        code = _CACHED_CODES[kernel_unique_name]
    else:
        window = create_window(window_size, num_channels).cpu().numpy()[0][0]
        code = pccm.code()
        if IsAppleSiliconMacOs:
            code.raw(f"""
            tv::array<uint32_t, 2> pixel_idx_xy{{threadgroupPositionInGrid.x * {tile_size_x} + threadPositionInThreadgroup.x, 
                                                 threadgroupPositionInGrid.y * {tile_size_y} + threadPositionInThreadgroup.y}};
            tv::array<uint32_t, 2> tile_idx_xy{{threadgroupPositionInGrid.x, threadgroupPositionInGrid.y}};
            tv::array<uint32_t, 2> thread_idx_xy{{threadPositionInThreadgroup.x, threadPositionInThreadgroup.y}};

            uint thread_rank = threadPositionInThreadgroup.y * {tile_size_x} + threadPositionInThreadgroup.x;
            int batch_idx = threadgroupPositionInGrid.z;
            """)
        else:
            code.raw(f"""
            tv::array<uint32_t, 2> pixel_idx_xy{{blockIdx.x * {tile_size_x} + threadIdx.x,
                                                 blockIdx.y * {tile_size_y} + threadIdx.y}};
            
            tv::array<uint32_t, 2> tile_idx_xy{{blockIdx.x, blockIdx.y}};
            tv::array<uint32_t, 2> thread_idx_xy{{threadIdx.x, threadIdx.y}};
            uint32_t thread_rank = threadIdx.y * {tile_size_x} + threadIdx.x;
            int batch_idx = blockIdx.z;
            """)

        code.raw(f"""
        tv::array<uint32_t, 2> pixel_idx_xy_base{{tile_idx_xy[0] * {tile_size_x},
                                                tile_idx_xy[1] * {tile_size_y}}};

        bool pixel_valid = pixel_idx_xy[0] < $width && pixel_idx_xy[1] < $height;
        auto image_x_ptr = $images_x + batch_idx * {num_channels} * $width * $height;
        auto image_y_ptr = $images_y + batch_idx * {num_channels} * $width * $height;
        
        auto dmu1_ptr = $dmu1 + batch_idx * {num_channels} * $width * $height;
        auto dmu11_ptr = $dmu11 + batch_idx * {num_channels} * $width * $height;
        auto dmu12_ptr = $dmu12 + batch_idx * {num_channels} * $width * $height;

        auto dimages_x_ptr = $dimages_x + batch_idx * {num_channels} * $width * $height;
        auto dssim_map_ptr = $dssim_map + batch_idx * {num_channels} * $width * $height;
        constexpr int kPaddedTileSizeX = {tile_size_x} + {2 * padding};
        constexpr int kPaddedTileSizeY = {tile_size_y} + {2 * padding};

        constexpr int kLoadCount = kPaddedTileSizeX * kPaddedTileSizeY;
        constexpr int kLoadIters = (kLoadCount + {block_size} - 1) / {block_size};

        TV_SHARED_MEMORY float buf_x[{tile_size_y} + {2 * padding}][{tile_size_x} + {2 * padding}];
        TV_SHARED_MEMORY float buf_y[{tile_size_y} + {2 * padding}][{tile_size_x} + {2 * padding}];
        TV_SHARED_MEMORY float buf_z[{tile_size_y} + {2 * padding}][{tile_size_x} + {2 * padding}];

        """)
        with code.for_(f"int i = 0; i < {num_channels}; ++i"):
            code.raw(f"""
            tv::parallel::block_sync_shared_io();

            // 32x32 block load (32 + 2 * padding) x (32 + 2 * padding) data
            TV_PRAGMA_UNROLL
            for (int j = 0; j < kLoadIters; ++j){{
                int load_idx = thread_rank + j * {block_size};
                int load_idx_x = load_idx % kPaddedTileSizeX;
                int load_idx_y = load_idx / kPaddedTileSizeX;
                int pixel_idx_x = pixel_idx_xy_base[0] + load_idx_x - {padding};
                int pixel_idx_y = pixel_idx_xy_base[1] + load_idx_y - {padding};
                bool valid = pixel_idx_x >= 0 && pixel_idx_x < $width && pixel_idx_y >= 0 && pixel_idx_y < $height;
                if (load_idx < kLoadCount){{
                    auto dssim_map_val = valid ? dssim_map_ptr[i * $width * $height + pixel_idx_y * $width + pixel_idx_x] : 0.0f;
                    buf_x[load_idx_y][load_idx_x] = valid ? dssim_map_val * dmu1_ptr[i * $width * $height + pixel_idx_y * $width + pixel_idx_x] : 0.0f;
                    buf_y[load_idx_y][load_idx_x] = valid ? dssim_map_val * dmu11_ptr[i * $width * $height + pixel_idx_y * $width + pixel_idx_x] : 0.0f;
                    buf_z[load_idx_y][load_idx_x] = valid ? dssim_map_val * dmu12_ptr[i * $width * $height + pixel_idx_y * $width + pixel_idx_x] : 0.0f;
                }}
            }}
            tv::parallel::block_sync_shared_io();
            float dimg1 = 0.0f;
            float dimg1_square = 0.0f;
            float dimg1img2 = 0.0f;

            int local_x = thread_idx_xy[0];
            int local_y = thread_idx_xy[1];
            """)
            for i in range(window_size):
                for j in range(window_size):
                    gauss_val = float(window[i, j].item())
                    code.raw(f"""
                    dimg1 += buf_x[local_y + {i}][local_x + {j}] * {gauss_val}f;
                    dimg1_square += buf_y[local_y + {i}][local_x + {j}] * {gauss_val}f;
                    dimg1img2 += buf_z[local_y + {i}][local_x + {j}] * {gauss_val}f;
                    """)
            with code.if_("pixel_valid"):
                if imgx_is_nhwc:
                    code.raw(f"""
                    auto img1_val = image_x_ptr[pixel_idx_xy[1] * $width * $num_channels + pixel_idx_xy[0] * $num_channels + i];
                    """)
                else:
                    code.raw(f"""
                    auto img1_val = image_x_ptr[i * $width * $height + pixel_idx_xy[1] * $width + pixel_idx_xy[0]];
                    """)

                code.raw(f"""
                auto img2_val = image_y_ptr[i * $width * $height + pixel_idx_xy[1] * $width + pixel_idx_xy[0]];
                dimg1 += 2.0f * img1_val * dimg1_square + dimg1img2 * img2_val;
                """)
                if imgx_is_nhwc:
                    code.raw(f"""
                    dimages_x_ptr[pixel_idx_xy[1] * $width * $num_channels + pixel_idx_xy[0] * $num_channels + i] = dimg1;
                    """)
                else:
                    code.raw(f"""
                    dimages_x_ptr[i * $width * $height + pixel_idx_xy[1] * $width + pixel_idx_xy[0]] = dimg1;
                    """)
        _CACHED_CODES[kernel_unique_name] = code
    launch_param = tv.LaunchParam(
        (tile_num_x, tile_num_y, batch_size),
        (tile_size_x, tile_size_y, 1))
    # with tv.measure_and_print(kernel_unique_name):
    INLINER.kernel_raw(kernel_unique_name, launch_param, code)
    return dimages_x

class _FusedSSIMMap(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        images_x: torch.Tensor,
        images_y: torch.Tensor,
        window_size: int,
        training: bool = True,
        imgx_is_nhwc: bool = False
        
    ):
        res, ctx_tensors = ssim_forward(images_x, images_y, window_size, training, imgx_is_nhwc)
        if training:
            ctx.save_for_backward(images_x, images_y, *ctx_tensors)
        ctx.window_size = window_size
        ctx.imgx_is_nhwc = imgx_is_nhwc
        return res

    @staticmethod
    @once_differentiable
    def backward(ctx, dssim_map):
        assert dssim_map.is_contiguous()
        tensors = ctx.saved_tensors
        mu_tensors = tensors[2:]
        images_x = tensors[0]
        images_y = tensors[1]

        dimages_x = ssim_backward(dssim_map, mu_tensors, images_x, images_y, ctx.window_size, ctx.imgx_is_nhwc)
        
        return dimages_x, None, None, None, None

def fused_ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, size_average: bool = True, pred_is_nhwc: bool = False):
    res = _FusedSSIMMap.apply(pred, target, window_size, True, pred_is_nhwc)
    if size_average:
        return res.mean()
    else:
        return res.mean(1).mean(1).mean(1)

def _test_ssim_bwd():
    img1 = torch.rand(1, 3, 1080, 1920).to(D3SIM_DEFAULT_DEVICE)
    img2 = torch.rand(1, 3, 1080, 1920).to(D3SIM_DEFAULT_DEVICE)

    img1.requires_grad_(True)
    imgx_is_nhwc = False
    img1_my = img1.detach().clone()
    if imgx_is_nhwc:
        img1_my = img1_my.permute(0, 2, 3, 1).contiguous()
    img1_my.requires_grad_(True)
    window_size = 11
    ssim_map_res, ctx = ssim_forward(img1_my, img2, window_size, training=True, imgx_is_nhwc=imgx_is_nhwc)
    ssim_map_res_ref = ssim_map(img1, img2, window_size)
    print("FWD RES", torch.linalg.norm(ssim_map_res_ref - ssim_map_res))
    dssim_map = torch.rand_like(ssim_map_res)

    ssim_map_res_ref.backward(dssim_map)
    dimg1_ref = img1.grad

    dimg1_my = ssim_backward(dssim_map, ctx, img1_my, img2, window_size, imgx_is_nhwc=imgx_is_nhwc)
    if imgx_is_nhwc:
        dimg1_my = dimg1_my.permute(0, 3, 1, 2)
    print("BWD_RES", torch.linalg.norm(dimg1_my - dimg1_ref))

    for j in range(10):
        img1.grad = None 
        with measure_and_print_torch("ssim ref"):
            ssim_map_res_ref = ssim_map(img1, img2, window_size)
        with measure_and_print_torch("ssim ref bwd"):

            ssim_map_res_ref.backward(dssim_map)


    for j in range(10):
        img1.grad = None 
        with measure_and_print_torch("ssim my"):
            ssim_map_res, ctx = ssim_forward(img1_my,img2, window_size, training=True, imgx_is_nhwc=imgx_is_nhwc)
        with measure_and_print_torch("ssim my bwd"):

            dimg1_my = ssim_backward(dssim_map, ctx, img1_my, img2, window_size, imgx_is_nhwc=imgx_is_nhwc)

    breakpoint()
    print("?")

def _test_ssim_fwd():
    img1 = torch.rand(1, 3, 1080, 1920).to(D3SIM_DEFAULT_DEVICE)
    img2 = torch.rand(1, 3, 1080, 1920).to(D3SIM_DEFAULT_DEVICE)

    imgx_is_nhwc = False
    img1_my = img1.detach().clone()
    if imgx_is_nhwc:
        img1_my = img1_my.permute(0, 2, 3, 1).contiguous()
    window_size = 11
    ssim_map_res, ctx = ssim_forward(img1_my, img2, window_size, training=False, imgx_is_nhwc=imgx_is_nhwc)
    ssim_map_res_ref = ssim_map(img1, img2, window_size)
    print("FWD RES", torch.linalg.norm(ssim_map_res_ref - ssim_map_res))

    for j in range(10):
        img1.grad = None 
        with measure_and_print_torch("ssim ref"):
            ssim_map_res_ref = ssim_map(img1, img2, window_size)


    for j in range(10):
        img1.grad = None 
        with measure_and_print_torch("ssim my"):
            ssim_map_res, ctx = ssim_forward(img1_my,img2, window_size, training=False, imgx_is_nhwc=imgx_is_nhwc)

    breakpoint()
    print("?")


def _test_check_ssim_grad():
    op = SSIMOperator().build()
    out_grad_name_dict = {
        "ssim_map": "dssim_map_val",
    }
    print(op.generate_gradients_code("mu1", out_grad_name_dict))
    print("----")
    print(op.generate_gradients_code("mu11", out_grad_name_dict))
    print("----")
    print(op.generate_gradients_code("mu12", out_grad_name_dict))

if __name__ == "__main__":
    _test_ssim_bwd()
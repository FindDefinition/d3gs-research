#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from cumm.inliner import measure_and_print_torch
from d3sim.constants import D3SIM_DEFAULT_DEVICE

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

class SSimLoss:
    def __init__(self, window_size, channel) -> None:
        self.window = create_window(window_size, channel).to(D3SIM_DEFAULT_DEVICE).float()
        self.window_size = window_size 
        self.channel = channel

    def __call__(self, img1, img2, size_average=True):
        return _ssim(img1, img2, self.window, self.window_size, self.channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.square()
    mu2_sq = mu2.square()
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSimLossV2:
    def __init__(self, window_size, channel) -> None:
        self.window = create_window(window_size, channel).to(D3SIM_DEFAULT_DEVICE).float()
        
        self.window_5 = self.window.repeat(5, 1, 1, 1) # 3*5, 1, 11, 11
        self.window_size = window_size 
        self.channel = channel

    def __call__(self, img1, img2, size_average=True):
        return _ssim_v2(img1, img2, self.window_5, self.window_size, self.channel, size_average)

def _ssim_v2(img1, img2, window_5, window_size, channel, size_average=True):
    
    i1i2_i1i1_i2i2_i1i2 = torch.cat([img1, img2, img1.square(), img2.square(), img1*img2], dim=1)
    i1i2_i1i1_i2i2_i1i2_conv = F.conv2d(i1i2_i1i1_i2i2_i1i2, window_5, padding=window_size // 2, groups=channel * 5)
    mu1 = i1i2_i1i1_i2i2_i1i2_conv[:, :channel]
    mu2 = i1i2_i1i1_i2i2_i1i2_conv[:, channel:channel*2]

    mu1_sq = mu1.square()
    mu2_sq = mu2.square()
    mu1_mu2 = mu1 * mu2

    sigma1_sq = i1i2_i1i1_i2i2_i1i2_conv[:, 2*channel:3*channel] - mu1_sq
    sigma2_sq = i1i2_i1i1_i2i2_i1i2_conv[:, 3*channel:4*channel] - mu2_sq
    sigma12 = i1i2_i1i1_i2i2_i1i2_conv[:, 4*channel:5*channel] - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSimLossV3:
    def __init__(self, window_size, channel) -> None:
        self.window = create_window(window_size, channel).to(D3SIM_DEFAULT_DEVICE).float()
        self.window_size = window_size 
        self.channel = channel

    def __call__(self, img1, img2, size_average=True):
        return self._ssim_v3(img1, img2, self.window, self.window_size, self.channel, size_average)

    def _conv2d_one_channel(self, img):
        c0 = F.conv2d(img[:, :1], self.window[:1], padding=self.window_size // 2)
        c1 = F.conv2d(img[:, 1:2], self.window[1:2], padding=self.window_size // 2)
        c2 = F.conv2d(img[:, 2:3], self.window[2:3], padding=self.window_size // 2)
        return torch.cat([c0, c1, c2], dim=1)
        
    def _ssim_v3(self, img1, img2, window, window_size, channel, size_average=True):
        
        mu1 = self._conv2d_one_channel(img1)
        mu2 = self._conv2d_one_channel(img2)

        mu1_sq = mu1.square()
        mu2_sq = mu2.square()
        mu1_mu2 = mu1 * mu2

        sigma1_sq = self._conv2d_one_channel(img1 * img1, ) - mu1_sq
        sigma2_sq = self._conv2d_one_channel(img2 * img2, ) - mu2_sq
        sigma12 = self._conv2d_one_channel(img1 * img2, ) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

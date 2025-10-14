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
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from kornia.filters import laplacian, spatial_gradient
import kornia
import kornia.filters as kn_filters
import kornia.morphology as kn_morph

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l1_loss_mask(network_output, gt, mask = None):
    if mask is None:
        return l1_loss(network_output, gt)
    else:
        return torch.abs((network_output - gt) * mask).sum() / mask.sum()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def l2_loss_mask(network_output, gt, mask = None):
    if mask is None:
        return l2_loss(network_output, gt)
    else:
        return ((network_output - gt) ** 2 * mask).sum() / mask.sum()

def compute_Ls(F_hat, F_gt, S_hat, S_gt):
    # F_hat, F_gt: [B, 512, H, W]
    # S_hat: [B, num_class, H, W]
    # S_gt: [B, H, W] (long)

    # 1. Cosine similarity loss (1 - cosine similarity)
    # Flatten spatial dims for cosine_similarity
    cos_loss = 1 - F.cosine_similarity(F_hat, F_gt, dim=1).mean()

    # 2. Cross entropy loss
    ce_loss = F.cross_entropy(S_hat, S_gt)

    # Total loss
    Ls = cos_loss + ce_loss
    return Ls


def self_albedo_lightinvariance_loss(albedo_samples, albedo_gs, mask, msgloss):
    """
    计算albedo光照不变性loss

    Args:
        albedo_samples: [3, 3, H, W]，同一图片不同光照采样逆渲染得到的albedo
        albedo_gs: [3, H, W]，由gaussian渲染直接得到的albedo
        mask: [1, H, W]，掩码
        msgloss: MSGLoss类实例

    Returns:
        total_loss: 标量损失值
    """
    total_loss = 0.0
    # 只与第0和第2个采样albedo做loss
    for idx in [0, 2]:
        sample_albedo = albedo_samples[idx]  # [3, H, W]
        # l2 mask loss
        l2 = l2_loss_mask(albedo_gs, sample_albedo, mask)
        # msgloss
        msg = msgloss(albedo_gs.unsqueeze(0), sample_albedo.unsqueeze(0), mask.unsqueeze(0))
        total_loss += l2 + msg
    
    total_loss /= 2.0  # 平均损失
    return total_loss

    
def inter_view_loss(F, M):
    """
    计算伪视角和真实视角之间的对齐损失
    
    Args:
        F: 特征张量，shape为[3, 64, H, W]，其中F[1]为真实视角
        M: 掩码张量，shape为[3, 1, H, W]，其中M[1]为真实视角掩码
    
    Returns:
        loss: 标量损失值
    """
    # 确保输入在同一设备上
    F = F.to(F.device)
    M = M.to(F.device)
    
    # 获取真实视角的特征和掩码
    F_real = F[1]  # [64, H, W] - 真实视角特征
    M_real = M[1]  # [1, H, W] - 真实视角掩码
    
    # 计算真实视角的归一化特征 (分母)
    # F_real * M_real 后在空间维度求和，得到 [64]
    F_real_masked = F_real * M_real  # [64, H, W]
    M_real_sum = M_real.sum(dim=(-2, -1), keepdim=True)  # [1, 1, 1]
    F_real_norm = F_real_masked.sum(dim=(-2, -1)) / (M_real_sum.squeeze() + 1e-8)  # [64]
    
    total_loss = 0.0
    P = 2  # 伪视角数量
    
    # 遍历伪视角 (p=0, p=2，跳过p=1真实视角)
    for p in range(3):
        if p == 1:  # 跳过真实视角
            continue
            
        # 获取第p个伪视角的特征和掩码
        F_p = F[p]  # [64, H, W]
        M_p = M[p]  # [1, H, W]
        
        # 计算伪视角的归一化特征 (分子第一项)
        F_p_masked = F_p * M_p  # [64, H, W]
        M_p_sum = M_p.sum(dim=(-2, -1), keepdim=True)  # [1, 1, 1]
        F_p_norm = F_p_masked.sum(dim=(-2, -1)) / (M_p_sum.squeeze() + 1e-8)  # [64]
        
        # 计算损失项: ||F_p^* ⊙ M_p^* / ∑M_p^* - F̂ ⊙ M / ∑M||_2
        loss_term = torch.norm(F_p_norm - F_real_norm, p=2)
        total_loss += loss_term
    
    return total_loss

def inter_view_albedo_loss(A, M):
    """
    计算伪视角和真实视角之间的albedo对齐损失

    Args:
        A: albedo张量，shape为[3, 3, H, W]，其中A[1]为真实视角
        M: 掩码张量，shape为[3, 1, H, W]，其中M[1]为真实视角掩码

    Returns:
        loss: 标量损失值
    """
    A = A.to(A.device)
    M = M.to(A.device)

    # 获取真实视角的albedo和掩码
    A_real = A[1]  # [3, H, W]
    M_real = M[1]  # [1, H, W]

    # 计算真实视角的归一化albedo
    A_real_masked = A_real * M_real  # [3, H, W]
    M_real_sum = M_real.sum(dim=(-2, -1), keepdim=True)  # [1, 1, 1]
    A_real_norm = A_real_masked.sum(dim=(-2, -1)) / (M_real_sum.squeeze() + 1e-8)  # [3]

    total_loss = 0.0

    for p in range(3):
        if p == 1:
            continue

        A_p = A[p]  # [3, H, W]
        M_p = M[p]  # [1, H, W]

        A_p_masked = A_p * M_p  # [3, H, W]
        M_p_sum = M_p.sum(dim=(-2, -1), keepdim=True)  # [1, 1, 1]
        A_p_norm = A_p_masked.sum(dim=(-2, -1)) / (M_p_sum.squeeze() + 1e-8)  # [3]

        loss_term = torch.norm(A_p_norm - A_real_norm, p=2)
        total_loss += loss_term

    return total_loss


# def region_majority_class(S, M, omega_s=None):
#     """
#     S: [num_class, H, W] softmax概率
#     M: [1, H, W] 区域掩码
#     omega_s: [num_class] 类别权重，可选
#     返回: 区域主类别索引 int
#     """
#     # [num_class, H, W] -> [num_class, N]
#     mask = M.bool()
#     S_masked = S[:, mask[0]]  # [num_class, N]
#     if omega_s is not None:
#         S_masked = S_masked * omega_s[:, None]
#     # 每个像素的最大类别
#     pixel_cls = S_masked.argmax(dim=0)  # [N]
#     # 区域内最大投票类别
#     cls, counts = pixel_cls.unique(return_counts=True)
#     majority_cls = cls[counts.argmax()].item()
#     return majority_cls

# ...existing code...
def region_majority_class(S, M, omega_s=None):
    """
    S: [num_class, H, W] softmax概率
    M: [1, H, W] 区域掩码
    omega_s: [num_class] 类别权重，可选
    返回: 区域主类别索引 int
    """
    # [num_class, H, W] -> [num_class, N]
    mask = M.bool()
    S_masked = S[:, mask[0]]  # [num_class, N]

    # If mask contains no pixels, fall back to global sums to pick a class
    if S_masked.shape[1] == 0:
        # sum over spatial dims -> [num_class]
        # sums = S.view(S.shape[0], -1).sum(dim=1)
        # if omega_s is not None:
        #     sums = sums * omega_s
        # majority_cls = int(sums.argmax().item())
        # return majority_cls
        return None

    if omega_s is not None:
        S_masked = S_masked * omega_s[:, None]
    # 每个像素的最大类别
    pixel_cls = S_masked.argmax(dim=0)  # [N]
    # 区域内最大投票类别
    cls, counts = pixel_cls.unique(return_counts=True)
    majority_cls = cls[counts.argmax()].item()
    return majority_cls

def region_semantic_uniform(S, M, omega_s=None):
    """
    S: [num_class, H, W] softmax概率
    M: [1, H, W] 区域掩码
    omega_s: [num_class] 类别权重，可选
    返回: 区域主类别的one-hot [num_class, H, W]
    """
    majority_cls = region_majority_class(S, M, omega_s)

    if majority_cls is None:
        # mask区域内没有像素，返回全0
        return None
    one_hot = torch.zeros_like(S)
    one_hot[majority_cls] = 1.0
    return one_hot * M  # 只在mask区域内有效

def inter_view_loss_semantic(Features, M, S, omega_s=None):
    """
    F: [3, C, H, W]
    M: [3, 1, H, W]
    S: [3, num_class, H, W]
    omega_s: [num_class] or None
    """
    device = Features.device
    total_loss = 0.0
    total_sem_loss = 0.0

    # 真实视角
    F_real = Features[1]
    M_real = M[1]
    S_real = S[1]

    # 区域归一化特征
    F_real_masked = F_real * M_real
    M_real_sum = M_real.sum(dim=(-2, -1), keepdim=True)
    F_real_norm = F_real_masked.sum(dim=(-2, -1)) / (M_real_sum.squeeze() + 1e-8)

    # 区域主类别 one-hot
    S_real_softmax = F.softmax(S_real, dim=0)
    region_onehot = region_semantic_uniform(S_real_softmax, M_real, omega_s)  # [num_class, H, W]

    if region_onehot is None:
        # mask区域内没有像素，直接返回0 loss
        return torch.tensor(0.0, device=device)

    for p in [0, 2]:
        F_p = Features[p]
        M_p = M[p]
        S_p = S[p]

        # 特征对齐损失
        F_p_masked = F_p * M_p
        M_p_sum = M_p.sum(dim=(-2, -1), keepdim=True)
        F_p_norm = F_p_masked.sum(dim=(-2, -1)) / (M_p_sum.squeeze() + 1e-8)
        total_loss += torch.norm(F_p_norm - F_real_norm, p=2)

        # 区域语义一致性损失
        S_p_softmax = F.softmax(S_p, dim=0)
        # reshape region_onehot 到 S_p shape
        region_onehot_reshape = region_onehot  # [num_class, H, W]
        # 交叉熵损失（只在mask区域内）
        log_prob = torch.log(S_p_softmax + 1e-8)
        sem_loss = -(region_onehot_reshape * log_prob * M_p).sum() / (M_p.sum() + 1e-8)
        total_sem_loss += sem_loss

    # 平均
    total_loss = total_loss / 2.0
    total_sem_loss = total_sem_loss / 2.0

    return total_loss + total_sem_loss


def normal_consistency_loss_robust(n_hat, n_m, epsilon=1e-8):
    """
    计算法向量一致性损失（鲁棒版本）
    
    Args:
        n_hat: Gaussian预测的法向量，shape为[C, H, W]
        n_m: 神经网络预测的法向量，shape为[C, H, W]
        epsilon: 数值稳定性参数
    
    Returns:
        loss: 标量损失值
    """
    # 确保输入是tensor并在同一设备上
    if not isinstance(n_hat, torch.Tensor):
        n_hat = torch.tensor(n_hat)
    if not isinstance(n_m, torch.Tensor):
        n_m = torch.tensor(n_m)
    n_m = n_m.to(n_hat.device)
    
    # 将shape从[C, H, W]转换为[C, H*W]
    C, H, W = n_hat.shape
    n_hat_flat = n_hat.view(C, -1)  # [C, H*W]
    n_m_flat = n_m.view(C, -1)      # [C, H*W]
    
    # 计算向量的模长
    n_hat_norm = torch.sqrt(torch.sum(n_hat_flat ** 2, dim=0) + epsilon)  # [H*W]
    n_m_norm = torch.sqrt(torch.sum(n_m_flat ** 2, dim=0) + epsilon)      # [H*W]
    
    # 计算点积
    dot_product = torch.sum(n_hat_flat * n_m_flat, dim=0)  # [H*W]
    
    # 计算余弦相似度
    cosine_similarity = dot_product / (n_hat_norm * n_m_norm + epsilon)
    
    # 将余弦相似度限制在[-1, 1]范围内，避免数值误差
    cosine_similarity = torch.clamp(cosine_similarity, -1.0, 1.0)
    
    # 计算损失：L_m = 1 - cosine_similarity
    loss_per_pixel = 1.0 - cosine_similarity
    
    # 返回平均损失
    loss = torch.mean(loss_per_pixel)
    
    return loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, mask=None, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if mask is not None:
        img1 = img1 * mask + (1 - mask)
        img2 = img2 * mask + (1 - mask)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
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

def cal_gradient(data):
    """
    data: [1, C, H, W]
    """
    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(data.device)

    kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(data.device)

    weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
    weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    grad_x = F.conv2d(data, weight_x, padding='same')
    grad_y = F.conv2d(data, weight_y, padding='same')
    gradient = torch.abs(grad_x) + torch.abs(grad_y)

    return gradient


def bilateral_smooth_loss(data, image, mask):
    """
    image: [C, H, W]
    data: [C, H, W]
    mask: [C, H, W]
    """
    rgb_grad = cal_gradient(image.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)  # [1, H, W]
    data_grad = cal_gradient(data.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)  # [1, H, W]

    smooth_loss = (data_grad * (-rgb_grad).exp() * mask).mean()

    return smooth_loss


def second_order_edge_aware_loss(data, img):
    return (spatial_gradient(data[None], order=2)[0, :, [0, 2]].abs() * torch.exp(-10*spatial_gradient(img[None], order=1)[0].abs())).sum(1).mean()


def first_order_edge_aware_loss(data, img):
    return (spatial_gradient(data[None], order=1)[0].abs() * torch.exp(-spatial_gradient(img[None], order=1)[0].abs())).sum(1).mean()

def first_order_edge_aware_norm_loss(data, img):
    return (spatial_gradient(data[None], order=1)[0].abs() * torch.exp(-spatial_gradient(img[None], order=1)[0].norm(dim=1, keepdim=True))).sum(1).mean()

def first_order_loss(data):
    return spatial_gradient(data[None], order=1)[0].abs().sum(1).mean()

def tv_loss(depth):
    # return spatial_gradient(data[None], order=2)[0, :, [0, 2]].abs().sum(1).mean()
    h_tv = torch.square(depth[..., 1:, :] - depth[..., :-1, :]).mean()
    w_tv = torch.square(depth[..., :, 1:] - depth[..., :, :-1]).mean()
    return h_tv + w_tv

def get_tv_loss(
    gt_image: torch.Tensor,  # [3, H, W]
    prediction: torch.Tensor,  # [C, H, W]
    pad: int = 1,
    step: int = 1,
) -> torch.Tensor:
    if pad > 1:
        gt_image = F.avg_pool2d(gt_image, pad, pad)
        prediction = F.avg_pool2d(prediction, pad, pad)
    rgb_grad_h = torch.exp(
        -(gt_image[:, 1:, :] - gt_image[:, :-1, :]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    rgb_grad_w = torch.exp(
        -(gt_image[:, :, 1:] - gt_image[:, :, :-1]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    tv_h = torch.pow(prediction[:, 1:, :] - prediction[:, :-1, :], 2)  # [C, H-1, W]
    tv_w = torch.pow(prediction[:, :, 1:] - prediction[:, :, :-1], 2)  # [C, H, W-1]
    tv_loss = (tv_h * rgb_grad_h).mean() + (tv_w * rgb_grad_w).mean()

    if step > 1:
        for s in range(2, step + 1):
            rgb_grad_h = torch.exp(
                -(gt_image[:, s:, :] - gt_image[:, :-s, :]).abs().mean(dim=0, keepdim=True)
            )  # [1, H-1, W]
            rgb_grad_w = torch.exp(
                -(gt_image[:, :, s:] - gt_image[:, :, :-s]).abs().mean(dim=0, keepdim=True)
            )  # [1, H-1, W]
            tv_h = torch.pow(prediction[:, s:, :] - prediction[:, :-s, :], 2)  # [C, H-1, W]
            tv_w = torch.pow(prediction[:, :, s:] - prediction[:, :, :-s], 2)  # [C, H, W-1]
            tv_loss += (tv_h * rgb_grad_h).mean() + (tv_w * rgb_grad_w).mean()

    return tv_loss

def get_masked_tv_loss(
    mask: torch.Tensor,  # [1, H, W]
    gt_image: torch.Tensor,  # [3, H, W]
    prediction: torch.Tensor,  # [C, H, W]
    erosion: bool = False,
) -> torch.Tensor:
    rgb_grad_h = torch.exp(
        -(gt_image[:, 1:, :] - gt_image[:, :-1, :]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    rgb_grad_w = torch.exp(
        -(gt_image[:, :, 1:] - gt_image[:, :, :-1]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    tv_h = torch.pow(prediction[:, 1:, :] - prediction[:, :-1, :], 2)  # [C, H-1, W]
    tv_w = torch.pow(prediction[:, :, 1:] - prediction[:, :, :-1], 2)  # [C, H, W-1]

    # erode mask
    mask = mask.float()
    if erosion:
        kernel = mask.new_ones([7, 7])
        mask = kornia.morphology.erosion(mask[None, ...], kernel)[0]
    mask_h = mask[:, 1:, :] * mask[:, :-1, :]  # [1, H-1, W]
    mask_w = mask[:, :, 1:] * mask[:, :, :-1]  # [1, H, W-1]

    tv_loss = (tv_h * rgb_grad_h * mask_h).mean() + (tv_w * rgb_grad_w * mask_w).mean()

    return tv_loss

@torch.jit.script
def compute_scale_and_shift(prediction, target, mask):
    """Computes the optimal scale and shift according to least-squares
    criteria between the prediction and the target in the masked area

    params:
        pred (torch.Tensor): network prediction tensor (B x H x W)
        grnd (torch.Tensor): ground truth tensor (B x H x W)
        mask (torch.Tensor): mask denoting valid pixels (must be B x H x W)

    returns:
        x_0 (torch.Tensor): scales (B)
        x_1 (torch.Tensor): shifts (B)
    """
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 -
    # a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] -
                  a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] +
                  a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

def compute_ssi_pred(pred, grnd, mask):
    """Returns the provided predictions shifted and scaled such that they 
    minimize the L2 difference with the ground truth in the masked area

    params:
        pred (torch.Tensor): network prediction tensor (B x H x W)
        grnd (torch.Tensor): ground truth tensor (B x H x W)
        mask (torch.Tensor): mask denoting valid pixels (must be B x H x W)

    returns:
        (TODO): the network prediction optimally shifted and scaled
    """
    scale, shift = compute_scale_and_shift(pred, grnd, mask)

    # NOTE: early in training this scale can be negative, so we can simply clip it
    # at zero. It could also probably just be set to one if less than 0, it's just
    # to help stabilize early training until the network is making reasonable preds

    scale = torch.nn.functional.relu(scale)
    # scale[scale <= 0] = 1.0
    # scale = torch.abs(scale)

    return (pred * scale.view(-1, 1, 1)) + shift.view(-1, 1, 1)


@torch.jit.script
def resize_aa(img, scale: int):
    """TODO DESCRIPTION

    params:
        img (TODO): TODO
        scale (TODO): TODO

    returns:
        (TODO): TODO
    """
    if scale == 0:
        return img

    # blurred = TF.gaussian_blur(img, self.k_size[scale])
    # scaled = blurred[:, :, ::2**scale, ::2**scale]
    # blurred = img

    # NOTE: interpolate is noticeably faster than blur and sub-sample
    scaled = torch.nn.functional.interpolate(
        img,
        scale_factor=1/(2**scale),
        mode='bilinear',
        align_corners=True,
        antialias=True
    )
    return scaled

def lp_loss(pred, grnd, mask, p=2):
    """Performs a regular LP loss where P is specified. Can be used to
    compute both MSE (p=2) and L1 (p=1) loss functions

    params:
        pred (torch.Tensor): network prediction tensor (B x C x H x W)
        grnd (torch.Tensor): ground truth tensor (B x C x H x W)
        mask (torch.Tensor): mask denoting valid pixels (must be B x 1 x H x W)
        p (int) optional: degree of L norm (default 2)

    returns:
        (TODO): the mean LP loss between pixels in prediction and ground truth
    """
    if p == 1:
        lp_term = torch.nn.functional.l1_loss(pred, grnd, reduction='none') * mask
    else:
        lp_term = torch.nn.functional.mse_loss(pred, grnd, reduction='none') * mask

    return lp_term.sum() / (mask.sum() * lp_term.shape[1])


class MSGLoss():
    """Multi-scale Gradient Loss implementation

    params:
        scales (int) optional: TODO (default 4)
        taps (list) optional: TODO (default [1,1,1,1])
        k_size (list) optional: TODO (default [3,3,3,3])
        device (str) optional: TODO (default None)
    """
    def __init__(self, scales=4, taps=[1, 1, 1, 1], k_size=[3, 3, 3, 3], device=None):
        """Create an instance of MSGLoss.

        params:
            scales (int) optional: TODO (default 4)
            taps (list) optional: TODO (default [1,1,1,1])
            k_size (list) optional: TODO (default [3,3,3,3])
            device (str) optional: TODO (default None)
        """
        self.n_scale = scales
        self.taps = taps
        self.k_size = k_size
        self.device = device

        # pylint: disable-next=line-too-long
        assert len(self.taps) == self.n_scale, 'number of scales and number of taps must be the same'
        # pylint: disable-next=line-too-long
        assert len(self.k_size) == self.n_scale, 'number of scales and number of kernels must be the same'

        self.imgDerivative = ImageDerivative()

        self.erod_kernels = [torch.ones(2 * t + 1, 2 * t + 1) for t in self.taps]

        if self.device is not None:
            self.to_device(self.device)


    def to_device(self, device):
        """TODO DESCRIPTION

        params:
            device (str): TODO
        """
        self.imgDerivative.to_device(device)
        self.device = device
        self.erod_kernels = [kernel.to(device) for kernel in self.erod_kernels]


    def __call__(self, output, target, mask=None):
        """TODO DESCRIPTION

        params:
            output (TODO): TODO
            target (TODO): TODO
            mask (TODO) optional: TODO (default None)

        returns:
            (TODO): TODO
        """
        return self.forward(output, target, mask)

    def forward(self, output, target, mask):
        """TODO DESCRIPTION

        params:
            output (TODO): TODO
            target (TODO): TODO
            mask (TODO): TODO

        returns:
            loss (TODO): TODO
        """
        diff = output - target

        if mask is None:
            mask = torch.ones(diff.shape[0], 1, diff.shape[2], diff.shape[3])
            # print("diff.shape:",diff.shape)
            # mask=torch.ones(1,1,diff.shape[1],diff.shape[2])
            mask = mask.to(self.device)

        loss = 0
        for i in range(self.n_scale):
            # resize with antialias
            mask_resized = torch.floor(resize_aa(mask, i) + 0.001)

            # erosion to mask out pixels that are effected by unkowns
            mask_resized = kn_morph.erosion(mask_resized, self.erod_kernels[i])
            diff_resized = resize_aa(diff, i)

            # compute grads
            grad_mag = self.gradient_mag(diff_resized, i)

            # mean over channels
            grad_mag = torch.mean(grad_mag, dim=1, keepdim=True)

            # print("mssk_resized.shape:",mask_resized.shape)
            # print("grad_mag.shape:",grad_mag.shape)
            # average the per pixel diffs
            temp = mask_resized * grad_mag
            
            mask_sum = torch.sum(mask_resized)
            if mask_sum != 0:
                # pylint: disable-next=line-too-long
                loss += torch.sum(mask_resized * grad_mag) / (mask_sum * grad_mag.shape[1])

        loss /= self.n_scale
        return loss



    def gradient_mag(self, diff, scale):
        """TODO DESCRIPTION

        params:
            diff (TODO): TODO
            scale (TODO): TODO

        returns:
            grad_magnitude (TODO): TODO
        """
        # B x C x H x W
        grad_x, grad_y = self.imgDerivative(diff, self.taps[scale])

        # B x C x H x W
        grad_magnitude = torch.sqrt(torch.pow(grad_x, 2) + torch.pow(grad_y, 2) + 1e-8)

        return grad_magnitude


class ImageDerivative():
    """TODO DESCRIPTION

    params:
        device (str) optional: TODO (default None)
    """
    def __init__(self, device=None):
        """Creates an instance of ImageDerivative

        params:
            device (str) optional: TODO (default None)
        """
        # seperable kernel: first derivative, second prefiltering
        tap_3 = torch.tensor([
            [0.425287, -0.0000, -0.425287],
            [0.229879, 0.540242, 0.229879]])
        tap_5 = torch.tensor([
            [0.109604,  0.276691,  0.000000, -0.276691, -0.109604],
            [0.037659,  0.249153,  0.426375,  0.249153,  0.037659]])
        tap_7 = torch.tensor([0])
        tap_9 = torch.tensor([
            [0.0032, 0.0350, 0.1190, 0.1458, -0.0000, -0.1458, -0.1190, -0.0350, -0.0032],
            [0.0009, 0.0151, 0.0890, 0.2349, 0.3201, 0.2349, 0.0890, 0.0151, 0.0009]])
        tap_11 = torch.tensor([0])
        tap_13 = torch.tensor([
            [0.0001, 0.0019, 0.0142, 0.0509, 0.0963, 0.0878, 0.0000,
             -0.0878, -0.0963, -0.0509, -0.0142, -0.0019, -0.0001],
            [0.0000, 0.0007, 0.0071, 0.0374, 0.1126, 0.2119, 0.2605,
             0.2119, 0.1126, 0.0374, 0.0071, 0.0007, 0.0000]])

        self.kernels = [tap_3, tap_5, tap_7, tap_9, tap_11, tap_13]

        # sending them to device
        if device is not None:
            self.to_device(device)


    def to_device(self, device):
        """TODO DESCRIPTION

        params:
            device (str): TODO
        """
        self.kernels = [kernel.to(device) for kernel in self.kernels]


    def __call__(self, img, t_id):
        """TODO DESCRIPTION

        params:
            img (TODO): image with dimensions B x C x H x W
            t_id (int): tap radius (for example t_id=1 will use the tap 3)

        returns:
            (TODO): TODO
        """
        if t_id in [3, 5]:
            assert False, "Not Implemented"

        return self.forward(img, t_id)


    def forward(self, img, t_id=1):
        """TODO DESCRIPTION

        params:
            img (TODO): image with dimensions B x C x H x W
            t_id (int) optional: tap radius (for example t_id=1 will use the tap 3) (default 1)

        returns:
            (tuple): TODO
        """
        kernel = self.kernels[t_id-1]

        p = kernel[1 : 2, ...]
        d1 = kernel[0 : 1, ...]

        # B x C x H x W
        grad_x = kn_filters.filter2d_separable(
            img,
            p,
            d1,
            border_type='reflect',
            normalized=False,
            padding='same')
        grad_y = kn_filters.filter2d_separable(
            img,
            d1,
            p,
            border_type='reflect',
            normalized=False,
            padding='same')

        return (grad_x, grad_y)

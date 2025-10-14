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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from torchvision.utils import make_grid, save_image

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2, mask=None):
    if mask is None:
        mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    else:
        mask_bin = (mask == 1.)
        mse = (((img1 - img2)[mask_bin]) ** 2).mean()
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def turbo_cmap(gray: np.ndarray) -> np.ndarray:
    """
    Visualize a single-channel image using matplotlib's turbo color map
    yellow is high value, blue is low
    :param gray: np.ndarray, (H, W) or (H, W, 1) unscaled
    :return: (H, W, 3) float32 in [0, 1]
    """
    colored = plt.cm.turbo(plt.Normalize()(gray.squeeze()))[..., :-1]
    return colored.astype(np.float32)

def visualize_depth(depth, near=0.2, far=13):
    depth = depth[0].detach().cpu().numpy()
    colormap = matplotlib.colormaps['turbo']
    curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)
    eps = np.finfo(np.float32).eps
    near = near if near else depth.min()
    far = far if far else depth.max()
    near -= eps
    far += eps
    near, far, depth = [curve_fn(x) for x in [near, far, depth]]
    depth = np.nan_to_num(
        np.clip((depth - np.minimum(near, far)) / np.abs(far - near), 0, 1))
    vis = colormap(depth)[:, :, :3]

    out_depth = np.clip(np.nan_to_num(vis), 0., 1.)
    return torch.from_numpy(out_depth).float().cuda().permute(2, 0, 1)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def save_image_with_mask(image: torch.Tensor, mask: torch.Tensor, save_path: str):
    """
    显示并保存带掩码的图片（原图与掩码叠加）。
    :param image: [c, h, w] 的图片张量，像素值范围[0,1]或[0,255]
    :param mask: [1, h, w] 的掩码张量，值为0或1
    :param save_path: 保存路径
    """
    # 转为cpu和numpy
    img_np = image.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()

    # 处理图片
    if img_np.shape[0] == 1:
        img_np = np.repeat(img_np, 3, axis=0)
    if img_np.max() > 1.0:
        img_np = img_np / 255.0
    img_np = np.transpose(img_np, (1, 2, 0))  # [h, w, c]

    mask_np = mask_np.squeeze()
    # 创建掩码颜色层（红色，带透明度）
    color_mask = np.zeros_like(img_np)
    color_mask[..., 0] = 1.0  # 红色通道
    alpha = 0.5  # 透明度
    mask_bool = mask_np.astype(bool)
    img_np[mask_bool] = img_np[mask_bool] * (1 - alpha) + color_mask[mask_bool] * alpha

    plt.imsave(save_path, np.clip(img_np, 0, 1))

def save_image_grid_with_masks(images: torch.Tensor, masks: torch.Tensor, save_path: str):
    """
    将带掩码的图片保存为一个grid。
    :param images: [3, c, h, w] 的图片张量，像素值范围[0,1]
    :param masks: [3, 1, h, w] 的掩码张量，值为0或1
    :param save_path: 保存路径
    """
    grid_imgs = []
    for i in range(images.shape[0]):
        img = images[i]  # [c, h, w]
        mask = masks[i]  # [1, h, w]
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        mask = mask.squeeze(0)  # [h, w]
        color_mask = torch.zeros_like(img)
        color_mask[0] = 1.0  # 红色通道
        alpha = 0.5
        mask_bool = mask.bool()
        img_overlay = img.clone()
        img_overlay[:, mask_bool] = img[:, mask_bool] * (1 - alpha) + color_mask[:, mask_bool] * alpha
        grid_imgs.append(img_overlay)
    grid = make_grid(torch.stack(grid_imgs), nrow=3)
    save_image(grid, save_path)

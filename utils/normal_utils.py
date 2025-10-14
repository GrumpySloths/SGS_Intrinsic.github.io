import torch
import itertools


def world_normal_to_camera_normal(render_normal, c2w, H, W, sign_idx=0):
    """
    将世界坐标系下的法线转换到相机坐标系下，并根据sign_idx选择不同的sign排列。

    Args:
        render_normal (torch.Tensor): [3, H, W] 世界坐标系下的法线
        c2w (torch.Tensor): [4, 4] 世界到相机的变换矩阵
        H (int): 图像高度
        W (int): 图像宽度
        sign_idx (int): sign排列的索引，范围[0, 7]

    Returns:
        torch.Tensor: [3, H, W] 相机坐标系下的法线
    """

    # 生成所有x, y, z为1或-1的全排列
    # sign_permutations = list(itertools.product([1, -1], repeat=3))
    # if not (0 <= sign_idx < len(sign_permutations)):
    #     raise ValueError(f"sign_idx should be in [0, {len(sign_permutations)-1}]")
    # sign = torch.tensor(sign_permutations[sign_idx], device=render_normal.device, dtype=render_normal.dtype).view(3, 1, 1)
    # sign=torch.tensor([1,-1,-1], device=render_normal.device, dtype=render_normal.dtype).view(3, 1, 1)  # [-1, -1, -1] for GeoWizard
    # sign=torch.tensor([1,-1,-1], device=render_normal.device, dtype=render_normal.dtype).view(3, 1, 1)  # [-1, -1, -1] for rgbx
    sign=torch.tensor([-1,-1,-1], device=render_normal.device, dtype=render_normal.dtype).view(3, 1, 1)  # [-1, -1, -1] for stablenorm

    R = c2w[:3, :3]  # 世界到相机的旋转矩阵
    render_normal_flat = render_normal.reshape(3, -1).T  # [H*W, 3]
    render_normal_cam_flat = torch.matmul(render_normal_flat, R)  # [H*W, 3]
    render_normal_cam = render_normal_cam_flat.T.reshape(3, H, W)
    render_normal_cam = render_normal_cam * sign
    return render_normal_cam
import torch
from PIL import Image
import numpy as np

omnidata_dir="/home/jiahao/ipsm_relighting/pretrained_models/omnidata"
# Load surface normal estimation model,一个比较奇怪的疑问就是omnidata输出的法线其值范围是(0,1)
model_normal = torch.hub.load(omnidata_dir, 'surface_normal_dpt_hybrid_384',source='local')

# Load depth estimation model
model_depth = torch.hub.load(omnidata_dir, 'depth_dpt_hybrid_384',source='local')

# img_path="/home/jiahao/Relightable3DGaussian/reference_methods/IPSM/datasets/mipnerf360/bonsai/images_4/DSCF5566.JPG"
# img_path="/home/jiahao/ipsm_relighting/rgb2x/example/DSCF4671.JPG"
img_path="rgb2x/example/000_im_denoised.png"

import torchvision.transforms as T
from torchvision import transforms

totensor = transforms.ToTensor()

def transform_normal_cam(x):
  '''
     2D3DS space: +X right, +Y down, +Z from screen to me
     Pytorch3D space: +X left, +Y up, +Z from me to screen
  '''
  x2 = -(totensor(x) - 0.5) * 2.0
  x2[-1,...] *= -1
  # print(x2.shape)
  return x2

def transform_normal_colmap(x):
    '''
    Transform normals to COLMAP space:
    COLMAP: +X right, +Y up, +Z forward (from camera into scene)
    Input: PIL Image or numpy array, shape [H, W, 3], values in [0,1] or [0,255]
    Output: torch.Tensor, shape [3, H, W], values in [-1,1]
    '''
    # Convert to tensor and normalize to [-1, 1]
    x2 = -(totensor(x) - 0.5) * 2.0
    # COLMAP: [X, Y, Z] = [X, -Y, -Z] (from 2D3DS/PyTorch3D)
    x2[1, ...] *= -1  # invert Y
    x2[2, ...] *= -1  # invert Z
    return x2

# 将RGB值映射到[-1, 1]的法线空间
def rgb_to_normal(img):
    return img.astype(np.float32) / 127.5 - 1.0

def normal_to_rgb(normal):
    return np.clip((normal + 1.0) * 127.5, 0, 255).astype(np.uint8)

sign=np.array([-1,-1,-1])

# Load and preprocess image
img = Image.open(img_path).convert('RGB')
transform = T.Compose([
    T.ToTensor(),
])
img_tensor = transform(img).to(next(model_normal.parameters()).device)

# Resize to match model input if needed
h, w = img_tensor.shape[1:3]
norm_img = (img_tensor[None] - 0.5) / 0.5
norm_img = torch.nn.functional.interpolate(
    norm_img,
    size=(384, 512),
    mode="bicubic",
    align_corners=False
)

with torch.no_grad():
    pred_depth = model_depth(norm_img)
    pred_depth = torch.nn.functional.interpolate(
        pred_depth.unsqueeze(1),
        size=(h, w),
        mode="bicubic",
        align_corners=False
    ).squeeze().cpu().numpy()

    pred_normal = model_normal(norm_img)
    pred_normal = torch.nn.functional.interpolate(
        pred_normal,
        size=(h, w),
        mode="bicubic",
        align_corners=False
    # ).squeeze()
    ).squeeze().cpu().numpy()

print("Predicted Depth shape:", pred_depth.shape)
print("Predicted Normal shape:", pred_normal.shape)
# 保存原始法线图,omnidata 预测的法线范围是[0,1]，需要转换为[-1,1]的法线空间
# pred_normal= pred_normal * 2 - 1 
# pred_normal=torch.clamp((pred_normal + 1) / 2, 0, 1)
# pred_normal = pred_normal.cpu().numpy()
normal_img = (pred_normal.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(normal_img).save("pred_normal.png")
# 将normal_img映射回[-1,1]法线空间
normal_float = normal_img.astype(np.float32) / 127.5 - 1.0
# 乘以sign
normal_signed = normal_float * sign
# 映射回[0,255]颜色空间
normal_signed_img = np.clip((normal_signed + 1.0) * 127.5, 0, 255).astype(np.uint8)
Image.fromarray(normal_signed_img).save("pred_normal_signed.png")

# 利用transform_normal_cam变换法线并保存
pred_normal_img = Image.fromarray((pred_normal.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8))
transformed_normal = transform_normal_colmap(pred_normal_img)
# 变换后范围为[-1,1]，映射回[0,255]
transformed_normal_img = ((transformed_normal.permute(1, 2, 0).numpy() + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(transformed_normal_img).save("pred_normal_transformed_colmap.png")
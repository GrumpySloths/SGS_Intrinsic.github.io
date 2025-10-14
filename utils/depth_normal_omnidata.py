import torch
import numpy as np

omnidata_dir="/home/jiahao/ipsm_relighting/pretrained_models/omnidata"
# Load surface normal estimation model
omni_normal = torch.hub.load(omnidata_dir, 'surface_normal_dpt_hybrid_384',source='local')

# Load depth estimation model
omni_depth = torch.hub.load(omnidata_dir, 'depth_dpt_hybrid_384',source='local')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

omni_normal.to(device)
omni_depth.to(device)
omni_normal.eval()
omni_depth.eval()

for param in omni_normal.parameters():
    param.requires_grad = False
for param in omni_depth.parameters():
    param.requires_grad = False

downsampling = 1

def estimate_depth_normal_omni(img,mode='test'):

    h, w = img.shape[1:3]  #h:378,w:504 for fern image_8
    norm_img = (img[None] - 0.5) / 0.5
    norm_img = torch.nn.functional.interpolate(  #shape:[1,3,384,512]
        norm_img,
        size=(384, 512),
        mode="bicubic",
        align_corners=False)
    if mode == 'test':
        with torch.no_grad():
            prediction_depth = omni_depth(norm_img) #shape:[1,384,512]
            prediction_depth = torch.nn.functional.interpolate(
                prediction_depth.unsqueeze(1), #shape:[1,1,384,512]
                size=(h//downsampling, w//downsampling),
                mode="bicubic",
                align_corners=False,
            ).squeeze()  #这里squeeze函数直接把前面所有的1都去掉了，不管前面有几个1

            prediction_normal = omni_normal(norm_img) #shape:[1,3,384,512]
            prediction_normal = torch.nn.functional.interpolate(
                prediction_normal,
                size=(h//downsampling, w//downsampling),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
    else:
        raise NotImplementedError("The mode '{}' is not implemented.".format(mode))

    return prediction_depth,prediction_normal

def normal_to_geowizard(normal):
    """
    将法线从OmniData模型的输出格式转换为GeoWizard所使用的法线模型空间。
    
    Args:
        normal (torch.Tensor or np.ndarray): [3, H, W] OmniData模型输出的法线

    Returns:
        torch.Tensor or np.ndarray: [3, H, W] GeoWizard所使用的法线
    """

    if isinstance(normal, torch.Tensor):
        normal = normal * 2 - 1  # [0,1] -> [-1,1]
        # sign = torch.tensor([-1, -1, -1], device=normal.device, dtype=normal.dtype).view(3, 1, 1)
        sign = torch.tensor([1, 1, 1], device=normal.device, dtype=normal.dtype).view(3, 1, 1)
        return normal * sign
    elif isinstance(normal, np.ndarray):
        normal = normal * 2 - 1  # [0,1] -> [-1,1]
        # sign = np.array([-1, -1, -1], dtype=normal.dtype).reshape(3, 1, 1)
        sign = np.array([1, 1, 1], dtype=normal.dtype).reshape(3, 1, 1)
        return normal * sign
    else:
        raise TypeError("Input normal must be a torch.Tensor or np.ndarray")
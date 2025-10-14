import sys
sys.path.append("./diffusion_renderer/src")
sys.path.append("./diffusion_renderer/")
from diffusion_renderer.src.models.custom_unet_st import UNetCustomSpatioTemporalConditionModel
from diffusion_renderer.src.models.env_encoder import EnvEncoder
from diffusion_renderer.src.pipelines.pipeline_rgbx import RGBXVideoDiffusionPipeline
import torch
from omegaconf import OmegaConf
from utils.utils_env_proj import process_environment_map
from diffusion_renderer.src.data.rendering_utils import envmap_vec
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision import transforms
from diffusers import (
    AutoencoderKLTemporalDecoder,
    EulerDiscreteScheduler,
)
from torchvision.utils import save_image
cfg = OmegaConf.load("configs/xrgb_inference.yaml")
inference_height, inference_width = 512, 512
cond_mode = "env"
use_deterministic_mode = False 
weight_dtype = torch.float16
# model construction
missing_kwargs = {}
missing_kwargs["cond_mode"] = cond_mode
missing_kwargs["use_deterministic_mode"] = use_deterministic_mode
text_encoder, image_encoder, vae, env_encoder = None, None, None, None
tokenizer, feature_extractor = None, None
env_encoder = EnvEncoder.from_pretrained(cfg.inference_model_weights, subfolder="env_encoder")
vae = AutoencoderKLTemporalDecoder.from_pretrained(
    cfg.inference_model_weights, subfolder="vae"
)
unet = UNetCustomSpatioTemporalConditionModel.from_pretrained(
    cfg.inference_model_weights, subfolder="unet"
)
noise_scheduler = EulerDiscreteScheduler.from_pretrained(cfg.inference_model_weights, subfolder="scheduler")
for module in [text_encoder, image_encoder, vae, unet, env_encoder]:
    if module is not None:
        module.to("cuda", dtype=weight_dtype)
        
pipeline = RGBXVideoDiffusionPipeline(
    vae=vae,
    image_encoder=image_encoder,
    feature_extractor=feature_extractor,
    unet=unet,
    scheduler=noise_scheduler,
    env_encoder=env_encoder,
    scale_cond_latents=cfg.model_pipeline.get('scale_cond_latents', False),
    cond_mode=cond_mode
)
pipeline.scheduler.register_to_config(timestep_spacing="trailing")
pipeline.load_lora_weights(cfg.inference_model_weights, subfolder="lora", adapter_name="real-lora")
pipeline = pipeline.to("cuda")
pipeline = pipeline.to(weight_dtype)
# pipeline.enable_model_cpu_offload() # for further memory savings
pipeline.set_progress_bar_config(disable=True)

def xrgb_svd_generate_rgb(
    basecolor,      # Tensor of shape [C, H, W]
    normal,         # Tensor of shape [C, H, W]
    depth,          # Tensor of shape [C, H, W]
    roughness,      # Tensor of shape [C, H, W]
    metallic,       # Tensor of shape [C, H, W]
    envlight_path="datasets/Environment_Maps/gamma_corrected_resized.exr",  # Path to environment map
    seed=42
):
    """
    Generate RGB image from G-buffer inputs using XRGB-SVD pipeline
    
    Args:
        basecolor: Basecolor tensor of shape [C, H, W]
        normal: Normal tensor of shape [C, H, W]
        depth: Depth tensor of shape [C, H, W]
        roughness: Roughness tensor of shape [C, H, W]
        metallic: Metallic tensor of shape [C, H, W]
        envlight_path: Path to environment map
        seed: Random seed for generation
    
    Returns:
        PIL Image of the generated RGB result
    """
    
    # Record original dimensions
    original_height, original_width = basecolor.shape[1], basecolor.shape[2]
    
    # Resize all inputs to inference dimensions and add batch+temporal dimensions
    def prepare_input_tensor(tensor):
        resized = resize_upscale_without_padding(tensor, inference_height, inference_width)
        return resized.unsqueeze(0).unsqueeze(0)  # Add batch and temporal dims -> [1, 1, C, H, W]
    
    cond_images = {
        'basecolor': prepare_input_tensor(basecolor),
        'normal': prepare_input_tensor(normal),
        'depth': prepare_input_tensor(depth),
        'roughness': prepare_input_tensor(roughness),
        'metallic': prepare_input_tensor(metallic)
    }
    height,width= cond_images['basecolor'].shape[3], cond_images['basecolor'].shape[4]
    print("Prepared input tensors:", {k: v.shape for k, v in cond_images.items()})

    # Process environment map
    env_resolution = (512, 512)
    envlight_dict = process_environment_map(
        envlight_path,
        resolution=env_resolution,
        num_frames=1,  # Single frame
        fixed_pose=True,
        rotate_envlight=cfg.get('rotate_light', 0),
        elevation=cfg.get('cam_elevation', 0),
        env_format=['proj', 'fixed', 'ball'],
        device="cuda",
    )
    
    # Add environment conditions to cond_images
    cond_images['env_ldr'] = envlight_dict['env_ldr'].unsqueeze(0).permute(0, 1, 4, 2, 3)
    cond_images['env_log'] = envlight_dict['env_log'].unsqueeze(0).permute(0, 1, 4, 2, 3)
    env_nrm = envmap_vec(env_resolution, device="cuda") * 0.5 + 0.5
    cond_images['env_nrm'] = env_nrm.unsqueeze(0).unsqueeze(0).permute(0, 1, 4, 2, 3).expand_as(cond_images['env_ldr'])
    
    # Set up generator
    generator = torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None
    
    # Define condition labels (based on inference_svd_xrgb.py)
    cond_labels = cfg.model_pipeline.cond_images
    
    # Run inference with single frame
    with torch.autocast("cuda", enabled=True):
        result = pipeline(
            cond_images, cond_labels,
            height=height, width=width,
            num_frames=1,  # Single frame
            num_inference_steps=cfg.inference_n_steps,
            min_guidance_scale=cfg.inference_min_guidance_scale,
            max_guidance_scale=cfg.inference_max_guidance_scale,
            fps=cfg.get('fps', 7),
            motion_bucket_id=cfg.get('motion_bucket_id', 127),
            noise_aug_strength=cfg.get('cond_aug', 0),
            generator=generator,
            cross_attention_kwargs={'scale': cfg.get('lora_scale', 0.0)},
            dynamic_guidance=False,
            decode_chunk_size=cfg.get('decode_chunk_size', None),
            output_type="pt"
        ).frames[0]  # Get first batch
    
    # Get the single frame and resize back to original dimensions
    result_tensor = result[0]  # Extract first frame
    
    # Convert PIL to tensor if needed, then resize back
    if isinstance(result_tensor, Image.Image):
        result_tensor = transforms.ToTensor()(result_tensor).to("cuda")
    
    # Resize back to original dimensions
    resized_result = torch.nn.functional.interpolate(
        result_tensor.unsqueeze(0), 
        size=(original_height, original_width), 
        mode='bilinear', 
        align_corners=False
    ).squeeze(0)
    
    # print("Result tensor shape after resizing:", resized_result.shape)
    # Convert back to PIL Image
    # result_pil = transforms.ToPILImage()(resized_result.cpu())
    return resized_result


def resize_upscale_without_padding(image, target_height, target_width, mode='bilinear', divisible_by=int(64)):
    """
    Resizes and upscales an image or tensor without padding, ensuring dimensions are divisible by 16.

    Parameters:
        image (PIL.Image.Image or torch.Tensor): Input image to be resized. For torch.Tensor, shape should be (C, H, W).
        target_height (int): Desired height of the output image.
        target_width (int): Desired width of the output image.
        mode (str): Resampling mode. Options for PIL: 'nearest', 'bilinear', 'bicubic', 'lanczos'.
                    For torch.Tensor, options are 'nearest', 'bilinear', 'bicubic', 'trilinear', etc.

    Returns:
        PIL.Image.Image or torch.Tensor: Resized image with dimensions divisible by 16, in the same format as the input.
    """

    if isinstance(image, Image.Image):
        # PIL Image case
        original_width, original_height = image.size

    elif isinstance(image, torch.Tensor):
        if image.dim() != 3:
            raise ValueError("Tensor image should have 3 dimensions (C, H, W)")

        original_height, original_width = image.shape[1:3]
    else:
        raise TypeError("Input image must be a PIL.Image.Image or torch.Tensor")

    # Calculate scale and new dimensions
    scale = max(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Ensure dimensions are divisible by 8 (SD) or 64 (SVD)
    new_width = max(divisible_by, (new_width + (divisible_by - 1)) // divisible_by * divisible_by)
    new_height = max(divisible_by, (new_height + (divisible_by - 1)) // divisible_by * divisible_by)

    # Resize the image
    if isinstance(image, Image.Image):
        resized_image = image.resize((new_width, new_height), resample=getattr(Image, mode.upper(), Image.BILINEAR))
        return resized_image

    elif isinstance(image, torch.Tensor):
        # Resize the image
        resized_image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(new_height, new_width),
                                                        mode=mode, align_corners=False).squeeze(0)
        return resized_image
    else:
        raise TypeError("Input image must be a PIL.Image.Image or torch.Tensor")


if __name__ == "__main__":
    # Test example
    
    # Load test images or create dummy data
    H, W = 512, 512
    # Load and preprocess image
    basecolor_path="x2rgb/examples/0000.0000.basecolor.png"
    depth_path="x2rgb/examples/0000.0000.depth.png"
    normal_path="x2rgb/examples/0000.0000.normal.png"
    roughness_path="x2rgb/examples/0000.0000.roughness.png"
    metallic_path="x2rgb/examples/0000.0000.metallic.png"
    # image = Image.open(image_path)
    basecolor = Image.open(basecolor_path)
    depth = Image.open(depth_path)
    print(f"basecolor size: {basecolor.size}")
    print(f"depth size: {depth.size}")
    normal = Image.open(normal_path)
    roughness = Image.open(roughness_path)
    metallic = Image.open(metallic_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    basecolor = transform(basecolor).to("cuda")
    depth = transform(depth).to("cuda")
    normal = transform(normal).to("cuda")
    roughness = transform(roughness).to("cuda")
    metallic = transform(metallic).to("cuda")

    print(f"basecolor shape: {basecolor.shape}")
    print(f"depth shape: {depth.shape}")
    print(f"normal shape: {normal.shape}")
    print(f"roughness shape: {roughness.shape}")
    print(f"metallic shape: {metallic.shape}")
    envlight_path = "datasets/Environment_Maps/gamma_corrected_resized.exr"  # Replace with actual path

    try:
        result_image = xrgb_svd_generate_rgb(
            basecolor=basecolor,
            normal=normal,
            depth=depth,
            roughness=roughness,
            metallic=metallic,
            envlight_path=envlight_path,
            seed=cfg.get('seed', 42)
        )
        
        # Save result
        save_image(result_image, "xrgb_svd_result.png")
        # result_image.save("xrgb_svd_result.png")
        # print("RGB result saved to xrgb_svd_result.png")
        # print(f"Result image size: {result_image.size}")
        
    except Exception as e:
        print(f"Error during inference: {e}")
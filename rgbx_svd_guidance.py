from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from accelerate import Accelerator
import sys
sys.path.append("./diffusion_renderer/src")
sys.path.append("./diffusion_renderer/")
from diffusion_renderer.src.pipelines.pipeline_rgbx import RGBXVideoDiffusionPipeline
import torch
from torchvision import transforms
from PIL import Image
from omegaconf import OmegaConf
from torchvision.utils import save_image

# Global pipeline configuration
inference_height, inference_width = 512, 512
# inference_model_weights = "./pretrained_models/diffusion_renderer-inverse-svd"
inference_model_weights = "/home/jiahao/ipsm_relighting/pretrained_models/diffusion_renderer-inverse-svd"

# Initialize pipeline globally
missing_kwargs = {}
missing_kwargs["cond_mode"] = "skip" 
missing_kwargs["use_deterministic_mode"] = False
missing_kwargs["image_encoder"] = CLIPVisionModelWithProjection.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid", subfolder="image_encoder",
)
missing_kwargs["feature_extractor"] = CLIPImageProcessor.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid", subfolder="feature_extractor",
)

pipeline = RGBXVideoDiffusionPipeline.from_pretrained(inference_model_weights, **missing_kwargs)
pipeline = pipeline.to("cuda")
pipeline = pipeline.to(torch.float16)
pipeline.set_progress_bar_config(disable=True)

# Load config
cfg = OmegaConf.load("/home/jiahao/ipsm_relighting/configs/rgbx_inference.yaml")
    
def svd_estimate_aovs(
    photo,  # Input photo tensor of shape [C, H, W]
    model_passes=["basecolor", "roughness", "metallic"],
    seed=42
):
    """
    Estimate AOVs using SVD RGBX pipeline
    
    Args:
        photo: Input photo tensor of shape [C, H, W]
        model_passes: List of AOV passes to generate
        seed: Random seed for generation
    
    Returns:
        List of AOV tensors, each of shape [C, H, W]
    """
    # Record original dimensions
    original_height, original_width = photo.shape[1], photo.shape[2]
    # print("Original image tensor shape:", photo.shape) 
    # Resize image tensor to inference dimensions
    image_tensor=resize_upscale_without_padding(photo, inference_height, inference_width)
    height, width = image_tensor.shape[1], image_tensor.shape[2]
    # print("Resized image tensor shape:", image_tensor.shape) 
    # Add batch and temporal dimensions to get shape [1,1,c,h,w]
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    
    generator = torch.Generator(device="cuda").manual_seed(seed)
    autocast_ctx = torch.autocast("cuda", enabled=True)
    
    cond_images = {"rgb": image_tensor}
    cond_labels = {"rgb": "vae"}
    
    aov_results = []
    
    for model_pass in model_passes:
        cond_images["input_context"] = model_pass
        with autocast_ctx:
            inference_image_list = pipeline(
                cond_images, cond_labels,
                height=height, width=width,
                num_frames=cfg.inference_n_frames,
                num_inference_steps=cfg.inference_n_steps,
                min_guidance_scale=cfg.inference_min_guidance_scale,
                max_guidance_scale=cfg.inference_max_guidance_scale,
                fps=cfg.get('fps', 7),
                motion_bucket_id=cfg.get('motion_bucket_id', 127),
                noise_aug_strength=cfg.get('cond_aug', 0),
                generator=generator,
                decode_chunk_size=cfg.get('decode_chunk_size', None),
                output_type="pt",
            ).frames[0]
        
        # Resize the tensor back to original dimensions
        resized_tensor = torch.nn.functional.interpolate(
            inference_image_list[0].unsqueeze(0), 
            size=(original_height, original_width), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        if model_pass=="roughness" or model_pass=="metallic":
            resized_tensor = resized_tensor[:1, ...] 
        aov_results.append(resized_tensor)
    
    return aov_results


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
    # Example usage
    image_path = "datasets/interiorverse/scene_0/images/000_im_denoised.png"
    
    # Load and preprocess image
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    photo = transform(image).to("cuda")
    
    # Generate AOVs
    model_passes = ["basecolor", "roughness"]
    aov_results = svd_estimate_aovs(photo, model_passes=model_passes, seed=cfg.seed)
    
    # Save results
    save_image(aov_results[0], "base_color_0.png")
    save_image(aov_results[1], "roughness_0.png")

    print(f"Generated {len(aov_results)} AOV passes:")
    for i, pass_name in enumerate(model_passes):
        print(f"  {pass_name}: shape {aov_results[i].shape}")
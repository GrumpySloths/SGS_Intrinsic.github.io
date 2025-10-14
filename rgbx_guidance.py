from rgb2x.pipeline_rgb2x_myversion import StableDiffusionAOVMatEstPipeline
import torch
from diffusers import DDIMScheduler
import torchvision
from rgb2x.load_image import load_ldr_image
from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt

# Load pipeline
pipe = StableDiffusionAOVMatEstPipeline.from_pretrained(
    "zheng95z/rgb-to-x",
    torch_dtype=torch.float16,
    cache_dir="/home/jiahao/ipsm_relighting/rgb2x/model_cache"
    # cache_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache"),
).to("cuda")

pipe.text_encoder.eval()
pipe.vae.eval()
pipe.unet.eval()

for p in pipe.text_encoder.parameters():
    p.requires_grad_(False)
for p in pipe.vae.parameters():
    p.requires_grad_(False)
for p in pipe.unet.parameters():
    p.requires_grad_(False)

pipe.scheduler = DDIMScheduler.from_config(
    pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
)
pipe.set_progress_bar_config(disable=True)
pipe.to("cuda")

def sd_estimate_aovs_normal(
    photo, # Input photo tensor of shape [C, H, W]
    inference_step=50,
):
    seed=42
    # Resize to multiples of 8 and max_side
    old_height, old_width = photo.shape[1], photo.shape[2]
    new_height, new_width = (512,512)

    photo = torchvision.transforms.Resize((new_height, new_width))(photo)

    prompts = {
            # "albedo": "Albedo (diffuse basecolor)"
        "normal": "Camera-space Normal",
    }

    aov_name="normal"
    prompt=prompts[aov_name]
    # Estimate AOVs
    generator = torch.Generator(device="cuda").manual_seed(seed)
    aov_results = pipe(
        photo=photo,
        prompt=prompt,
        num_inference_steps=inference_step,
        generator=generator,
        required_aovs=[aov_name],
        height=new_height,
        width=new_width,
    )  #aov_results.shape:[1,3,512,512]
    # Normalize to [0, 1]
    # aov_results=(aov_results/2+0.5).clamp(0,1)  # Convert from [-1, 1] to [0, 1]

    # torchvision.utils.save_image(aov_results, "test_rgb2x/aov_result_linear.png")
    # aov_results = aov_results.pow(1.0/2.2)
    # Interpolate aov_results back to original shape
    aov_results = torch.nn.functional.interpolate(
        aov_results, size=(old_height, old_width), mode="bilinear", align_corners=False
    ).squeeze(0)  #[3, H, W]
    
    return aov_results

def sd_estimate_aovs(
    photo, # Input photo tensor of shape [C, H, W]
    inference_step=50,
):
    seed=42
    # Resize to multiples of 8 and max_side
    old_height, old_width = photo.shape[1], photo.shape[2]
    new_height, new_width = (512,512)

    photo = torchvision.transforms.Resize((new_height, new_width))(photo)

    prompts = {
            "albedo": "Albedo (diffuse basecolor)"
    }

    aov_name="albedo"
    prompt=prompts[aov_name]
    # Estimate AOVs
    generator = torch.Generator(device="cuda").manual_seed(seed)
    aov_results = pipe(
        photo=photo,
        prompt=prompt,
        num_inference_steps=inference_step,
        generator=generator,
        required_aovs=[aov_name],
        height=new_height,
        width=new_width,
    )  #aov_results.shape:[1,3,512,512]
    # Normalize to [0, 1]
    aov_results=(aov_results/2+0.5).clamp(0,1)  # Convert from [-1, 1] to [0, 1]

    # torchvision.utils.save_image(aov_results, "test_rgb2x/aov_result_linear.png")
    aov_results = aov_results.pow(1.0/2.2)
    # Interpolate aov_results back to original shape
    aov_results = torch.nn.functional.interpolate(
        aov_results, size=(old_height, old_width), mode="bilinear", align_corners=False
    ).squeeze(0)  #[3, H, W]
    
    return aov_results


def sd_estimate_aovs_batch(
    photos,  # Input photo tensor of shape [B, C, H, W]
    inference_step=50,
):
    seed = 42
    batch_size, c, old_height, old_width = photos.shape
    new_height, new_width = (512, 512)

    # Resize each image in the batch
    photos_resized = torch.nn.functional.interpolate(
        photos, size=(new_height, new_width), mode="bilinear", align_corners=False
    )

    prompts = {
        "albedo": "Albedo (diffuse basecolor)"
    }
    aov_name = "albedo"
    prompt = prompts[aov_name]

    generator = torch.Generator(device="cuda").manual_seed(seed)
    aov_results = pipe(
        photo=photos_resized,
        prompt=[prompt] * batch_size,
        num_inference_steps=inference_step,
        generator=generator,
        required_aovs=[aov_name],
        height=new_height,
        width=new_width,
    )  # [B, 3, 512, 512]

    # Normalize to [0, 1]
    aov_results = (aov_results / 2 + 0.5).clamp(0, 1)
    aov_results = aov_results.pow(1.0 / 2.2)

    # Interpolate back to original shape
    aov_results = torch.nn.functional.interpolate(
        aov_results, size=(old_height, old_width), mode="bilinear", align_corners=False
    )  # [B, 3, H, W]

    return aov_results

if __name__ == "__main__":

    # Example input image path (you can change this to your own image)
    input_paths = [
        "rgb2x/example/DSCF5568_bonsai.jpg",
        "rgb2x/example/DSCF4671.JPG",
        "rgb2x/example/000_im_denoised.png"
    ]

    # Load images as tensors and stack into a batch
    photos = [load_ldr_image(p, from_srgb=True).to("cuda") for p in input_paths]
    target_shape = photos[0].shape[-2:]  # (H, W) of the first image
    # photos = [torch.nn.functional.interpolate(p.unsqueeze(0), size=target_shape, mode="bilinear", align_corners=False).squeeze(0) for p in photos]
    # photos = torch.stack(photos, dim=0)  # [B, C, H, W]

    # Run batch AOV estimation
    # aov_results = sd_estimate_aovs_batch(photos, inference_step=50)  # [B, 3, H, W]
    aov_result= sd_estimate_aovs(photos[2], inference_step=50).cpu().numpy()  # [3, H, W]
    print(f"AOV result shape: {aov_result.shape}")
    normal_visual = ((aov_result + 1) * 0.5 * 255).astype(np.uint8)
    normal_visual = np.transpose(normal_visual, (1, 2, 0))
    normal_png_filename = "./test_rgb2x/normal_result.png" 
    # plt.imsave(depth_png_filename, depth_visual, cmap='viridis')
    plt.imsave(normal_png_filename, normal_visual)
    # # Check tensor dtypes
    # for i, aov_tensor in enumerate(aov_results):
    #     print(f"AOV tensor {i} dtype: {aov_tensor.dtype}")

    # # Convert results to PIL and save
    # os.makedirs("test_rgb2x", exist_ok=True)
    # for i, aov_tensor in enumerate(aov_results):
    #     albedo = aov_tensor.permute(1, 2, 0).cpu().numpy()  # HWC
    #     albedo = (albedo * 255).clip(0, 255).astype("uint8")
    #     albedo_pil = Image.fromarray(albedo)
    #     output_path = f"test_rgb2x/albedo_result_{i}.png"
    #     albedo_pil.save(output_path)
    #     print(f"Albedo result saved to {output_path}")


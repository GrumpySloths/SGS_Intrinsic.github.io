from x2rgb.pipeline_x2rgb_myversion import StableDiffusionAOVDropoutPipeline
from diffusers import DDIMScheduler
import torch
import torchvision
from x2rgb.load_image import load_exr_image, load_ldr_image
from PIL import Image
import os

# filepath: /home/jiahao/Relightable3DGaussian/reference_methods/IPSM/x2rgb_guidance.py

# Load pipeline
pipe = StableDiffusionAOVDropoutPipeline.from_pretrained(
    "zheng95z/x-to-rgb",
    torch_dtype=torch.float16, 
    cache_dir="/home/jiahao/ipsm_relighting/x2rgb/model_cache"
    # cache_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache"),
).to("cuda")

pipe.scheduler = DDIMScheduler.from_config(
    pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
)
pipe.set_progress_bar_config(disable=True)
pipe.to("cuda")

def load_input(path, exr_kwargs=None, ldr_kwargs=None):
    if path is None or path == "None":
        return None
    if path.endswith(".exr"):
        return load_exr_image(path, **(exr_kwargs or {})).to("cuda")
    elif path.lower().endswith((".png", ".jpg", ".jpeg")):
        return load_ldr_image(path, **(ldr_kwargs or {})).to("cuda")
    else:
        print(f"Unsupported file type: {path}")
        return None


def sd_generate_rgb(
    albedo=None, #albedo 范围是[0,1],from srgb to linear rgb
    normal=None, #normal 是在法线空间的，即范围为[-1, 1],且经过normalization
    roughness=None, #clamped to [0, 1]
    metallic=None,  #clamped to [0, 1]
    irradiance=None,
    prompt="living room",
    seed=42,
    inference_step=50,
    guidance_scale=7.5,
    image_guidance_scale=1.5,
    max_side= 512
):

    # print("albedo shape:", albedo.shape if albedo is not None else "None")  #[3,519,779]
    # print("normal shape:", normal.shape if normal is not None else "None")  #[3,519,779]
    # print("roughness shape:", roughness.shape if roughness is not None else "None") #[3,519,779]
    # print("metallic shape:", metallic.shape if metallic is not None else "None")  #[3,519,779]
    # print("irradiance shape:", irradiance.shape if irradiance is not None else "None")
    
    # Record original shapes
    orig_shapes = {}
    for name, aov in zip(
        ["albedo", "normal", "roughness", "metallic", "irradiance"],
        [albedo, normal, roughness, metallic, irradiance]
    ):
        if aov is not None:
            orig_shapes[name] = aov.shape[-2:]

    # Resize all AOVs to (max_side, max_side)
    height = width = max_side

    def resize_aov(aov, height=height, width=width):
        if aov is None:
            return None
        if aov.shape[-2] == height and aov.shape[-1] == width:
            return aov
        return torch.nn.functional.interpolate(
            aov.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
        ).squeeze(0)

    albedo = resize_aov(albedo, height, width)
    normal = resize_aov(normal, height, width)
    roughness = resize_aov(roughness, height, width)
    metallic = resize_aov(metallic, height, width)
    irradiance = resize_aov(irradiance, height, width)


    required_aovs = ["albedo", "normal", "roughness", "metallic", "irradiance"]
    generator = torch.Generator(device="cuda").manual_seed(seed)

    result = pipe(
        prompt=prompt,
        albedo=albedo,
        normal=normal,
        roughness=roughness,
        metallic=metallic,
        irradiance=irradiance,
        num_inference_steps=inference_step,
        height=height,
        width=width,
        generator=generator,
        required_aovs=required_aovs,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        guidance_rescale=0.7,
        output_type="pt",
    )  #shape:[1,3,512,512]

    print("Result shape:", result.shape)
    
    # Normalize output to [0,1]
    result = (result / 2 + 0.5).clamp(0, 1)
    # Tonemap (gamma correction)
    result = result.pow(1.0 / 2.2)

    # Resize back to original shape if available
    if "albedo" in orig_shapes:
        orig_h, orig_w = orig_shapes["albedo"]
        result = torch.nn.functional.interpolate(
            result, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        ).squeeze(0)  # shape: [3, orig_h, orig_w]

    return result
    # img = (result.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype("uint8")
    # img = Image.fromarray(img)
    # out_path = os.path.join(output_dir, "x2rgb_result.png")
    # img.save(out_path)
    # print(f"RGB result saved to {out_path}")

if __name__ == "__main__":
    # Example input paths (change as needed)
    albedo_path = "rgb2x/test_rgb2x/gt_room_albedo_0.png"
    normal_path = "rgb2x/test_rgb2x/gt_room_normal_0.png"
    roughness_path = "rgb2x/test_rgb2x/gt_room_roughness_0.png"
    metallic_path = "rgb2x/test_rgb2x/gt_room_metallic_0.png"
    irradiance_path = "rgb2x/test_rgb2x/gt_room_irradiance_0.png"
    prompt = "living room"
    output_dir = "test_x2rgb"
    seed = 42
    inference_step = 50
    guidance_scale = 7.5
    image_guidance_scale = 1.5
    max_side = 512 

    # 加载AOV输入，参考run_x2rgb_test.py的load_input用法
    albedo = load_input(albedo_path, exr_kwargs={"clamp": True}, ldr_kwargs={"from_srgb": True})
    normal = load_input(normal_path, exr_kwargs={"normalize": True}, ldr_kwargs={"normalize": True})
    roughness = load_input(roughness_path, exr_kwargs={"clamp": True}, ldr_kwargs={"clamp": True})
    metallic = load_input(metallic_path, exr_kwargs={"clamp": True}, ldr_kwargs={"clamp": True})
    irradiance = load_input(irradiance_path, exr_kwargs={"tonemaping": True, "clamp": True}, ldr_kwargs={"from_srgb": True, "clamp": True})

    result = sd_generate_rgb(
        albedo=albedo,
        normal=normal,
        roughness=roughness,
        metallic=metallic,
        irradiance=irradiance,
        prompt=prompt,
        seed=seed,
        inference_step=inference_step,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        max_side=max_side
    )

    img = (result.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype("uint8")
    img = Image.fromarray(img)
    out_path = os.path.join(output_dir, "x2rgb_result.png")
    img.save(out_path)
    print(f"RGB result saved to {out_path}")
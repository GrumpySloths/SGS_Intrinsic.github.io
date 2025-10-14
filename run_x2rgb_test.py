from x2rgb.pipeline_x2rgb import StableDiffusionAOVDropoutPipeline
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
    cache_dir="/home/jiahao/Relightable3DGaussian/reference_methods/IPSM/x2rgb/model_cache"
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
    albedo_path=None,
    normal_path=None,
    roughness_path=None,
    metallic_path=None,
    irradiance_path=None,
    prompt="",
    output_dir="output",
    seed=42,
    inference_step=50,
    guidance_scale=7.5,
    image_guidance_scale=1.5,
    max_side=768
):
    os.makedirs(output_dir, exist_ok=True)
    # Load AOVs
    albedo = load_input(albedo_path, exr_kwargs={"clamp": True}, ldr_kwargs={"from_srgb": True})
    normal = load_input(normal_path, exr_kwargs={"normalize": True}, ldr_kwargs={"normalize": True})
    roughness = load_input(roughness_path, exr_kwargs={"clamp": True}, ldr_kwargs={"clamp": True})
    metallic = load_input(metallic_path, exr_kwargs={"clamp": True}, ldr_kwargs={"clamp": True})
    irradiance = load_input(irradiance_path, exr_kwargs={"tonemaping": True, "clamp": True}, ldr_kwargs={"from_srgb": True, "clamp": True})

    # Set output size
    height = width = max_side

    def resize_aov(aov, height=height, width=width):
        if aov is None:
            return None
        if aov.shape[1] == height and aov.shape[2] == width:
            return aov
        aov = torch.nn.functional.interpolate(
            aov.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
        ).squeeze(0)
        return aov

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
        output_type="np",
    ).images[0]

    img = (result * 255).clip(0, 255).astype("uint8")
    img = Image.fromarray(img)
    out_path = os.path.join(output_dir, "x2rgb_result.png")
    img.save(out_path)
    print(f"RGB result saved to {out_path}")

if __name__ == "__main__":
    # Example input paths (change as needed)
    albedo_path = "/home/jiahao/rgbx/rgb2x/test/DSCF4671_albedo_0.png"
    normal_path = "/home/jiahao/rgbx/rgb2x/test/DSCF4671_normal_0.png"
    roughness_path = "/home/jiahao/rgbx/rgb2x/test/DSCF4671_roughness_0.png"
    metallic_path = "/home/jiahao/rgbx/rgb2x/test/DSCF4671_metallic_0.png"
    irradiance_path = "None"
    prompt = "living room"
    output_dir = "test_x2rgb"
    seed = 42
    inference_step = 50
    guidance_scale = 7.5
    image_guidance_scale = 1.5
    max_side = 768

    sd_generate_rgb(
        albedo_path=albedo_path,
        normal_path=normal_path,
        roughness_path=roughness_path,
        metallic_path=metallic_path,
        irradiance_path=irradiance_path,
        prompt=prompt,
        output_dir=output_dir,
        seed=seed,
        inference_step=inference_step,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        max_side=max_side
    )
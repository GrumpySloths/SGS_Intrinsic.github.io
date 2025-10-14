import os
import sys
import torch
import torchvision
from diffusers import DDIMScheduler
from rgb2x.load_image import load_exr_image, load_ldr_image
from rgb2x.pipeline_rgb2x import StableDiffusionAOVMatEstPipeline
from PIL import Image

def main(
    input_path,
    output_dir,
    seed=42,
    inference_step=50,
    num_samples=1,
    max_side=1000
):
    os.makedirs(output_dir, exist_ok=True)
    torch.set_grad_enabled(False)

    # Load pipeline
    pipe = StableDiffusionAOVMatEstPipeline.from_pretrained(
        "zheng95z/rgb-to-x",
        torch_dtype=torch.float16,
        cache_dir="/home/jiahao/Relightable3DGaussian/reference_methods/IPSM/rgb2x/model_cache"
        # cache_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache"),
    ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda")

    # Load image,在加载图片的时候就已经将其进行归一化并将其变为tensor了
    if input_path.endswith(".exr"):
        photo = load_exr_image(input_path, tonemaping=True, clamp=True).to("cuda")
    elif input_path.lower().endswith((".png", ".jpg", ".jpeg")):
        photo = load_ldr_image(input_path, from_srgb=True).to("cuda")
    else:
        print("Unsupported file type.")
        return

    # Resize to multiples of 8 and max_side
    old_height, old_width = photo.shape[1], photo.shape[2]
    print(f"Original size: {old_height}x{old_width}")
    radio = old_height / old_width
    if old_height > old_width:
        new_height = max_side
        new_width = int(new_height / radio)
    else:
        new_width = max_side
        new_height = int(new_width * radio)
    new_width = new_width // 8 * 8
    new_height = new_height // 8 * 8
    print(f"Resized to: {new_height}x{new_width}")
    photo = torchvision.transforms.Resize((new_height, new_width))(photo)

    required_aovs = ["albedo", "normal", "roughness", "metallic", "irradiance"]
    prompts = {
        "albedo": "Albedo (diffuse basecolor)",
        "normal": "Camera-space Normal",
        "roughness": "Roughness",
        "metallic": "Metallicness",
        "irradiance": "Irradiance (diffuse lighting)",
    }

    generator = torch.Generator(device="cuda").manual_seed(seed)
    for i in range(num_samples):
        for aov_name in required_aovs:
            prompt = prompts[aov_name]
            result = pipe(
                prompt=prompt,
                photo=photo,
                num_inference_steps=inference_step,
                height=new_height,
                width=new_width,
                generator=generator,
                required_aovs=[aov_name],
            ).images[0][0]  # result is a list of PIL Images，对于pytorch来说可能后续需要转换为tensor
            # Resize back to original size
            result = torchvision.transforms.Resize((old_height, old_width))(result)
            # Convert to PIL and save
            img = result  # result is already a PIL Image
            out_path = os.path.join(
                output_dir, f"{os.path.splitext(os.path.basename(input_path))[0]}_{aov_name}_{i}.png"
            )
            img.save(out_path)
            print(f"Saved: {out_path}")

if __name__ == "__main__":
    # input_path = sys.argv[1] if len(sys.argv) > 1 else "/home/jiahao/rgbx/rgb2x/example/Castlereagh_corridor_photo.png"
    input_path = sys.argv[1] if len(sys.argv) > 1 else "rgb2x/example/DSCF5568_bonsai.jpg"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "test_rgb2x"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    inference_step = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    num_samples = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    main(input_path, output_dir, seed, inference_step, num_samples)

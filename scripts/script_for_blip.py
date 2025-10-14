import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import torch
import json

processor = BlipProcessor.from_pretrained("./pretrained_models/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("./pretrained_models/blip-image-captioning-base").to("cuda")

print("打印processor架构信息")
print(processor)
print("打印model架构信息")
print(model)

torch.manual_seed(42)
n_views = 24
dataset_dir = './datasets/mipnerf360'
scene_list = sorted(os.listdir(dataset_dir))
for scene_name in scene_list:
    scene_dir = os.path.join(dataset_dir, scene_name)
    n_views_path = f'{str(n_views)}_views'
    n_views_dir = os.path.join(scene_dir, n_views_path, 'images')
            
    img_list = sorted(os.listdir(n_views_dir))
    random_img_idx = torch.randint(low=0, high=n_views, size=(1, ))[0]
    print(scene_name, random_img_idx, img_list[random_img_idx])
    img_dir = os.path.join(n_views_dir, img_list[random_img_idx])
    raw_image = Image.open(img_dir).convert('RGB')
    
    text = "a photography of"
    inputs = processor(raw_image, text, return_tensors="pt").to("cuda")
    out = model.generate(**inputs)
    blip_rst = processor.decode(out[0], skip_special_tokens=True)
    img_name = img_list[random_img_idx].split('.')[0]
    blip_rst_name = 'blip_rst.txt'
    blip_rst_dir = os.path.join(scene_dir, blip_rst_name)
    with open(blip_rst_dir, 'w') as f:
        writing_content = f'random select {img_name} blip result:{blip_rst}'
        f.write(writing_content)
        print(writing_content)
        f.close()

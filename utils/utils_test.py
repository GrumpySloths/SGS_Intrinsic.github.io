import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Load the image
image_path = "../debug_images/iter_3500/render_image.png"
image = Image.open(image_path).convert("RGB")

# Transform the image to a tensor
transform = transforms.ToTensor()
image_tensor = transform(image)

midas_dir = '../pretrained_models/midas'
# midas_dir = './pretrained_models/MiDaS'

midas = torch.hub.load(midas_dir, "DPT_Hybrid", source='local')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
for param in midas.parameters():
    param.requires_grad = False

midas_transforms = torch.hub.load(midas_dir, "transforms", source='local')
transform = midas_transforms.dpt_transform
downsampling = 1

def estimate_depth(img, mode='test'):
    h, w = img.shape[1:3]  #h:378,w:504 for fern image_8
    norm_img = (img[None] - 0.5) / 0.5
    norm_img = torch.nn.functional.interpolate(
        norm_img,
        size=(384, 512),
        mode="bicubic",
        align_corners=False)

    if mode == 'test':
        with torch.no_grad():
            prediction = midas(norm_img)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h//downsampling, w//downsampling),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
    else:
        prediction = midas(norm_img)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h//downsampling, w//downsampling),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction

image_tensor = image_tensor.to(device)
image_tensor.requires_grad = True

# Estimate depth
for i in range(1000):
    print("current iter: ", i)
    depth = estimate_depth(image_tensor, mode='train')
    
    # Create a tensor of ones with the same shape as depth
    target = torch.ones_like(depth, device=device)
    
    # Calculate mean squared error
    mse_loss = torch.nn.functional.mse_loss(depth, target)
    
    # Backpropagate the loss
    mse_loss.backward()
    
tensor = depth.cpu().detach()
tensor_normalized = (tensor - tensor.min()) / (tensor.max() - tensor.min())
transform = transforms.ToPILImage()
image = transform(tensor_normalized)
image.save("./test_depth.png")
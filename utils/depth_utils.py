import torch

midas_dir = './pretrained_models/midas'

midas = torch.hub.load(midas_dir, "DPT_Hybrid", source='local')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
for param in midas.parameters():
    param.requires_grad = False

midas_transforms = torch.hub.load(midas_dir, "transforms", source='local')
transform = midas_transforms.dpt_transform
downsampling = 1

def estimate_depth(img, mode='test'):  # Here, img should be a tensor.
    h, w = img.shape[1:3]  #h:378,w:504 for fern image_8
    norm_img = (img[None] - 0.5) / 0.5
    norm_img = torch.nn.functional.interpolate(  #shape:[1,3,384,512]
        norm_img,
        size=(384, 512),
        mode="bicubic",
        align_corners=False)

    if mode == 'test':
        with torch.no_grad():
            prediction = midas(norm_img) #shape:[1,384,512]
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), #shape:[1,1,384,512]
                size=(h//downsampling, w//downsampling),
                mode="bicubic",
                align_corners=False,
            ).squeeze()  # squeeze removes all leading singleton dimensions.
    else:
        prediction = midas(norm_img)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h//downsampling, w//downsampling),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction #shape:[378,504]


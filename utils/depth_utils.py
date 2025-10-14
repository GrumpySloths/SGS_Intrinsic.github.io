import torch

midas_dir = './pretrained_models/midas'
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

def estimate_depth(img, mode='test'):  #这里的img应该是一个tensor
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
            ).squeeze()  #这里squeeze函数直接把前面所有的1都去掉了，不管前面有几个1
    else:
        prediction = midas(norm_img)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h//downsampling, w//downsampling),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction #shape:[378,504]


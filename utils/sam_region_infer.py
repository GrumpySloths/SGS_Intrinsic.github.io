import os
import numpy as np
import torch
from PIL import Image
import re
from pathlib import Path
from sam2.build_sam import build_sam2_video_predictor
import matplotlib.pyplot as plt
from utils.image_utils import save_image_with_mask

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
sam2_checkpoint = "/home/jiahao/ipsm_relighting_v2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
# Set requires_grad=False for all parameters in predictor
for param in predictor.parameters():
    param.requires_grad = False

def extract_number(filename):
    match = re.search(r'\d+', filename.stem)
    return int(match.group()) if match else float('inf')

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_video_masks(video_dir, save_dir="./video_infer_results"):

    ensure_dir(save_dir)

    # Initialize the inference state.
    inference_state = predictor.init_state(video_path=video_dir)
    h,w=inference_state["video_height"],inference_state["video_width"]
    # print(f"Processing video with resolution: {w}x{h}")

    # Example: segment and track one object (can be parameterized).
    predictor.reset_state(inference_state)
    ann_frame_idx = 0
    ann_obj_id = 1
    # points = np.array([[210, 350]], dtype=np.float32)
    points = np.array([[np.random.randint(0, w), np.random.randint(0, h)]], dtype=np.float32)
    # print(f"Using point: {points[0]} for annotation")

    labels = np.array([1], np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    # Propagate the prompts to get the masklet across the video.
    masks = []
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        # Assuming out_mask_logits is a numpy array or torch tensor
        mask=(out_mask_logits[0]>0)
        masks.append(mask)
    # Convert masks to a tensor and save them.
    masks=torch.stack(masks,dim=0)  #shape:[3,1,480,640]

    return masks  # List of mask arrays, len(masks) == number of frames


if __name__ == "__main__":
    # Example usage:
    img_path = "room/000_im_denoised.png"
    img = Image.open(img_path).convert("RGB")
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # [C, H, W]
    for i in range(15):
        masks = get_video_masks("./room")
        print(f"Iteration {i}: {len(masks)}")
        print(f"Iteration {i}: masks.shape: {masks.shape}")
        save_path = f"masked_000_im_denoised_iter_{i}.png"
        save_image_with_mask(img_tensor, masks[0], save_path)

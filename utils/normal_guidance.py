import torch
from PIL import Image
import numpy as np
import sys
import os


predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal_turbo", trust_repo=True)

def get_normal_tensor(input_tensor):
    # input_tensor: [3, h, w], float or uint8, range [0, 1] or [0, 255]
    # Convert to PIL Image
    if isinstance(input_tensor, torch.Tensor):
        input_tensor = input_tensor.cpu().numpy()
    if input_tensor.dtype == np.float32 or input_tensor.dtype == np.float64:
        input_tensor = (input_tensor * 255).astype(np.uint8)
    input_tensor = np.clip(input_tensor, 0, 255).astype(np.uint8)
    input_tensor = np.transpose(input_tensor, (1, 2, 0))  # [h, w, 3]
    input_tensor = Image.fromarray(input_tensor)
    # Apply the model to the image
    normal_image = predictor(input_tensor)  # This is a PIL Image

    # Convert PIL Image to tensor and ensure shape [3, h, w]
    normal_tensor = torch.from_numpy(np.array(normal_image)).permute(2, 0, 1).float() / 255.0
    # Normalize to [-1, 1]
    normal_tensor = normal_tensor * 2 - 1

    return normal_tensor  # shape: [3, h, w]


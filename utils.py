import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For reproducibility (slower):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_images_in_folders(root):
    total = 0
    for sub, _, files in os.walk(root):
        total += sum(1 for f in files if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp")))
    return total

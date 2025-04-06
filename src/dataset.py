import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch


class MRISegDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None):
        self.img_files = sorted(os.listdir(img_dir))
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = np.array(Image.open(img_path), dtype=np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # [1,H,W]

        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, self.img_files[idx])
            mask = np.array(Image.open(mask_path), dtype=np.float32) / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)
        else:
            mask = torch.zeros_like(img)
        return img, mask

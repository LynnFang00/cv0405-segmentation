import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

class MRISegDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))

        self.transform = T.Compose([
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        image = self.transform(image)
        mask = self.transform(mask)

        return image, mask

def preprocess_image(img_path):
    transform = T.Compose([
        T.Grayscale(),  # Ensure single channel
        T.ToTensor(),
    ])
    image = Image.open(img_path).convert("L")
    return transform(image)

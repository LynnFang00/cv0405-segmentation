import os
import torch
from PIL import Image
import numpy as np
from model import get_model
from dataset import preprocess_image
from tqdm import tqdm

def save_predictions(model, test_img_dir, output_dir, device):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for fname in tqdm(sorted(os.listdir(test_img_dir)), desc='Inference'):
            img_path = os.path.join(test_img_dir, fname)
            img = preprocess_image(img_path).to(device)

            pred = model(img.unsqueeze(0))
            pred = torch.sigmoid(pred).cpu().squeeze().numpy()
            pred_mask = (pred > 0.5).astype(np.uint8) * 255

            save_path = os.path.join(output_dir, fname)
            Image.fromarray(pred_mask).save(save_path)

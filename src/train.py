import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import MRISegDataset
from model import get_model
from PIL import Image

# ─── Paths ─────────────────────────────────────────────────────
PROJECT_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
train_img_dir   = os.path.join(PROJECT_ROOT, 'data', 'train', 'images')
train_mask_dir  = os.path.join(PROJECT_ROOT, 'data', 'train', 'masks')
test_img_dir    = os.path.join(PROJECT_ROOT, 'data', 'test',  'images')
output_dir      = os.path.join(PROJECT_ROOT, 'outputs')
os.makedirs(output_dir, exist_ok=True)

# ─── Hyperparameters ────────────────────────────────────────────
num_epochs   = 20
batch_size   = 8
lr           = 1e-3
patience     = 3   # early‑stop patience
weight_decay = 1e-4

# ─── Argparse ───────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Train or test U-Net segmentation model')
parser.add_argument('--mode', choices=['train', 'test'], default='train',
                    help='Mode: "train" to train model, "test" to run inference')
args = parser.parse_args()

# ─── Device ─────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── Losses & Metrics ───────────────────────────────────────────
bce_loss = nn.BCEWithLogitsLoss()

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, logits, targets):
        pred = torch.sigmoid(logits)
        inter = (pred * targets).sum(dim=(1,2,3))
        union = pred.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
        dice = (2 * inter + self.eps) / (union + self.eps)
        return 1 - dice.mean()

dice_loss = DiceLoss()

def dice_coeff(logits, targets, eps=1e-6):
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * targets).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    return ((2 * inter + eps) / (union + eps)).mean().item()

# ─── Training Function ──────────────────────────────────────────
def train():
    # Data split
    full_ds = MRISegDataset(train_img_dir, train_mask_dir)
    n_val   = int(len(full_ds) * 0.2)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # Model, optimizer, scheduler
    model     = get_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )

    best_val_dice = 0.0
    wait = 0

    for epoch in range(1, num_epochs+1):
        t0 = time.time()
        # Train
        model.train()
        train_loss, train_dice = 0.0, 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss   = 0.5 * bce_loss(logits, masks) + 0.5 * dice_loss(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            train_dice += dice_coeff(logits, masks) * imgs.size(0)
        train_loss /= n_train
        train_dice /= n_train

        # Validate
        model.eval()
        val_loss, val_dice = 0.0, 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss   = 0.5 * bce_loss(logits, masks) + 0.5 * dice_loss(logits, masks)
                val_loss += loss.item() * imgs.size(0)
                val_dice += dice_coeff(logits, masks) * imgs.size(0)
        val_loss /= n_val
        val_dice /= n_val

        # Scheduler & checkpoint
        scheduler.step(val_dice)
        ckpt_msg = ''
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), os.path.join(PROJECT_ROOT, 'best_model.pth'))
            wait = 0
            ckpt_msg = '✅ Saved best model'
        else:
            wait += 1
        if wait >= patience:
            print(f"⏹ Early stopping at epoch {epoch}")
            break

        dt = time.time() - t0
        print(f"Epoch {epoch}/{num_epochs}  Time: {dt:.1f}s  "
              f"Train Loss: {train_loss:.4f}  Train Dice: {train_dice:.4f}  "
              f"Val Loss: {val_loss:.4f}  Val Dice: {val_dice:.4f}  {ckpt_msg}")

    print(f"Best Val Dice: {best_val_dice:.4f}")

# ─── Inference Function ─────────────────────────────────────────
def test():
    model = get_model().to(device)
    model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, 'best_model.pth')))
    model.eval()

    test_ds     = MRISegDataset(test_img_dir, mask_dir=None)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    with torch.no_grad():
        for i, (imgs, _) in enumerate(test_loader):
            imgs = imgs.to(device)
            logits = model(imgs)
            preds  = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype('uint8')
            for j, p in enumerate(preds):
                out_path = os.path.join(output_dir, f"test_{i*4 + j}.png")
                Image.fromarray(p[0]*255).save(out_path)
    print("✅ Inference done. Predictions saved to outputs/")


# ─── Main ───────────────────────────────────────────────────────
if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        test()

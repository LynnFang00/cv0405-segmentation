import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import MRISegDataset
from model import get_model
from infer import save_predictions
from utils import dice_score  # New utility file for clean code!

# Configurations
train_img_dir = 'data/train/images'
train_mask_dir = 'data/train/masks'
test_img_dir = 'data/test/images'
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 50
batch_size = 8
learning_rate = 1e-4

# Dataset and DataLoader
train_dataset = MRISegDataset(train_img_dir, train_mask_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model, Loss, Optimizer
model = get_model().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
print("\n🚀 Starting training...")

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    epoch_dice = 0

    loop = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]')

    for imgs, masks in loop:
        imgs, masks = imgs.to(device), masks.to(device)

        preds = model(imgs)
        loss = criterion(preds, masks)
        epoch_loss += loss.item()

        # Dice score calculation
        dice = dice_score(preds, masks)
        epoch_dice += dice

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(train_loader)
    avg_dice = epoch_dice / len(train_loader)

    print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_loss:.4f} - Train Dice: {avg_dice:.4f}')

# Save the model
torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
print(f'✅ Training complete. Model saved to {output_dir}/best_model.pth')

# Optional: Predict after training
save_predictions(model, test_img_dir, output_dir, device)
print(f'✅ Inference complete. Predictions saved to {output_dir}/')

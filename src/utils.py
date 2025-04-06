import torch

def dice_score(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()
    total = preds.sum() + targets.sum()

    dice = (2. * intersection + smooth) / (total + smooth)
    return dice.item()

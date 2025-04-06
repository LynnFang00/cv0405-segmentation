# =============================
# model.py
# =============================

import segmentation_models_pytorch as smp

def get_model():
    model = smp.Unet(
        encoder_name="resnet18",  # ✅ Pretrained encoder
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    )
    return model

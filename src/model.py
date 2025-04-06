import segmentation_models_pytorch as smp

def get_model():
    # Use ResNet18 as the encoder and output a single‐channel foreground probability map
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=1,
        classes=1,
    )
    return model

import os
import nibabel as nib
import numpy as np
from PIL import Image

# Source directories
cases_tr_dir  = 'data/train/cases_tr'
labels_tr_dir = 'data/train/labels_tr'
cases_ts_dir  = 'data/test/cases_ts'

# Output directories for slices
train_img_out = 'data/train/images'
train_mask_out= 'data/train/masks'
test_img_out  = 'data/test/images'

# Ensure output directories exist
for d in [train_img_out, train_mask_out, test_img_out]:
    os.makedirs(d, exist_ok=True)

# 1) Process the training set
for fname in sorted(os.listdir(cases_tr_dir)):
    if not fname.endswith(('.nii', '.nii.gz')):
        continue
    # Load the 3D volumes
    img_vol  = nib.load(os.path.join(cases_tr_dir,  fname)).get_fdata()
    mask_vol = nib.load(os.path.join(labels_tr_dir, fname)).get_fdata()
    # Normalize image volume to 0–255 and convert to uint8
    img_vol  = (img_vol  / np.max(img_vol)  * 255).astype(np.uint8)
    mask_vol = (mask_vol > 0).astype(np.uint8) * 255

    # Save each slice as a PNG
    for z in range(img_vol.shape[2]):
        base = fname.replace('.nii.gz','').replace('.nii','') + f'_{z:03d}.png'
        Image.fromarray(img_vol[:,:,z]).save(os.path.join(train_img_out,  base))
        Image.fromarray(mask_vol[:,:,z]).save(os.path.join(train_mask_out, base))

# 2) Process the test set
for fname in sorted(os.listdir(cases_ts_dir)):
    if not fname.endswith(('.nii', '.nii.gz')):
        continue
    img_vol = nib.load(os.path.join(cases_ts_dir, fname)).get_fdata()
    img_vol = (img_vol / np.max(img_vol) * 255).astype(np.uint8)
    for z in range(img_vol.shape[2]):
        base = fname.replace('.nii.gz','').replace('.nii','') + f'_{z:03d}.png'
        Image.fromarray(img_vol[:,:,z]).save(os.path.join(test_img_out, base))

print("Slices generated and saved in data/train/images, data/train/masks, and data/test/images")

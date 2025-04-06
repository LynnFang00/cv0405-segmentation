# MRI Segmentation with U-Net

This project implements MRI image segmentation using U-Net architecture. The goal is to predict brain tumor masks from MRI scans.

## Project Structure

- `data/` — Training and test data (images and masks)
- `src/` — Source code (training, inference, models)
- `outputs/` — Model checkpoints and inference predictions
- `requirements.txt` — Required Python packages

## Setup

```bash
# Clone repository and navigate to folder
git clone https://github.com/yourusername/cv0405-segmentation.git
cd cv0405-segmentation

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verihubs AI Engineer Test

A complete PyTorch implementation for fine-tuning Faster R-CNN ResNet50 FPN on face mask detection dataset.

## Project Overview

This project provides a full pipeline for training object detection models to detect:
- People wearing masks correctly (`with_mask`)
- People not wearing masks (`without_mask`)  
- People wearing masks incorrectly (`mask_weared_incorrect`)

## Project Structure

```
facemask-2/
├── data/                       # Dataset directory
│ ├── images/                   # Original images
│ ├── annotations/              # Pascal VOC XML annotations
│ ├── mosaic/                   # Mosaic augmented images
│ └── mosaic_annotations/       # Mosaic XML annotations
├── src/ # Source code
│ ├── data/
│ │ ├── dataset.py              # FaceMaskDataset (Pascal VOC)
│ │ └── transform.py            # Albumentations pipelines
│ ├── models/
│ │ └── faster_rcnn.py          # Model definitions (V1/V2)
│ ├── utils/
│ │ ├── utils.py                # Training utilities
│ │ └── inference_utils.py      # Inference utilities
│ ├── augment.py                # Mosaic generation
│ ├── train.py                  # Training script
│ └── inference.py              # Inference script
├── notebooks/                  # Analysis notebooks
│ ├── 000-data-explorations.ipynb
│ ├── 001-train-fasterrcnn.ipynb
│ └── 002-data-augment.ipynb
├── models/                     # Checkpoints (.pth)
├── outputs/                    # Inference results
├── README.md                   # Documentation
├── REPORT.md                   # Experiment results
├── .gitignore                  # Git exclusions
└── requirements.txt            # Dependencies


```

## Quick Start

### 1. Installation

```bash
# Install required packages
pip install -r requirements.txt
```

### 2. Download the dataset:
   - Visit https://www.kaggle.com/andrewmvd/face-mask-detection
   - Download `archive.zip` (or use Kaggle API)
   
   Using Kaggle API:
   ```bash
   # Install Kaggle API if not installed
   pip install kaggle
   
   # Make sure your Kaggle API credentials are set up
   # Download the dataset
   cd data/
   kaggle datasets download -d andrewmvd/face-mask-detection
   unzip face-mask-detection.zip -d ./
   rm face-mask-detection.zip
   cd ..
   ```

   After extraction, you should have:
   ```
   data/
   ├── images/           # 853 PNG images
   └── annotations/      # 853 XML files (PASCAL VOC format)
   ```

### 3. Generate Mosaic Augmentation

This preprocessing includes creating offline mosaic augmentation techniques.

```bash
cd src
python augment.py --data_dir ../data/ --num_incorrect_mask 300 --num_without_mask 1000
```

Or, you also can run interactively via `notebooks/002-data-augment.ipynb`.

### 4. Train Model

To train the model, you can run:

```bash
python train.py
```

where you also can customize the parameters via parser.

### 5. Run Inference

```bash
cd src
python inference.py
```
where the prediction result will be saved in `outputs/`. The model weights will be downloaded automatically as you run this file.
The prediction data is available in `.data/test/`. You can pass the image name as argument, e.g., `--image "../data/test/images/maksssksksss803.png"`.



**For experiment reproducibility, please see `REPORT.md`.**
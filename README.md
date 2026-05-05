# AMG-Colorization

**Advanced Multi-GANs towards near to real Image and Video Colorization**

PyTorch implementation of a research-level multi-GAN architecture for automatic
grayscale-to-RGB colorization of both images and video, with temporal harmony
for frame-consistent video output.

---

## Method Overview

The core idea is to decompose the colorization problem by image region, training
a **separate GAN per intensity cluster** so that each GAN specializes in
colorizing regions of a specific brightness range.

### Pipeline

1. **Edge Detection** — Canny edge detector (approximating the Dollar edge
   method) extracts structural boundaries from the grayscale input.
2. **Segmentation** — Connected-component labeling on the edge-inverted image
   partitions the image into homogeneous regions.
3. **SMV & Clustering** — The Segment Mean Value (mean grayscale intensity)
   of each region is computed, and regions are clustered via K-Means with
   a **dynamically determined K** derived from the image histogram.
4. **Multi-GAN Training** — One GAN (U-Net generator + PatchGAN discriminator)
   is trained per cluster, with **no parameter sharing**.
5. **Inference** — Regions are routed to their cluster's GAN for colorization,
   then composited into the final output.
6. **Video Harmony** — SIFT/ORB feature matching tracks regions across frames;
   a weighted mean blends matched-region colors to suppress flicker.

---

## Folder Structure

```
AMG-Colorization/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── configs/
│   └── default.yaml           # All hyperparameters
├── data/
│   └── dataset_loader.py      # ImageFolder & CIFAR-10 datasets
├── models/
│   ├── generator.py           # U-Net generator
│   ├── discriminator.py       # PatchGAN discriminator
│   ├── gan.py                 # Single GAN wrapper
│   └── multi_gan.py           # Multi-GAN ensemble container
├── preprocessing/
│   ├── edge_detection.py      # Canny & Dollar edge detectors
│   ├── segmentation.py        # Region segmentation via CC labeling
│   └── clustering.py          # Dynamic K-Means clustering
├── training/
│   ├── train.py               # Main training entry point
│   └── trainer.py             # Training loop orchestration
├── inference/
│   ├── image_infer.py         # Single-image colorization
│   └── video_infer.py         # Video colorization + harmony
├── utils/
│   ├── metrics.py             # PSNR, SSIM, FID
│   ├── visualization.py       # Plotting helpers
│   └── device.py              # Auto device selection
└── notebooks/
    └── demo.ipynb             # Demo notebook
```

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd AMG-Colorization

# Install dependencies
pip install -r requirements.txt

# (Optional) Install as editable package
pip install -e .
```

---

## Training

### Quick start with CIFAR-10

```bash
python training/train.py --config configs/default.yaml
```

This will:
- Download CIFAR-10 automatically
- Compute dynamic clustering on each image
- Train one GAN per cluster (up to 64 GANs)
- Save checkpoints to `./checkpoints/`
- Log losses and validation metrics

### Custom dataset

1. Prepare two folders:
   ```
   data/train/   # RGB images
   data/val/     # RGB images
   ```
2. Update `configs/default.yaml`:
   ```yaml
   data:
     dataset: "image_folder"
     train_path: "./data/train"
     val_path: "./data/val"
   ```
3. Run training:
   ```bash
   python training/train.py --config configs/default.yaml
   ```

### GPU

Training automatically uses CUDA if available. Force CPU:
```bash
python training/train.py --device cpu
```

---

## Inference

### Single image

```python
import cv2
from models.multi_gan import MultiGAN
from inference.image_infer import colorize_image

# Load model
multi_gan = MultiGAN(num_clusters=64, device=device)
multi_gan.load_all("./checkpoints/epoch_0100")

# Colorize
gray = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
result = colorize_image(multi_gan, gray, config, device)
cv2.imwrite("output.jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
```

### Video

```python
from inference.video_infer import colorize_video

colorize_video(
    multi_gan=multi_gan,
    input_path="gray_video.mp4",
    output_path="colorized_video.mp4",
    config=config,
    device=device,
)
```

---

## Configuration

All hyperparameters live in `configs/default.yaml`:

| Section | Key | Description | Default |
|---|---|---|---|
| data | dataset | `cifar10` or `image_folder` | `cifar10` |
| data | image_size | Input resize [H, W] | `[256, 256]` |
| clustering | gamma | Histogram frequency threshold | `0.05` |
| clustering | k_min / k_max | Cluster count bounds | `10` / `64` |
| training | epochs | Number of epochs | `100` |
| training | l1_lambda | L1 loss weight | `100.0` |
| inference | alpha | Harmony blend factor | `0.5` |

---

## Metrics

- **PSNR** — Peak Signal-to-Noise Ratio (dB)
- **SSIM** — Structural Similarity Index
- **FID** — Frechet Inception Distance (using pretrained InceptionV3)

---

## Requirements

- Python 3.8+
- PyTorch 1.12+
- torchvision
- OpenCV
- scikit-learn
- NumPy, SciPy, Matplotlib, tqdm

See `requirements.txt` for pinned versions.

---

## License

This implementation is provided for research purposes.


---

## Citation

You are welcome to cite our paper if you find it useful:

Jampour, M., Zare, M. & Javidi, M. Advanced multi-GANs towards near to real image and video colorization. J Ambient Intell Human Comput 14, 12857–12874 (2023). https://doi.org/10.1007/s12652-022-04206-z




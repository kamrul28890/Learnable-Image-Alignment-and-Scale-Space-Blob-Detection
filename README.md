# CS59300CVD — Assignment 2: Learnable Alignment & Blob Detection

## Overview

This repository contains two computer vision implementations:

- **Part 1**: Differentiable image alignment using spatial transformers and
  gradient descent (AdamW) to colourise Prokudin-Gorsky tri-plate photographs.
- **Part 2**: Scale-space blob detection using the Laplacian of Gaussian (LoG)
  with a downsampling pyramid and 3-D non-maximum suppression.

---

## Directory Structure

```
starter_code/
├── alignment_model.py      # Base class: exhaustive search alignment (coarse-to-fine)
├── diff_alignment.py       # DiffAlignment + AlignNet: gradient-descent alignment
├── main_p1.py              # CLI for Part 1
├── main_p2.py              # CLI for Part 2 (blob detection)
├── metrics.py              # ncc, mse, ssim loss functions
├── utils/
│   ├── misc_helper.py      # custom_shifts (circular / zero padding)
│   ├── io_helper.py        # torch_read_image, torch_save_image
│   └── draw_helper.py      # draw_all_circles (matplotlib)
├── data/
│   ├── part1/              # 1.jpg – 6.jpg (tri-plate images)
│   └── part2/              # butterfly, einstein, fishes, sunflowers
├── outputs/                # Created automatically on first run
└── README.md
```

---

## Requirements

### Python version
```
Python >= 3.10  (required for `X | Y` union type hints)
```

### Packages
Install dependencies with:
```bash
pip install torch torchvision scikit-image matplotlib numpy
```

| Package        | Tested version |
|----------------|---------------|
| torch          | 2.2.0         |
| torchvision    | 0.17.0        |
| scikit-image   | 0.22.0        |
| matplotlib     | 3.8.2         |
| numpy          | 1.26.4        |

> **GPU note**: The code automatically detects CUDA. If no GPU is available,
> it falls back to CPU transparently — no changes needed.

---

## Part 1 — Differentiable Alignment

### How it works

1. Load the stacked tri-plate JPEG (grayscale, normalised to [0, 1]).
2. Crop ~6 % film borders from all sides to remove dark film artefacts.
3. Split into three equal horizontal strips: Blue (top), Green (middle), Red (bottom).
4. For each of G→B and R→B: instantiate `AlignNet`, a tiny network whose only
   parameters are two learnable translation values `(tx, ty)` in normalised
   coordinates [-1, 1].
5. Apply the translation differentiably via `affine_grid` + `grid_sample`.
6. Minimise the chosen loss (MSE or NCC) with AdamW for 5000 steps.
7. Convert the final normalised shift to integer pixel offsets and apply with
   `custom_shifts`.
8. Stack the aligned channels as `(R, G, B)` and save.

### Run a single image

```bash
cd starter_code
python main_p1.py -i data/part1/1.jpg -m mse
```

### Run all images with NCC

```bash
python main_p1.py -i all -m ncc
```

### Options

| Flag       | Default | Description                                      |
|------------|---------|--------------------------------------------------|
| `-i`       | —       | Image path, or `all` for all 6 Part 1 images     |
| `-m`       | `mse`   | Loss metric: `mse` or `ncc`                      |
| `--lr`     | `0.005` | AdamW learning rate                              |
| `--steps`  | `5000`  | Number of gradient-descent iterations            |

### Outputs

Results are saved to `outputs/<id>_<metric>_aligned_DIFF.png`.

---

## Part 2 — LoG Blob Detection

### How it works

1. Load the image as grayscale, normalised to [0, 1].
2. Build a scale-normalised LoG kernel: `σ² · ∇²G(x, y; σ)`.
3. Build a scale-space pyramid by progressively **downsampling** the image by
   factor `1/k` at each level (equivalent to increasing the kernel size by `k`,
   but computationally cheaper). The LoG kernel size stays fixed.
4. Apply the LoG filter at each pyramid level; upsample responses back to the
   original resolution.
5. Stack squared responses into a 3-D volume `(n, H, W)`.
6. Run 3-D NMS using `max_pool3d` to find local maxima across space and scale.
7. Draw circles at detected keypoints with radius `σ · √2`.

### Run a single image

```bash
python main_p2.py -i data/part2/butterfly.jpg -s 2.0 -k 1.5 -n 10 --ksize 15
```

### Run all images

```bash
python main_p2.py -i all -s 2.0 -k 1.5 -n 10 --ksize 15
```

### Options

| Flag          | Default | Description                                         |
|---------------|---------|-----------------------------------------------------|
| `-i`          | —       | Image path, or `all` for all 4 Part 2 images        |
| `-s`          | `2.0`   | Base sigma (scale) for LoG kernel                   |
| `-k`          | `1.5`   | Scale factor between pyramid levels                 |
| `-n`          | `10`    | Number of pyramid levels                            |
| `--ksize`     | `15`    | LoG kernel side length (must be odd)                |
| `--threshold` | `0.003` | NMS response threshold (lower → more keypoints)     |

### Recommended parameters per image

| Image       | sigma | k   | n  | ksize | threshold |
|-------------|-------|-----|----|-------|-----------|
| butterfly   | 2.0   | 1.5 | 10 | 15    | 0.003     |
| einstein    | 3.0   | 1.5 | 8  | 15    | 0.003     |
| fishes      | 4.0   | 1.5 | 8  | 15    | 0.005     |
| sunflowers  | 5.0   | 1.5 | 8  | 15    | 0.005     |

### Outputs

Results are saved to `outputs/<name>-blob.jpg`.

---

## Code Style

All code follows **PEP 8**. To verify:
```bash
pip install pycodestyle
pycodestyle alignment_model.py diff_alignment.py main_p1.py main_p2.py metrics.py
```

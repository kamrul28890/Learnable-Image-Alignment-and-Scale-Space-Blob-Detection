# Learnable Image Alignment and Scale-Space Blob Detection

This repository contains a clean implementation of **CS59300CVD HW2** with two parts:

1. **Part 1**: Learnable channel alignment using differentiable warping and gradient-based optimization.
2. **Part 2**: Laplacian scale-space blob detection with multiple detection profiles.

## Repository Layout

- `alignment_model.py`: starter/base alignment model interface.
- `diff_alignment.py`: final differentiable alignment implementation.
- `main_p1.py`: Part 1 entry point.
- `main_p2.py`: Part 2 entry point.
- `metrics.py`: NCC/MSE/SSIM metrics.
- `utils/`: I/O, drawing, and helper utilities.
- `eval_part2_proxy.py`: proxy evaluation for Part 2.
- `eval_part2_manual_gt.py`: manual GT evaluation for Part 2.
- `manual_gt_workflow.py`: interactive manual GT annotation workflow.
- `data/`: input folder structure only (`part1/`, `part2/`).
- `outputs/`: output folder structure only.

## Requirements

- Python `>=3.10`
- Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Setup

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data

Input images are intentionally **not committed**.

Populate:

- `data/part1/` with assignment Part 1 images (`1.jpg` ... `6.jpg`)
- `data/part2/` with assignment Part 2 images (`butterfly.jpg`, `einstein.jpg`, `fishes.jpg`, `sunflowers.jpg`)

## Run Instructions

### Part 1

Single image:

```bash
python main_p1.py -i data/part1/1.jpg -m ncc
```

All images:

```bash
python main_p1.py -i all -m ncc
python main_p1.py -i all -m mse
```

### Part 2

Profiles available:

- `balanced`: conservative, cleaner detections
- `high_recall`: more detections with moderate extra false positives
- `dense`: very dense detections
- `exact`: exact dense pipeline (default)

Run all images with a profile:

```bash
python main_p2.py -i all --profile exact
python main_p2.py -i all --profile balanced
python main_p2.py -i all --profile high_recall
```

Run all 4 profiles and automatically generate proxy TP/FP/FN summaries:

```bash
python main_p2.py -i all --run_proxy_eval
```

This writes:

- `logs/part2_profile_proxy_per_image.csv`
- `logs/part2_profile_proxy_overall.csv`
- `logs/part2_profile_proxy_eval.md`
- `logs/part2_profile_proxy_eval.json`

## Evaluation (Optional)

Proxy evaluation:

```bash
python eval_part2_proxy.py
```

Manual GT workflow:

```bash
python manual_gt_workflow.py --mode bootstrap --bootstrap_source silver
python manual_gt_workflow.py --mode annotate
python eval_part2_manual_gt.py
```

## Reproducibility Notes

- `data/` and `outputs/` contents are intentionally excluded from version control.
- Runtime logs are generated under `logs/` locally.
- Core algorithm choices and run notes are in `HW2_REPORT_REFERENCE.md`.

## Report Build

Report sources are under `report_template/`:

```bash
cd report_template
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Final report content is in:
- `report_template/student_response/report.tex`
- `report_template/student_response/figures/`

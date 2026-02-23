"""Entry point for Part 1: Differentiable image alignment.

Runs DiffAlignment on one or all Prokudin-Gorsky tri-plate images and saves
the colourised RGB outputs to the outputs/ directory.

Usage:
    python main_p1.py -i data/part1/1.jpg -m mse
    python main_p1.py -i all -m ncc
"""

import argparse
import os

from diff_alignment import DiffAlignment


def main(args: argparse.Namespace) -> None:
    """Dispatch to single-image or batch mode."""
    os.makedirs('outputs', exist_ok=True)

    if args.image_name == 'all':
        run_all(args)
        return

    _run_single(args.image_name, args.metric, args.lr, args.steps)


def run_all(args: argparse.Namespace) -> None:
    """Run alignment on all six Part 1 images with the chosen metric."""
    for image_id in range(1, 7):
        image_name = f'data/part1/{image_id}.jpg'
        _run_single(image_name, args.metric, args.lr, args.steps)


def _run_single(image_name: str, metric: str,
                lr: float, steps: int) -> None:
    """Align a single image and save the result.

    Args:
        image_name (str): Path to the tri-plate JPEG.
        metric (str): Loss function ('mse' or 'ncc').
        lr (float): AdamW learning rate.
        steps (int): Number of optimisation steps.
    """
    base = os.path.splitext(os.path.basename(image_name))[0]
    output_path = f'outputs/{base}_{metric}_aligned_DIFF.png'

    print(f"\n{'='*60}")
    print(f"Image  : {image_name}")
    print(f"Metric : {metric} | LR: {lr} | Steps: {steps}")
    print(f"Output : {output_path}")
    print('=' * 60)

    model = DiffAlignment(image_name, metric=metric, lr=lr, n_steps=steps)
    model.align()
    model.save(output_path)
    print(f"Saved  : {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CS59300CVD Assignment 2 — Part 1: Differentiable Alignment'
    )
    parser.add_argument(
        '-i', '--image_name', required=True, type=str,
        help='Input image path, or "all" to process all Part 1 images.'
    )
    parser.add_argument(
        '-m', '--metric', default='mse', type=str,
        choices=['mse', 'ncc'],
        help='Loss metric for alignment. Default: mse'
    )
    parser.add_argument(
        '--lr', type=float, default=0.01,
        help='AdamW learning rate. Default: 0.005'
    )
    parser.add_argument(
        '--steps', type=int, default=5000,
        help='Number of gradient-descent steps. Default: 5000'
    )
    args = parser.parse_args()
    main(args)


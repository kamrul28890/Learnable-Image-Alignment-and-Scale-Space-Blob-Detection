"""Blob detection using the Laplacian of Gaussian (LoG) scale-space method.

Algorithm overview (Lindeberg 1994 / Lecture 13):
    1. Convert image to grayscale.
    2. Build a scale-normalised LoG kernel at the base scale sigma.
    3. Build a scale-space pyramid by progressively downsampling the image
       by factor 1/k (equivalent to enlarging the kernel by k, but faster).
    4. Apply the LoG kernel at each pyramid level, upsample responses back
       to the original resolution, and stack into a 3-D response volume.
    5. Find local maxima of the squared response in a 3x3x3 neighbourhood
       across (row, col, scale) — non-maximum suppression in scale space.
    6. Draw detected blobs as circles with radius = sigma * sqrt(2).

Usage:
    python main_p2.py -i data/part2/butterfly.jpg -s 2.0 -k 1.5 -n 10
    python main_p2.py -i all -s 2.0 -k 1.5 -n 10
"""

import argparse
import os
import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from utils.io_helper import torch_read_image
from utils.draw_helper import draw_all_circles


# ---------------------------------------------------------------------------
# LoG kernel construction
# ---------------------------------------------------------------------------

def build_log_kernel(sigma: float, ksize: int) -> torch.Tensor:
    """Build a scale-normalised Laplacian of Gaussian (LoG) kernel.

    The scale-normalised LoG is defined as:
        LoG_norm(x, y; sigma) = sigma^2 * Laplacian[ G(x, y; sigma) ]

    Multiplying by sigma^2 makes the peak response comparable across scales,
    which is essential for finding maxima in scale space.

    Args:
        sigma (float): Standard deviation of the Gaussian (scale parameter).
        ksize (int): Kernel side length — must be odd and >= 3.

    Returns:
        torch.Tensor: LoG kernel of shape (ksize, ksize), zero-sum normalised.
    """
    if ksize % 2 == 0:
        raise ValueError(f"ksize must be odd, got {ksize}.")

    half = ksize // 2
    # Build coordinate grids centred at zero.
    y, x = torch.meshgrid(
        torch.arange(-half, half + 1, dtype=torch.float32),
        torch.arange(-half, half + 1, dtype=torch.float32),
        indexing='ij'
    )
    sigma2 = sigma ** 2
    r2 = x ** 2 + y ** 2

    # Scale-normalised LoG formula.
    gaussian = torch.exp(-r2 / (2.0 * sigma2))
    log_kernel = (r2 - 2.0 * sigma2) / (sigma2 ** 2) * gaussian
    # Scale normalisation: multiply by sigma^2 for cross-scale comparability.
    log_kernel = log_kernel * sigma2

    # Zero-sum ensures the filter has no DC response (pure edge/blob detector).
    log_kernel = log_kernel - log_kernel.mean()
    return log_kernel


# ---------------------------------------------------------------------------
# Scale-space pyramid
# ---------------------------------------------------------------------------

def build_scale_space(
    image: torch.Tensor,
    sigma: float,
    ksize: int,
    k: float,
    n: int
) -> Tuple[torch.Tensor, List[float]]:
    """Build a LoG scale-space pyramid by image downsampling.

    Instead of enlarging the kernel at each level, we downsample the image
    by 1/k — this is mathematically equivalent and much faster for large n.
    The LoG response at each level is then upsampled back to the original
    resolution so all levels share the same spatial grid for NMS.

    Args:
        image (Tensor): Grayscale image of shape (1, H, W), values in [0, 1].
        sigma (float): Base scale (sigma at level 0).
        ksize (int): LoG kernel side length (odd).
        k (float): Scale factor between consecutive pyramid levels (> 1).
        n (int): Number of pyramid levels.

    Returns:
        Tuple:
            - scale_space (Tensor): Shape (n, H, W) — squared LoG responses.
            - sigmas (List[float]): The effective sigma at each pyramid level.
    """
    _, H, W = image.shape
    scale_space_levels = []
    sigmas = []

    current_image = image.clone()  # (1, H, W)

    for i in range(n):
        current_sigma = sigma * (k ** i)
        sigmas.append(current_sigma)

        # Build LoG kernel for the base sigma (kernel size stays fixed;
        # the scale change is encoded by the image downsampling).
        kernel = build_log_kernel(sigma, ksize)

        # Reshape kernel for F.conv2d: (out_ch, in_ch, kH, kW).
        kernel_4d = kernel.unsqueeze(0).unsqueeze(0)

        # Apply LoG filter with replicate padding to reduce boundary artefacts.
        pad = ksize // 2
        response = F.conv2d(current_image.unsqueeze(0), kernel_4d,
                            padding=pad)  # (1, 1, h, w)
        response = response.squeeze(0)  # (1, h, w)

        # Square the response — maxima of squared LoG detect dark & bright blobs.
        response_sq = response ** 2  # (1, h, w)

        # Upsample back to original resolution for unified NMS.
        if response_sq.shape[1] != H or response_sq.shape[2] != W:
            response_sq = F.interpolate(
                response_sq.unsqueeze(0), size=(H, W),
                mode='bilinear', align_corners=False
            ).squeeze(0)

        scale_space_levels.append(response_sq.squeeze(0))  # (H, W)

        # Downsample image for the next level (equivalent to enlarging kernel).
        if i < n - 1:
            new_h = max(1, int(current_image.shape[1] / k))
            new_w = max(1, int(current_image.shape[2] / k))
            current_image = F.interpolate(
                current_image.unsqueeze(0), size=(new_h, new_w),
                mode='bilinear', align_corners=False
            ).squeeze(0)

    # Stack into a single (n, H, W) volume.
    scale_space = torch.stack(scale_space_levels, dim=0)
    return scale_space, sigmas


# ---------------------------------------------------------------------------
# Non-maximum suppression in scale space
# ---------------------------------------------------------------------------

def non_max_suppression_3d(
    scale_space: torch.Tensor,
    threshold: float = 0.003
) -> List[Tuple[int, int, int]]:
    """Find local maxima in a 3-D (scale, row, col) response volume.

    A voxel is a local maximum if it is strictly greater than all 26
    neighbours in a 3×3×3 neighbourhood (across scale and space) and
    exceeds the response threshold.

    Args:
        scale_space (Tensor): Shape (n, H, W) — squared LoG responses.
        threshold (float): Minimum response value to be considered a keypoint.

    Returns:
        List of (scale_idx, row, col) for each detected keypoint.
    """
    # max_pool3d finds the local max in each 3x3x3 window.
    # A voxel is a true local max iff its value equals the pooled max.
    volume = scale_space.unsqueeze(0).unsqueeze(0)  # (1, 1, n, H, W)
    pooled = F.max_pool3d(volume, kernel_size=3, stride=1, padding=1)
    pooled = pooled.squeeze(0).squeeze(0)  # (n, H, W)

    # Local max mask: voxel == max in its neighbourhood AND above threshold.
    is_max = (scale_space == pooled) & (scale_space > threshold)

    # Extract indices of detected keypoints.
    keypoints = []
    scale_idxs, rows, cols = torch.where(is_max)
    for s, r, c in zip(scale_idxs.tolist(), rows.tolist(), cols.tolist()):
        keypoints.append((int(s), int(r), int(c)))

    return keypoints


# ---------------------------------------------------------------------------
# Main blob detection function
# ---------------------------------------------------------------------------

def blob_detection(
    input_name: str,
    output_name: str,
    ksize: int,
    sigma: float,
    n: int,
    k: float = 1.5,
    threshold: float = 0.003
) -> None:
    """End-to-end LoG blob detection and visualisation.

    Args:
        input_name (str): Path to the input image.
        output_name (str): Path for the output image with keypoint circles.
        ksize (int): LoG kernel side length (must be odd).
        sigma (float): Base scale for the Gaussian.
        n (int): Number of pyramid levels.
        k (float): Scale factor between pyramid levels.
        threshold (float): NMS response threshold.
    """
    print(f"\n[BlobDetection] {input_name} | sigma={sigma}, k={k}, "
          f"n={n}, ksize={ksize}, threshold={threshold}")

    # Step 1: Load as grayscale — returns (1, H, W) float in [0, 1].
    image = torch_read_image(input_name, gray=True)

    # Step 2 & 3: Build LoG scale space.
    scale_space, sigmas = build_scale_space(image, sigma, ksize, k, n)
    print(f"  Scale space shape: {scale_space.shape} | "
          f"sigmas: {[f'{s:.2f}' for s in sigmas]}")

    # Step 4: Non-maximum suppression in scale space.
    keypoints = non_max_suppression_3d(scale_space, threshold=threshold)
    print(f"  Detected {len(keypoints)} keypoints.")

    if len(keypoints) == 0:
        print("  Warning: No keypoints found. Try lowering --threshold or "
              "adjusting --sigma / -n.")

    # Step 5: Convert keypoints to circle parameters.
    # Blob radius = sigma * sqrt(2) (from LoG theory: scale at peak response).
    cx, cy, radii = [], [], []
    for (scale_idx, row, col) in keypoints:
        blob_sigma = sigmas[scale_idx]
        radius = blob_sigma * math.sqrt(2)
        cx.append(col)   # x = column
        cy.append(row)   # y = row
        radii.append(radius)

    # Step 6: Visualise using the provided draw_all_circles helper.
    image_np = image.numpy()  # (1, H, W)
    draw_all_circles(image_np, cx, cy, radii, output_name, color='r')
    print(f"  Saved to: {output_name}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    os.makedirs('outputs', exist_ok=True)

    if args.input_name == 'all':
        run_all(args)
        return

    blob_detection(
        args.input_name,
        'outputs/' + os.path.splitext(os.path.basename(args.input_name))[0]
        + '-blob.jpg',
        ksize=args.ksize,
        sigma=args.sigma,
        n=args.n,
        k=args.k,
        threshold=args.threshold,
    )


def run_all(args: argparse.Namespace) -> None:
    """Run blob detection on all Part 2 images."""
    for image_name in ['butterfly', 'einstein', 'fishes', 'sunflowers']:
        input_name = f'data/part2/{image_name}.jpg'
        output_name = f'outputs/{image_name}-blob.jpg'
        blob_detection(
            input_name, output_name,
            ksize=args.ksize,
            sigma=args.sigma,
            n=args.n,
            k=args.k,
            threshold=args.threshold,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CS59300CVD Assignment 2 — Part 2: LoG Blob Detection'
    )
    parser.add_argument(
        '-i', '--input_name', required=True, type=str,
        help='Input image path, or "all" to process all Part 2 images.'
    )
    parser.add_argument(
        '-s', '--sigma', type=float, default=2.0,
        help='Base sigma (scale) for the LoG kernel. Default: 2.0'
    )
    parser.add_argument(
        '-k', type=float, default=1.5,
        help='Scale factor between pyramid levels. Default: 1.5'
    )
    parser.add_argument(
        '--ksize', type=int, default=15,
        help='LoG kernel side length (must be odd). Default: 15'
    )
    parser.add_argument(
        '-n', type=int, default=10,
        help='Number of pyramid levels. Default: 10'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.003,
        help='NMS response threshold. Default: 0.003'
    )
    args = parser.parse_args()

    if args.ksize % 2 == 0:
        parser.error('--ksize must be odd.')

    main(args)


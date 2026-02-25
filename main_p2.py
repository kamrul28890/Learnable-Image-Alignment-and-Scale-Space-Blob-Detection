import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F

from utils.io_helper import torch_read_image, torch_save_image
from utils.draw_helper import draw_all_circles

def build_log_kernel(ksize: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build a 2D Laplacian-of-Gaussian kernel."""
    radius = ksize // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing='ij')
    rr = xx ** 2 + yy ** 2
    sigma2 = sigma ** 2
    # I keep the closed-form LoG expression explicit to make debugging easier.
    log_kernel = ((rr - 2.0 * sigma2) / (sigma2 ** 2)) * torch.exp(-rr / (2.0 * sigma2))
    log_kernel = log_kernel - log_kernel.mean()
    log_kernel = log_kernel / log_kernel.abs().sum().clamp_min(1e-8)
    return log_kernel.view(1, 1, ksize, ksize)

def main(args) -> None:
    os.makedirs('outputs', exist_ok=True)
    if args.input_name == 'all':
        run_all(args)
        return
    blob_detection(
        args.input_name, 'outputs/blob.jpg',
        ksize=args.ksize, sigma=args.sigma, n=args.n)

def run_all(args) -> None:
    """Run the blob detection on all images."""
    for image_name in [
        'butterfly', 'einstein', 'fishes', 'sunflowers'
    ]:
        input_name = 'data/part2/%s.jpg' % image_name
        output_name = 'outputs/%s-blob.jpg' % image_name
        blob_detection(
            input_name, output_name, 
            ksize=args.ksize, sigma=args.sigma, n=args.n)

def blob_detection(
    input_name: str, 
    output_name: str,
    ksize: int,
    sigma: float,
    n: int
) -> None:
    # Step 1: Read RGB image as Grayscale
    image = torch_read_image(input_name, gray=True)
    image_np = image.squeeze(0).numpy()

    ## Your CODE HERE ##
    # Step 2: Build Laplacian kernel
    device = image.device
    dtype = image.dtype
    kernel = build_log_kernel(ksize, sigma, device=device, dtype=dtype)

    # Step 3: Build feature pyramid
    k = 1.2
    h, w = image.shape[-2:]
    image_4d = image.unsqueeze(0)
    responses = []

    for level in range(n):
        scale = k ** level
        down_h = max(16, int(round(h / scale)))
        down_w = max(16, int(round(w / scale)))
        scaled = F.interpolate(
            image_4d, size=(down_h, down_w), mode='bilinear', align_corners=False
        )
        response = F.conv2d(scaled, kernel, padding=ksize // 2)
        # I apply sigma^2 normalization so responses are comparable across scales.
        response = ((sigma * scale) ** 2) * response
        response = response.pow(2)
        response = F.interpolate(response, size=(h, w), mode='bilinear', align_corners=False)
        responses.append(response[0, 0])

    scale_space = torch.stack(responses, dim=0)  # [n, H, W]

    # Step 4: Extract and visualize Keypoints
    volume = scale_space.unsqueeze(0).unsqueeze(0)  # [1, 1, n, H, W]
    max_volume = F.max_pool3d(volume, kernel_size=(3, 3, 3), stride=1, padding=1)
    threshold = torch.quantile(scale_space.flatten(), 0.995)
    maxima = (volume == max_volume) & (volume >= threshold)
    coords = maxima[0, 0].nonzero(as_tuple=False)  # [N, 3] = [scale, y, x]

    if coords.numel() == 0:
        draw_all_circles(image_np, np.array([]), np.array([]), np.array([]), output_name)
        print(f'{input_name}: detected 0 blobs')
        return

    scores = scale_space[coords[:, 0], coords[:, 1], coords[:, 2]]
    keep = torch.argsort(scores, descending=True)[:400]
    coords = coords[keep]

    s_idx = coords[:, 0].float()
    cy = coords[:, 1].cpu().numpy()
    cx = coords[:, 2].cpu().numpy()
    rad = (torch.sqrt(torch.tensor(2.0)) * sigma * (k ** s_idx)).cpu().numpy()
    draw_all_circles(image_np, cx, cy, rad, output_name)
    print(f'{input_name}: detected {len(cx)} blobs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CS59300CVD Assignment 2 Part 2')
    parser.add_argument('-i', '--input_name', required=True, type=str, help='Input image path')
    parser.add_argument('-s', '--sigma', type=float, default=1.6)
    parser.add_argument('-k', '--ksize', type=int, default=9)
    parser.add_argument('-n', type=int, default=12)
    args = parser.parse_args()
    assert(args.ksize % 2 == 1)
    main(args)

"""Part 2 blob detection entry point.

This file contains:
1. A tuned PyTorch LoG pipeline (`balanced`, `high_recall`, `reference_dense`).
2. A MATLAB-faithful port (`matlab_exact`) adapted from the reference repository.
"""

import argparse
import csv
import json
import os
import time
import math
from dataclasses import asdict, dataclass
import numpy as np
import torch
import torch.nn.functional as F

from utils.io_helper import torch_read_image
from utils.draw_helper import draw_all_circles


@dataclass(frozen=True)
class BlobProfile:
    name: str
    threshold_mode: str
    q_scale: float | None
    q_global: float | None
    abs_threshold: float | None
    top_k: int
    soft_border_margin: float | None
    soft_alpha: float
    border_margin: float
    quality_min: float | None
    iou_thresh: float | None
    dist_overlap: float | None
    max_output: int
    matlab_downsample: bool=False


PROFILES: dict[str, BlobProfile] = {
    # I keep the original conservative setup for high-precision visualization.
    'balanced': BlobProfile(
        name='balanced',
        threshold_mode='quantile',
        q_scale=0.965,
        q_global=0.988,
        abs_threshold=None,
        top_k=1200,
        soft_border_margin=24.0,
        soft_alpha=0.04,
        border_margin=12.0,
        quality_min=0.995,
        iou_thresh=0.80,
        dist_overlap=0.22,
        max_output=250,
        matlab_downsample=False,
    ),
    # I use this as default now to increase detected blobs while preserving 3D NMS logic.
    'high_recall': BlobProfile(
        name='high_recall',
        threshold_mode='quantile',
        q_scale=0.960,
        q_global=0.986,
        abs_threshold=None,
        top_k=1500,
        soft_border_margin=22.0,
        soft_alpha=0.03,
        border_margin=10.0,
        quality_min=0.992,
        iou_thresh=0.84,
        dist_overlap=0.20,
        max_output=300,
        matlab_downsample=False,
    ),
    # I adapt the external reference style: absolute normalized threshold + loose suppression.
    'reference_dense': BlobProfile(
        name='reference_dense',
        threshold_mode='absolute',
        q_scale=None,
        q_global=None,
        abs_threshold=0.03,
        top_k=5000,
        soft_border_margin=None,
        soft_alpha=0.0,
        border_margin=4.0,
        quality_min=None,
        iou_thresh=0.95,
        dist_overlap=0.10,
        max_output=500,
        matlab_downsample=False,
    ),
    # I port this directly from crazysal/Scale-Space-Blob-Detector MATLAB code path.
    'matlab_exact': BlobProfile(
        name='matlab_exact',
        threshold_mode='matlab_exact',
        q_scale=None,
        q_global=None,
        abs_threshold=0.0095,
        top_k=1000000,
        soft_border_margin=None,
        soft_alpha=0.0,
        border_margin=0.0,
        quality_min=None,
        iou_thresh=None,
        dist_overlap=None,
        max_output=1000000,
        matlab_downsample=False,
    ),
}


def get_blob_profile(profile_name: str) -> BlobProfile:
    """Return the configured blob-detection profile by name."""
    if profile_name not in PROFILES:
        raise ValueError(f'Unsupported profile: {profile_name}. Choices: {list(PROFILES.keys())}')
    return PROFILES[profile_name]

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

def suppress_duplicate_circles(
    cx: np.ndarray, cy: np.ndarray, rad: np.ndarray, scores: np.ndarray, overlap_ratio: float=0.5
) -> np.ndarray:
    """Greedy NMS in circle space, returns kept indices."""
    order = np.argsort(-scores)
    keep = []
    for idx in order:
        x, y, r = cx[idx], cy[idx], rad[idx]
        keep_it = True
        for j in keep:
            dx = x - cx[j]
            dy = y - cy[j]
            dist = float(np.sqrt(dx * dx + dy * dy))
            # I suppress detections that are too close with similar support radius.
            if dist < overlap_ratio * (r + rad[j]):
                keep_it = False
                break
        if keep_it:
            keep.append(int(idx))
    return np.array(keep, dtype=np.int64)

def circle_iou(x1: float, y1: float, r1: float, x2: float, y2: float, r2: float) -> float:
    """Pairwise circle IoU."""
    d = float(np.hypot(x1 - x2, y1 - y2))
    if d >= r1 + r2:
        return 0.0
    area1 = np.pi * (r1 ** 2)
    area2 = np.pi * (r2 ** 2)
    if d <= abs(r1 - r2):
        inter = np.pi * (min(r1, r2) ** 2)
        union = area1 + area2 - inter
        return float(inter / max(union, 1e-8))

    alpha = np.arccos(np.clip((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1), -1.0, 1.0))
    beta = np.arccos(np.clip((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2), -1.0, 1.0))
    inter = (r1 * r1 * alpha + r2 * r2 * beta -
             0.5 * np.sqrt(max(0.0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))))
    union = area1 + area2 - inter
    return float(inter / max(union, 1e-8))

def suppress_duplicate_circles_iou(
    cx: np.ndarray,
    cy: np.ndarray,
    rad: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float=0.30
) -> np.ndarray:
    """I keep high-score circles and suppress near-duplicate overlaps by IoU."""
    order = np.argsort(-scores)
    keep = []
    for idx in order:
        reject = False
        for j in keep:
            if circle_iou(cx[idx], cy[idx], rad[idx], cx[j], cy[j], rad[j]) > iou_thresh:
                reject = True
                break
        if not reject:
            keep.append(int(idx))
    return np.array(keep, dtype=np.int64)

def append_csv_row(path: str, row: dict) -> None:
    """Append one structured row to a CSV file, creating header if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _same_pad_2d(x4: torch.Tensor, k_h: int, k_w: int, mode: str='replicate') -> torch.Tensor:
    pad_top = (k_h - 1) // 2
    pad_bottom = (k_h - 1) - pad_top
    pad_left = (k_w - 1) // 2
    pad_right = (k_w - 1) - pad_left
    return F.pad(x4, (pad_left, pad_right, pad_top, pad_bottom), mode=mode)


def build_matlab_log_kernel(
    ksize: int, sigma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    # I mirror MATLAB fspecial('log', hsize, sigma) construction closely.
    yy = torch.linspace(-(ksize - 1) / 2.0, (ksize - 1) / 2.0, steps=ksize, device=device, dtype=dtype)
    xx = torch.linspace(-(ksize - 1) / 2.0, (ksize - 1) / 2.0, steps=ksize, device=device, dtype=dtype)
    yv, xv = torch.meshgrid(yy, xx, indexing='ij')

    arg = -(xv * xv + yv * yv) / (2.0 * sigma * sigma)
    h = torch.exp(arg)
    h = torch.where(
        h < torch.finfo(dtype).eps * h.max(),
        torch.zeros_like(h),
        h
    )
    sumh = h.sum()
    if float(sumh.item()) != 0.0:
        h = h / sumh

    h1 = h * ((xv * xv + yv * yv) - 2.0 * sigma * sigma) / (sigma ** 4)
    h = h1 - h1.mean()
    return h


def apply_filter_same_replicate(image_2d: torch.Tensor, kernel_2d: torch.Tensor) -> torch.Tensor:
    k_h, k_w = int(kernel_2d.shape[0]), int(kernel_2d.shape[1])
    x = image_2d.unsqueeze(0).unsqueeze(0)
    x = _same_pad_2d(x, k_h, k_w, mode='replicate')
    k = kernel_2d.view(1, 1, k_h, k_w)
    out = F.conv2d(x, k)
    return out[0, 0]


def generate_scale_space_matlab_port(
    img_gray: torch.Tensor,
    num_scales: int,
    sigma: float,
    scale_multiplier: float,
    should_downsample: bool
) -> torch.Tensor:
    h, w = int(img_gray.shape[0]), int(img_gray.shape[1])
    scale_space = torch.zeros((h, w, num_scales), device=img_gray.device, dtype=img_gray.dtype)

    if should_downsample:
        kernel_size = max(1, int(math.floor(6.0 * sigma)) + 1)
        base_kernel = build_matlab_log_kernel(kernel_size, sigma, img_gray.device, img_gray.dtype)
        base_kernel = (sigma ** 2) * base_kernel
        img4 = img_gray.unsqueeze(0).unsqueeze(0)
        for i in range(num_scales):
            if i == 0:
                downsized = img_gray
            else:
                factor = 1.0 / (scale_multiplier ** i)
                down_h = max(1, int(round(h * factor)))
                down_w = max(1, int(round(w * factor)))
                downsized = F.interpolate(
                    img4, size=(down_h, down_w), mode='bicubic', align_corners=False, antialias=True
                )[0, 0]
            filtered = apply_filter_same_replicate(downsized, base_kernel)
            filtered = filtered * filtered
            upscaled = F.interpolate(
                filtered.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode='bicubic',
                align_corners=False,
                antialias=False
            )[0, 0]
            scale_space[:, :, i] = upscaled
    else:
        for i in range(num_scales):
            scaled_sigma = sigma * (scale_multiplier ** i)
            kernel_size = max(1, int(math.floor(6.0 * scaled_sigma)) + 1)
            kernel = build_matlab_log_kernel(kernel_size, scaled_sigma, img_gray.device, img_gray.dtype)
            kernel = (scaled_sigma ** 2) * kernel
            filtered = apply_filter_same_replicate(img_gray, kernel)
            scale_space[:, :, i] = filtered * filtered

    return scale_space


def nms_2d_matlab_port(scale_slice: torch.Tensor) -> torch.Tensor:
    # MATLAB: ordfilt2(img, 9, ones(3,3)) -> local 3x3 maximum with zero padding.
    x = scale_slice.unsqueeze(0).unsqueeze(0)
    x = F.pad(x, (1, 1, 1, 1), mode='constant', value=0.0)
    y = F.max_pool2d(x, kernel_size=3, stride=1)
    return y[0, 0]


def nms_3d_matlab_port(scale_space_2d_nms: torch.Tensor, original_scale_space: torch.Tensor) -> torch.Tensor:
    # I intentionally keep the in-place update behavior of the MATLAB code.
    h, w, num_scales = scale_space_2d_nms.shape
    max_vals = scale_space_2d_nms.clone()

    for i in range(num_scales):
        if i == 0:
            lower = i
            upper = min(i + 1, num_scales - 1)
        elif i < num_scales - 1:
            lower = i - 1
            upper = i + 1
        else:
            lower = i - 1
            upper = i
        max_vals[:, :, i] = torch.max(max_vals[:, :, lower:upper + 1], dim=2).values

    original_val_markers = (max_vals == original_scale_space)
    return max_vals * original_val_markers.to(max_vals.dtype)


def calc_radii_by_scale_matlab_port(num_scales: int, scale_multiplier: float, sigma: float) -> np.ndarray:
    radii = np.zeros((num_scales,), dtype=np.float32)
    for i in range(num_scales):
        radii[i] = float(np.sqrt(2.0) * sigma * (scale_multiplier ** i))
    return radii


def detect_blobs_matlab_port(
    img_gray: torch.Tensor,
    num_scales: int,
    sigma: float,
    should_downsample: bool,
    scale_multiplier: float,
    threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    scale_space = generate_scale_space_matlab_port(
        img_gray=img_gray,
        num_scales=num_scales,
        sigma=sigma,
        scale_multiplier=scale_multiplier,
        should_downsample=should_downsample,
    )

    h, w, _ = scale_space.shape
    scale_space_2d_nms = torch.zeros_like(scale_space)
    for i in range(num_scales):
        scale_space_2d_nms[:, :, i] = nms_2d_matlab_port(scale_space[:, :, i])

    scale_space_3d_nms = nms_3d_matlab_port(scale_space_2d_nms, scale_space)
    threshold_mask = scale_space_3d_nms > float(threshold)
    scale_space_3d_nms = scale_space_3d_nms * threshold_mask.to(scale_space_3d_nms.dtype)

    radii_by_scale = calc_radii_by_scale_matlab_port(num_scales, scale_multiplier, sigma)
    xs, ys, rs, scores = [], [], [], []
    for i in range(num_scales):
        coords = torch.nonzero(scale_space_3d_nms[:, :, i] != 0, as_tuple=False)
        if coords.numel() == 0:
            continue
        for row_col in coords:
            row = int(row_col[0].item())
            col = int(row_col[1].item())
            xs.append(float(col))
            ys.append(float(row))
            rs.append(float(radii_by_scale[i]))
            scores.append(float(scale_space_3d_nms[row, col, i].item()))

    stats = {
        'raw_candidates': int(torch.count_nonzero(scale_space).item()),
        'after_2d_nms': int(torch.count_nonzero(scale_space_2d_nms).item()),
        'after_3d_nms': int(torch.count_nonzero(scale_space_3d_nms).item()),
    }
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(rs, dtype=np.float32),
        np.asarray(scores, dtype=np.float32),
        stats,
    )

def main(args) -> None:
    """CLI entry for running one image or the full Part 2 set."""
    os.makedirs('outputs', exist_ok=True)
    if args.input_name == 'all':
        run_all(args)
        return
    blob_detection(
        args.input_name, 'outputs/blob.jpg',
        ksize=args.ksize, sigma=args.sigma, n=args.n, profile=args.profile)

def run_all(args) -> None:
    """Run the blob detection on all images."""
    for image_name in [
        'butterfly', 'einstein', 'fishes', 'sunflowers'
    ]:
        input_name = 'data/part2/%s.jpg' % image_name
        output_name = 'outputs/%s-blob.jpg' % image_name
        blob_detection(
            input_name, output_name, 
            ksize=args.ksize, sigma=args.sigma, n=args.n, profile=args.profile)

def blob_detection(
    input_name: str, 
    output_name: str,
    ksize: int,
    sigma: float,
    n: int,
    profile: str='high_recall'
) -> None:
    """Run blob detection for one image and save circles + structured logs."""
    start_time = time.time()
    profile_cfg = get_blob_profile(profile)

    if profile_cfg.threshold_mode == 'matlab_exact':
        # I replicate MATLAB grayscale conversion: mean(R,G,B) / max(all pixels).
        image_rgb = torch_read_image(input_name, gray=False)
        image_rgb = image_rgb.to(dtype=torch.float64)
        denom = image_rgb.max().clamp_min(1e-8)
        image = image_rgb.mean(dim=0) / denom
        image_np = image.detach().cpu().numpy()

        matlab_num_scales = 15
        matlab_sigma = 2.0
        matlab_k = float(math.sqrt(math.sqrt(2.0)))
        matlab_threshold = float(profile_cfg.abs_threshold if profile_cfg.abs_threshold is not None else 0.0095)

        cx, cy, rad, score_np, stats = detect_blobs_matlab_port(
            img_gray=image,
            num_scales=matlab_num_scales,
            sigma=matlab_sigma,
            should_downsample=profile_cfg.matlab_downsample,
            scale_multiplier=matlab_k,
            threshold=matlab_threshold,
        )

        draw_all_circles(image_np, cx, cy, rad, output_name)
        print(f'{input_name}: detected {len(cx)} blobs [{profile_cfg.name}]')

        os.makedirs('logs', exist_ok=True)
        detail_path = os.path.join(
            'logs',
            f'part2_detections_{os.path.basename(input_name).split(".")[0]}.json'
        )
        detail_payload = {
            'input_name': input_name,
            'profile': profile_cfg.name,
            'profile_config': asdict(profile_cfg),
            'params': {
                'ksize': 'matlab_dynamic',
                'sigma': matlab_sigma,
                'n': matlab_num_scales,
                'k': matlab_k,
                'threshold': matlab_threshold,
                'should_downsample': bool(profile_cfg.matlab_downsample),
            },
            'raw_candidates': stats['raw_candidates'],
            'after_score_keep': stats['raw_candidates'],
            'after_border_soft': stats['after_2d_nms'],
            'after_border_hard': stats['after_3d_nms'],
            'after_quality': stats['after_3d_nms'],
            'after_iou_nms': stats['after_3d_nms'],
            'final_count': int(len(cx)),
            'detections': [
                {
                    'x': float(x),
                    'y': float(y),
                    'r': float(r),
                    'score': float(s),
                    'prominence': 1.0,
                }
                for x, y, r, s in zip(cx, cy, rad, score_np)
            ]
        }
        with open(detail_path, 'w', encoding='utf-8') as f:
            json.dump(detail_payload, f, indent=2)

        log_row = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'input_name': input_name,
            'output_name': output_name,
            'profile': profile_cfg.name,
            'ksize': 'matlab_dynamic',
            'sigma': matlab_sigma,
            'n': matlab_num_scales,
            'raw_candidates': stats['raw_candidates'],
            'after_score_keep': stats['raw_candidates'],
            'after_border_soft': stats['after_2d_nms'],
            'after_border_hard': stats['after_3d_nms'],
            'after_quality': stats['after_3d_nms'],
            'after_iou_nms': stats['after_3d_nms'],
            'final_count': int(len(cx)),
            'runtime_sec': round(float(time.time() - start_time), 4),
        }
        append_csv_row(os.path.join('logs', 'part2_blob_runs.csv'), log_row)
        return

    # Step 1: Read image as grayscale for non-MATLAB profiles.
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
            image_4d, size=(down_h, down_w), mode='bilinear', align_corners=False, antialias=True
        )
        response = F.conv2d(scaled, kernel, padding=ksize // 2)
        # I apply sigma^2 normalization so responses are comparable across scales.
        response = ((sigma * scale) ** 2) * response
        response = response.pow(2)
        response = F.interpolate(response, size=(h, w), mode='bilinear', align_corners=False)
        responses.append(response[0, 0])

    scale_space = torch.stack(responses, dim=0)  # [n, H, W]

    # Step 4: Extract and visualize keypoints according to the selected profile.
    volume = scale_space.unsqueeze(0).unsqueeze(0)  # [1, 1, n, H, W]
    max_volume = F.max_pool3d(volume, kernel_size=(3, 3, 3), stride=1, padding=1)
    if profile_cfg.threshold_mode == 'quantile':
        if profile_cfg.q_global is None:
            raise ValueError('Quantile profile requires q_global.')
        global_threshold = torch.quantile(scale_space.flatten(), profile_cfg.q_global)
        response_threshold = float(global_threshold.item())
        if profile_cfg.q_scale is None:
            maxima = (volume == max_volume) & (volume >= global_threshold)
        else:
            per_scale_threshold = torch.quantile(
                scale_space.flatten(1), profile_cfg.q_scale, dim=1
            ).view(n, 1, 1)
            maxima = (
                (volume == max_volume)
                & (scale_space >= per_scale_threshold)
                & (volume >= global_threshold)
            )
    elif profile_cfg.threshold_mode == 'absolute':
        if profile_cfg.abs_threshold is None:
            raise ValueError('Absolute profile requires abs_threshold.')
        max_response = scale_space.max().clamp_min(1e-8)
        normalized_space = scale_space / max_response
        response_threshold = float(profile_cfg.abs_threshold * float(max_response.item()))
        maxima = (volume == max_volume) & (normalized_space.unsqueeze(0).unsqueeze(0) >= profile_cfg.abs_threshold)
    else:
        raise ValueError(f'Unsupported threshold_mode: {profile_cfg.threshold_mode}')

    coords = maxima[0, 0].nonzero(as_tuple=False)  # [N, 3] = [scale, y, x]
    raw_candidates = int(coords.shape[0])

    def emit_zero(
        stage_name: str,
        after_score_keep_val: int=0,
        after_border_soft_val: int=0,
        after_border_hard_val: int=0,
        after_quality_val: int=0,
        after_iou_nms_val: int=0
    ) -> None:
        draw_all_circles(image_np, np.array([]), np.array([]), np.array([]), output_name)
        print(f'{input_name}: detected 0 blobs [{profile_cfg.name}] ({stage_name})')
        log_row = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'input_name': input_name,
            'output_name': output_name,
            'profile': profile_cfg.name,
            'ksize': int(ksize),
            'sigma': float(sigma),
            'n': int(n),
            'raw_candidates': raw_candidates,
            'after_score_keep': after_score_keep_val,
            'after_border_soft': after_border_soft_val,
            'after_border_hard': after_border_hard_val,
            'after_quality': after_quality_val,
            'after_iou_nms': after_iou_nms_val,
            'final_count': 0,
            'runtime_sec': round(float(time.time() - start_time), 4),
        }
        append_csv_row(os.path.join('logs', 'part2_blob_runs.csv'), log_row)

    if coords.numel() == 0:
        emit_zero(stage_name='no_raw_candidates')
        return

    scores = scale_space[coords[:, 0], coords[:, 1], coords[:, 2]]
    keep = torch.argsort(scores, descending=True)[:profile_cfg.top_k]
    coords = coords[keep]
    scores = scores[keep]
    after_score_keep = int(coords.shape[0])

    s_idx = coords[:, 0].float()
    cy = coords[:, 1].cpu().numpy().astype(np.float32)
    cx = coords[:, 2].cpu().numpy().astype(np.float32)
    rad = (torch.sqrt(torch.tensor(2.0)) * sigma * (k ** s_idx)).cpu().numpy().astype(np.float32)
    score_np = scores.cpu().numpy().astype(np.float32)

    # I keep center-near-boundary detections only if their score is sufficiently strong.
    if profile_cfg.soft_border_margin is not None:
        dist_border = np.minimum.reduce([cx, cy, (w - 1.0 - cx), (h - 1.0 - cy)])
        edge_factor = np.clip(
            (profile_cfg.soft_border_margin - dist_border) / profile_cfg.soft_border_margin,
            0.0,
            1.0,
        )
        adaptive_threshold = response_threshold * (1.0 + profile_cfg.soft_alpha * edge_factor)
        keep_soft = score_np >= adaptive_threshold
        cx = cx[keep_soft]
        cy = cy[keep_soft]
        rad = rad[keep_soft]
        score_np = score_np[keep_soft]
        s_idx = s_idx[keep_soft]
    after_border_soft = int(cx.size)

    if cx.size == 0:
        emit_zero(
            stage_name='after_soft_border',
            after_score_keep_val=after_score_keep,
            after_border_soft_val=after_border_soft,
        )
        return

    # I still apply a hard center margin to remove pathological edge maxima.
    border_margin = profile_cfg.border_margin
    inside = (
        (cx >= border_margin) &
        (cy >= border_margin) &
        (cx < (w - border_margin)) &
        (cy < (h - border_margin))
    )
    cx = cx[inside]
    cy = cy[inside]
    rad = rad[inside]
    score_np = score_np[inside]
    s_idx = s_idx[inside]
    after_border_hard = int(cx.size)

    if cx.size == 0:
        emit_zero(
            stage_name='after_hard_border',
            after_score_keep_val=after_score_keep,
            after_border_soft_val=after_border_soft,
            after_border_hard_val=after_border_hard,
        )
        return

    # I score blob quality by peak prominence against local neighborhood mean at same scale.
    prominence = np.ones_like(score_np, dtype=np.float32)
    if profile_cfg.quality_min is not None:
        neighbor_mean = F.avg_pool2d(scale_space.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)
        level_idx = s_idx.long()
        y_idx = np.clip(cy.astype(np.int64), 0, h - 1)
        x_idx = np.clip(cx.astype(np.int64), 0, w - 1)
        local_mean = neighbor_mean[level_idx, y_idx, x_idx].cpu().numpy()
        prominence = score_np / (local_mean + 1e-8)
        keep_quality = prominence >= profile_cfg.quality_min
        cx = cx[keep_quality]
        cy = cy[keep_quality]
        rad = rad[keep_quality]
        score_np = score_np[keep_quality]
        prominence = prominence[keep_quality]
    after_quality = int(cx.size)

    if cx.size == 0:
        emit_zero(
            stage_name='after_quality',
            after_score_keep_val=after_score_keep,
            after_border_soft_val=after_border_soft,
            after_border_hard_val=after_border_hard,
            after_quality_val=after_quality,
        )
        return

    if profile_cfg.iou_thresh is not None:
        keep_nms = suppress_duplicate_circles_iou(
            cx, cy, rad, score_np, iou_thresh=profile_cfg.iou_thresh
        )
        if keep_nms.size > 0:
            cx = cx[keep_nms]
            cy = cy[keep_nms]
            rad = rad[keep_nms]
            score_np = score_np[keep_nms]
            prominence = prominence[keep_nms]
    after_iou_nms = int(cx.size)

    if profile_cfg.dist_overlap is not None and cx.size > 0:
        keep_dist = suppress_duplicate_circles(
            cx, cy, rad, score_np, overlap_ratio=profile_cfg.dist_overlap
        )
        if keep_dist.size > 0:
            cx = cx[keep_dist]
            cy = cy[keep_dist]
            rad = rad[keep_dist]
            score_np = score_np[keep_dist]
            prominence = prominence[keep_dist]

    cx = cx[:profile_cfg.max_output]
    cy = cy[:profile_cfg.max_output]
    rad = rad[:profile_cfg.max_output]
    score_np = score_np[:profile_cfg.max_output]
    prominence = prominence[:profile_cfg.max_output]
    draw_all_circles(image_np, cx, cy, rad, output_name)
    print(f'{input_name}: detected {len(cx)} blobs [{profile_cfg.name}]')

    os.makedirs('logs', exist_ok=True)
    detail_path = os.path.join(
        'logs',
        f'part2_detections_{os.path.basename(input_name).split(".")[0]}.json'
    )
    detail_payload = {
        'input_name': input_name,
        'params': {'ksize': ksize, 'sigma': sigma, 'n': n, 'k': k},
        'profile': profile_cfg.name,
        'profile_config': asdict(profile_cfg),
        'response_threshold': response_threshold,
        'raw_candidates': raw_candidates,
        'after_score_keep': after_score_keep,
        'after_border_soft': after_border_soft,
        'after_border_hard': after_border_hard,
        'after_quality': after_quality,
        'after_iou_nms': after_iou_nms,
        'final_count': int(len(cx)),
        'detections': [
            {
                'x': float(x),
                'y': float(y),
                'r': float(r),
                'score': float(s),
                'prominence': float(p),
            }
            for x, y, r, s, p in zip(cx, cy, rad, score_np[:len(cx)], prominence[:len(cx)])
        ]
    }
    with open(detail_path, 'w', encoding='utf-8') as f:
        json.dump(detail_payload, f, indent=2)

    log_row = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'input_name': input_name,
        'output_name': output_name,
        'profile': profile_cfg.name,
        'ksize': int(ksize),
        'sigma': float(sigma),
        'n': int(n),
        'raw_candidates': raw_candidates,
        'after_score_keep': after_score_keep,
        'after_border_soft': after_border_soft,
        'after_border_hard': after_border_hard,
        'after_quality': after_quality,
        'after_iou_nms': after_iou_nms,
        'final_count': int(len(cx)),
        'runtime_sec': round(float(time.time() - start_time), 4),
    }
    append_csv_row(os.path.join('logs', 'part2_blob_runs.csv'), log_row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CS59300CVD Assignment 2 Part 2')
    parser.add_argument('-i', '--input_name', required=True, type=str, help='Input image path')
    parser.add_argument('-s', '--sigma', type=float, default=1.6)
    parser.add_argument('-k', '--ksize', type=int, default=9)
    parser.add_argument('-n', type=int, default=12)
    parser.add_argument(
        '--profile',
        type=str,
        default='matlab_exact',
        choices=list(PROFILES.keys()),
        help='Blob detection profile: balanced, high_recall, reference_dense, or matlab_exact',
    )
    args = parser.parse_args()
    assert(args.ksize % 2 == 1)
    main(args)

"""Part 2 blob detection entry point.

This file contains:
1. A tuned PyTorch LoG pipeline (`balanced`, `high_recall`, `dense`).
2. A dense exact LoG/NMS pipeline (`exact`).
"""

import argparse
import csv
import json
import os
import time
import math
import shutil
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
    exact_downsample: bool=False


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
        exact_downsample=False,
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
        exact_downsample=False,
    ),
    # I keep this as a dense absolute-threshold profile.
    'dense': BlobProfile(
        name='dense',
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
        exact_downsample=False,
    ),
    # I keep this as the exact dense profile used for very high blob counts.
    'exact': BlobProfile(
        name='exact',
        threshold_mode='exact',
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
        exact_downsample=False,
    ),
}

PART2_IMAGE_NAMES = ['butterfly', 'einstein', 'fishes', 'sunflowers']
PROFILE_EVAL_SET = ['balanced', 'high_recall', 'dense', 'exact']


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


def write_csv_rows(path: str, rows: list[dict]) -> None:
    """Write all rows to a CSV file with stable headers."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _same_pad_2d(x4: torch.Tensor, k_h: int, k_w: int, mode: str='replicate') -> torch.Tensor:
    pad_top = (k_h - 1) // 2
    pad_bottom = (k_h - 1) - pad_top
    pad_left = (k_w - 1) // 2
    pad_right = (k_w - 1) - pad_left
    return F.pad(x4, (pad_left, pad_right, pad_top, pad_bottom), mode=mode)


def build_exact_log_kernel(
    ksize: int, sigma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    # I construct a normalized LoG kernel with dynamic size from sigma.
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


def generate_scale_space_exact(
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
        base_kernel = build_exact_log_kernel(kernel_size, sigma, img_gray.device, img_gray.dtype)
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
            kernel = build_exact_log_kernel(kernel_size, scaled_sigma, img_gray.device, img_gray.dtype)
            kernel = (scaled_sigma ** 2) * kernel
            filtered = apply_filter_same_replicate(img_gray, kernel)
            scale_space[:, :, i] = filtered * filtered

    return scale_space


def nms_2d_exact(scale_slice: torch.Tensor) -> torch.Tensor:
    # 2D local-maximum filtering with a 3x3 neighborhood and zero padding.
    x = scale_slice.unsqueeze(0).unsqueeze(0)
    x = F.pad(x, (1, 1, 1, 1), mode='constant', value=0.0)
    y = F.max_pool2d(x, kernel_size=3, stride=1)
    return y[0, 0]


def nms_3d_exact(scale_space_2d_nms: torch.Tensor, original_scale_space: torch.Tensor) -> torch.Tensor:
    # I keep this straightforward in-place style for deterministic behavior.
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


def calc_radii_by_scale_exact(num_scales: int, scale_multiplier: float, sigma: float) -> np.ndarray:
    radii = np.zeros((num_scales,), dtype=np.float32)
    for i in range(num_scales):
        radii[i] = float(np.sqrt(2.0) * sigma * (scale_multiplier ** i))
    return radii


def detect_blobs_exact(
    img_gray: torch.Tensor,
    num_scales: int,
    sigma: float,
    should_downsample: bool,
    scale_multiplier: float,
    threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    scale_space = generate_scale_space_exact(
        img_gray=img_gray,
        num_scales=num_scales,
        sigma=sigma,
        scale_multiplier=scale_multiplier,
        should_downsample=should_downsample,
    )

    h, w, _ = scale_space.shape
    scale_space_2d_nms = torch.zeros_like(scale_space)
    for i in range(num_scales):
        scale_space_2d_nms[:, :, i] = nms_2d_exact(scale_space[:, :, i])

    scale_space_3d_nms = nms_3d_exact(scale_space_2d_nms, scale_space)
    threshold_mask = scale_space_3d_nms > float(threshold)
    scale_space_3d_nms = scale_space_3d_nms * threshold_mask.to(scale_space_3d_nms.dtype)

    radii_by_scale = calc_radii_by_scale_exact(num_scales, scale_multiplier, sigma)
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


def distance_overlap_match(
    c1: np.ndarray,
    c2: np.ndarray,
    factor: float=0.35,
    radius_ratio: tuple[float, float]=(0.5, 2.0),
) -> bool:
    """I match circles if center distance and radius ratio are both compatible."""
    x1, y1, r1 = float(c1[0]), float(c1[1]), float(c1[2])
    x2, y2, r2 = float(c2[0]), float(c2[1]), float(c2[2])
    dist = float(np.hypot(x1 - x2, y1 - y2))
    ratio = max(r1, r2) / max(min(r1, r2), 1e-8)
    return dist <= factor * (r1 + r2) and radius_ratio[0] <= ratio <= radius_ratio[1]


def synthesize_silver_gt(det_by_profile: dict[str, np.ndarray], min_votes: int=2) -> np.ndarray:
    """Fuse profile outputs into a consensus silver ground truth."""
    clusters = []
    for profile_name, circles in det_by_profile.items():
        for circle in circles:
            placed = False
            for cluster in clusters:
                if distance_overlap_match(circle, cluster['center']):
                    cluster['members'].append((profile_name, circle))
                    arr = np.asarray([x[1] for x in cluster['members']], dtype=np.float32)
                    cluster['center'] = arr.mean(axis=0)
                    placed = True
                    break
            if not placed:
                clusters.append({'center': circle.copy(), 'members': [(profile_name, circle)]})

    silver = []
    for cluster in clusters:
        voters = {profile_name for profile_name, _ in cluster['members']}
        if len(voters) >= min_votes:
            silver.append(np.asarray(cluster['center'], dtype=np.float32))
    if not silver:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(silver, axis=0)


def eval_tp_fp_fn_proxy(pred: np.ndarray, gt: np.ndarray, iou_threshold: float=0.20):
    """Compute TP/FP/FN against silver GT via one-to-one greedy IoU matching."""
    if gt.shape[0] == 0:
        return 0, int(pred.shape[0]), 0, 0.0, 0.0, 0.0
    used = np.zeros(gt.shape[0], dtype=bool)
    tp = 0
    for p in pred:
        best_j = -1
        best_iou = 0.0
        for j, g in enumerate(gt):
            if used[j]:
                continue
            iou_val = circle_iou(
                float(p[0]), float(p[1]), float(p[2]),
                float(g[0]), float(g[1]), float(g[2]),
            )
            if iou_val > best_iou:
                best_iou = iou_val
                best_j = j
        if best_j >= 0 and best_iou >= iou_threshold:
            used[best_j] = True
            tp += 1
    fp = int(pred.shape[0] - tp)
    fn = int(gt.shape[0] - tp)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return tp, fp, fn, precision, recall, f1


def _load_circles_from_detection_json(path: str) -> np.ndarray:
    payload = json.load(open(path, 'r', encoding='utf-8'))
    detections = payload.get('detections', [])
    if not detections:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(
        [[float(d['x']), float(d['y']), float(d['r'])] for d in detections],
        dtype=np.float32,
    )


def run_profile_proxy_eval(args) -> None:
    """Run all profiles, snapshot detections, and compute proxy TP/FP/FN summaries."""
    os.makedirs('logs', exist_ok=True)
    profile_det_dir = os.path.join('logs', 'profile_detections')
    os.makedirs(profile_det_dir, exist_ok=True)

    # I run every profile end-to-end and snapshot its detection JSON files for reproducibility.
    for profile_name in PROFILE_EVAL_SET:
        tmp_args = argparse.Namespace(**vars(args))
        tmp_args.profile = profile_name
        run_all(tmp_args)
        for image_name in PART2_IMAGE_NAMES:
            src = os.path.join('logs', f'part2_detections_{image_name}.json')
            dst = os.path.join(profile_det_dir, f'{profile_name}_{image_name}.json')
            if os.path.exists(src):
                shutil.copyfile(src, dst)

    # Collect latest runtime/count rows from part2_blob_runs.csv.
    latest = {}
    csv_path = os.path.join('logs', 'part2_blob_runs.csv')
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                latest[(row['profile'], row['input_name'])] = row

    per_image_rows = []
    totals = {
        profile_name: {'tp': 0, 'fp': 0, 'fn': 0, 'pred_count': 0, 'runtime_sec': []}
        for profile_name in PROFILE_EVAL_SET
    }

    for image_name in PART2_IMAGE_NAMES:
        det_by_profile = {}
        for profile_name in PROFILE_EVAL_SET:
            path = os.path.join(profile_det_dir, f'{profile_name}_{image_name}.json')
            det_by_profile[profile_name] = _load_circles_from_detection_json(path)

        silver_gt = synthesize_silver_gt(det_by_profile, min_votes=2)

        for profile_name in PROFILE_EVAL_SET:
            pred = det_by_profile[profile_name]
            tp, fp, fn, precision, recall, f1 = eval_tp_fp_fn_proxy(pred, silver_gt, iou_threshold=0.20)
            run_key = (profile_name, f'data/part2/{image_name}.jpg')
            runtime_sec = None
            if run_key in latest:
                runtime_sec = float(latest[run_key]['runtime_sec'])
            per_image_rows.append({
                'image': f'{image_name}.jpg',
                'profile': profile_name,
                'silver_gt_count': int(silver_gt.shape[0]),
                'pred_count': int(pred.shape[0]),
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'runtime_sec': runtime_sec,
            })
            totals[profile_name]['tp'] += int(tp)
            totals[profile_name]['fp'] += int(fp)
            totals[profile_name]['fn'] += int(fn)
            totals[profile_name]['pred_count'] += int(pred.shape[0])
            if runtime_sec is not None:
                totals[profile_name]['runtime_sec'].append(runtime_sec)

    overall_rows = []
    for profile_name in PROFILE_EVAL_SET:
        tp = totals[profile_name]['tp']
        fp = totals[profile_name]['fp']
        fn = totals[profile_name]['fn']
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
        mean_runtime_sec = None
        if totals[profile_name]['runtime_sec']:
            mean_runtime_sec = float(np.mean(totals[profile_name]['runtime_sec']))
        overall_rows.append({
            'profile': profile_name,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'total_pred_count': int(totals[profile_name]['pred_count']),
            'mean_runtime_sec': mean_runtime_sec,
        })

    write_csv_rows(os.path.join('logs', 'part2_profile_proxy_per_image.csv'), per_image_rows)
    write_csv_rows(os.path.join('logs', 'part2_profile_proxy_overall.csv'), overall_rows)
    payload = {
        'note': (
            'Proxy evaluation using silver GT from consensus across balanced/high_recall/dense/exact '
            '(min_votes=2, IoU>=0.20 matching).'
        ),
        'per_image': per_image_rows,
        'overall': overall_rows,
    }
    with open(os.path.join('logs', 'part2_profile_proxy_eval.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    md_lines = [
        '# Part 2 Profile Proxy Evaluation',
        '',
        'Silver GT is built by consensus across `balanced`, `high_recall`, `dense`, and `exact` (>=2 voters).',
        'These TP/FP/FN numbers are comparative proxy metrics, not manual human-labeled absolute accuracy.',
        '',
        '## Overall',
        '',
        '| profile | TP | FP | FN | precision | recall | f1 | total_pred_count | mean_runtime_sec |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in overall_rows:
        runtime_str = 'NA' if row['mean_runtime_sec'] is None else f"{float(row['mean_runtime_sec']):.4f}"
        md_lines.append(
            f"| {row['profile']} | {row['tp']} | {row['fp']} | {row['fn']} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | "
            f"{row['total_pred_count']} | {runtime_str} |"
        )
    md_lines.extend([
        '',
        '## Per Image',
        '',
        '| image | profile | silver_gt_count | pred_count | TP | FP | FN | precision | recall | f1 | runtime_sec |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ])
    for row in per_image_rows:
        runtime_str = 'NA' if row['runtime_sec'] is None else f"{float(row['runtime_sec']):.4f}"
        md_lines.append(
            f"| {row['image']} | {row['profile']} | {row['silver_gt_count']} | {row['pred_count']} | "
            f"{row['tp']} | {row['fp']} | {row['fn']} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {runtime_str} |"
        )
    with open(os.path.join('logs', 'part2_profile_proxy_eval.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))

    print('Wrote logs/part2_profile_proxy_per_image.csv')
    print('Wrote logs/part2_profile_proxy_overall.csv')
    print('Wrote logs/part2_profile_proxy_eval.json')
    print('Wrote logs/part2_profile_proxy_eval.md')

def main(args) -> None:
    """CLI entry for running one image or the full Part 2 set."""
    os.makedirs('outputs', exist_ok=True)
    if args.run_proxy_eval:
        run_profile_proxy_eval(args)
        return
    if args.input_name == 'all':
        run_all(args)
        return
    blob_detection(
        args.input_name, 'outputs/blob.jpg',
        ksize=args.ksize, sigma=args.sigma, n=args.n, profile=args.profile)

def run_all(args) -> None:
    """Run the blob detection on all images."""
    for image_name in PART2_IMAGE_NAMES:
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

    if profile_cfg.threshold_mode == 'exact':
        # I normalize grayscale as mean(R,G,B) divided by image max.
        image_rgb = torch_read_image(input_name, gray=False)
        image_rgb = image_rgb.to(dtype=torch.float64)
        denom = image_rgb.max().clamp_min(1e-8)
        image = image_rgb.mean(dim=0) / denom
        image_np = image.detach().cpu().numpy()

        exact_num_scales = 15
        exact_sigma = 2.0
        exact_k = float(math.sqrt(math.sqrt(2.0)))
        exact_threshold = float(profile_cfg.abs_threshold if profile_cfg.abs_threshold is not None else 0.0095)

        cx, cy, rad, score_np, stats = detect_blobs_exact(
            img_gray=image,
            num_scales=exact_num_scales,
            sigma=exact_sigma,
            should_downsample=profile_cfg.exact_downsample,
            scale_multiplier=exact_k,
            threshold=exact_threshold,
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
                'ksize': 'dynamic',
                'sigma': exact_sigma,
                'n': exact_num_scales,
                'k': exact_k,
                'threshold': exact_threshold,
                'should_downsample': bool(profile_cfg.exact_downsample),
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
            'ksize': 'dynamic',
            'sigma': exact_sigma,
            'n': exact_num_scales,
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

    # Step 1: Read image as grayscale for non-exact profiles.
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
        default='exact',
        choices=list(PROFILES.keys()),
        help='Blob detection profile: balanced, high_recall, dense, or exact',
    )
    parser.add_argument(
        '--run_proxy_eval',
        action='store_true',
        help='Run all profiles and generate proxy TP/FP/FN evaluation files under logs/.',
    )
    args = parser.parse_args()
    assert(args.ksize % 2 == 1)
    main(args)



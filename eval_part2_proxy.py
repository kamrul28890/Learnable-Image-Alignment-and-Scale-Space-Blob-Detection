"""Proxy (silver-GT) evaluation helpers for Part 2 blob detection."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision


@dataclass
class BlobConfig:
    name: str
    q_scale: float | None
    q_global: float
    top_k: int
    soft_border_margin: float | None
    soft_alpha: float
    border_margin: float
    quality_min: float | None
    iou_thresh: float | None
    dist_overlap: float | None


def read_gray(path: Path) -> torch.Tensor:
    """Read an image as grayscale in [0, 1]."""
    return torchvision.io.read_image(str(path), torchvision.io.ImageReadMode.GRAY) / 255.0


def build_log_kernel(ksize: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build a normalized LoG kernel."""
    radius = ksize // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing='ij')
    rr = xx ** 2 + yy ** 2
    sigma2 = sigma ** 2
    log_kernel = ((rr - 2.0 * sigma2) / (sigma2 ** 2)) * torch.exp(-rr / (2.0 * sigma2))
    log_kernel = log_kernel - log_kernel.mean()
    log_kernel = log_kernel / log_kernel.abs().sum().clamp_min(1e-8)
    return log_kernel.view(1, 1, ksize, ksize)


def circle_iou(c1, c2) -> float:
    x1, y1, r1 = c1
    x2, y2, r2 = c2
    d = float(np.hypot(x1 - x2, y1 - y2))
    if d >= r1 + r2:
        return 0.0
    area1 = np.pi * (r1 ** 2)
    area2 = np.pi * (r2 ** 2)
    if d <= abs(r1 - r2):
        inter = np.pi * (min(r1, r2) ** 2)
        return float(inter / max(area1 + area2 - inter, 1e-8))

    alpha = np.arccos(np.clip((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1), -1.0, 1.0))
    beta = np.arccos(np.clip((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2), -1.0, 1.0))
    inter = (
        r1 * r1 * alpha
        + r2 * r2 * beta
        - 0.5 * np.sqrt(max(0.0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)))
    )
    return float(inter / max(area1 + area2 - inter, 1e-8))


def distance_overlap_match(c1, c2, factor=0.35, radius_ratio=(0.5, 2.0)) -> bool:
    x1, y1, r1 = c1
    x2, y2, r2 = c2
    d = float(np.hypot(x1 - x2, y1 - y2))
    ratio = max(r1, r2) / max(min(r1, r2), 1e-8)
    return d <= factor * (r1 + r2) and radius_ratio[0] <= ratio <= radius_ratio[1]


def suppress_iou(circles, scores, thresh):
    order = np.argsort(-scores)
    keep = []
    for idx in order:
        if all(circle_iou(circles[idx], circles[j]) <= thresh for j in keep):
            keep.append(int(idx))
    return np.array(keep, dtype=np.int64)


def suppress_dist(circles, scores, overlap_ratio):
    order = np.argsort(-scores)
    keep = []
    for idx in order:
        x, y, r = circles[idx]
        ok = True
        for j in keep:
            xj, yj, rj = circles[j]
            d = float(np.hypot(x - xj, y - yj))
            if d < overlap_ratio * (r + rj):
                ok = False
                break
        if ok:
            keep.append(int(idx))
    return np.array(keep, dtype=np.int64)


def build_scale_space(
    image: torch.Tensor, ksize=9, sigma=1.6, n=12, k=1.2
) -> tuple[torch.Tensor, dict]:
    """Compute squared, scale-normalized LoG responses over n levels."""
    h, w = image.shape[-2:]
    kernel = build_log_kernel(ksize, sigma, image.device, image.dtype)
    responses = []
    image_4d = image.unsqueeze(0)
    for level in range(n):
        scale = k ** level
        down_h = max(16, int(round(h / scale)))
        down_w = max(16, int(round(w / scale)))
        scaled = F.interpolate(
            image_4d, size=(down_h, down_w), mode='bilinear', align_corners=False, antialias=True
        )
        response = F.conv2d(scaled, kernel, padding=ksize // 2)
        response = ((sigma * scale) ** 2) * response
        response = response.pow(2)
        response = F.interpolate(response, size=(h, w), mode='bilinear', align_corners=False)
        responses.append(response[0, 0])
    scale_space = torch.stack(responses, dim=0)
    return scale_space, {'h': h, 'w': w, 'k': k, 'sigma': sigma, 'n': n}


def detect_from_scale_space(scale_space: torch.Tensor, meta: dict, cfg: BlobConfig):
    """Detect circles from scale space using profile thresholds and NMS."""
    h, w = meta['h'], meta['w']
    k, sigma, n = meta['k'], meta['sigma'], meta['n']
    volume = scale_space.unsqueeze(0).unsqueeze(0)
    max_volume = F.max_pool3d(volume, kernel_size=(3, 3, 3), stride=1, padding=1)
    if cfg.q_scale is None:
        maxima = (volume == max_volume) & (volume >= torch.quantile(scale_space.flatten(), cfg.q_global))
    else:
        per_scale_threshold = torch.quantile(scale_space.flatten(1), cfg.q_scale, dim=1).view(n, 1, 1)
        global_threshold = torch.quantile(scale_space.flatten(), cfg.q_global)
        maxima = (volume == max_volume) & (scale_space >= per_scale_threshold) & (volume >= global_threshold)

    coords = maxima[0, 0].nonzero(as_tuple=False)
    if coords.numel() == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    scores = scale_space[coords[:, 0], coords[:, 1], coords[:, 2]]
    keep = torch.argsort(scores, descending=True)[: cfg.top_k]
    coords = coords[keep]
    scores = scores[keep]

    s_idx = coords[:, 0].float()
    cy = coords[:, 1].cpu().numpy().astype(np.float32)
    cx = coords[:, 2].cpu().numpy().astype(np.float32)
    rad = (torch.sqrt(torch.tensor(2.0)) * sigma * (k ** s_idx)).cpu().numpy().astype(np.float32)
    score_np = scores.cpu().numpy().astype(np.float32)
    circles = np.stack([cx, cy, rad], axis=1)

    if cfg.soft_border_margin is not None:
        dist_border = np.minimum.reduce([cx, cy, (w - 1.0 - cx), (h - 1.0 - cy)])
        edge_factor = np.clip((cfg.soft_border_margin - dist_border) / cfg.soft_border_margin, 0.0, 1.0)
        global_threshold = float(torch.quantile(scale_space.flatten(), cfg.q_global))
        adaptive_threshold = global_threshold * (1.0 + cfg.soft_alpha * edge_factor)
        keep_soft = score_np >= adaptive_threshold
        circles = circles[keep_soft]
        score_np = score_np[keep_soft]
        s_idx = s_idx[keep_soft]

    inside = (
        (circles[:, 0] >= cfg.border_margin)
        & (circles[:, 1] >= cfg.border_margin)
        & (circles[:, 0] < (w - cfg.border_margin))
        & (circles[:, 1] < (h - cfg.border_margin))
    )
    circles = circles[inside]
    score_np = score_np[inside]
    s_idx = s_idx[inside]

    if circles.shape[0] == 0:
        return circles, score_np

    if cfg.quality_min is not None:
        neighbor_mean = F.avg_pool2d(scale_space.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)
        y_idx = np.clip(circles[:, 1].astype(np.int64), 0, h - 1)
        x_idx = np.clip(circles[:, 0].astype(np.int64), 0, w - 1)
        lvl_idx = s_idx.long()
        local_mean = neighbor_mean[lvl_idx, y_idx, x_idx].cpu().numpy()
        prominence = score_np / (local_mean + 1e-8)
        keep_quality = prominence >= cfg.quality_min
        circles = circles[keep_quality]
        score_np = score_np[keep_quality]

    if circles.shape[0] == 0:
        return circles, score_np

    if cfg.iou_thresh is not None:
        keep_iou = suppress_iou(circles, score_np, cfg.iou_thresh)
        circles = circles[keep_iou]
        score_np = score_np[keep_iou]

    if cfg.dist_overlap is not None and circles.shape[0] > 0:
        keep_dist = suppress_dist(circles, score_np, cfg.dist_overlap)
        circles = circles[keep_dist]
        score_np = score_np[keep_dist]

    return circles[:250], score_np[:250]


def synthesize_silver_gt(det_by_method: dict[str, np.ndarray], min_votes=2):
    """Fuse multiple detector outputs into a consensus pseudo-ground-truth."""
    methods = list(det_by_method.keys())
    clusters = []
    for m in methods:
        for c in det_by_method[m]:
            placed = False
            for cl in clusters:
                if distance_overlap_match(c, cl['center'], factor=0.35):
                    cl['members'].append((m, c))
                    arr = np.array([x[1] for x in cl['members']], dtype=np.float32)
                    cl['center'] = arr.mean(axis=0)
                    placed = True
                    break
            if not placed:
                clusters.append({'center': c.copy(), 'members': [(m, c)]})

    gt = []
    for cl in clusters:
        voters = {m for m, _ in cl['members']}
        if len(voters) >= min_votes:
            gt.append(np.array(cl['center'], dtype=np.float32))
    if not gt:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(gt, axis=0)


def eval_tp_fp_fn(pred: np.ndarray, gt: np.ndarray):
    """Match predicted circles to GT circles with IoU and compute P/R/F1 stats."""
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
            iou = circle_iou(p, g)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= 0.20:
            used[best_j] = True
            tp += 1
    fp = int(pred.shape[0] - tp)
    fn = int(gt.shape[0] - tp)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return tp, fp, fn, precision, recall, f1


def main():
    """Run proxy evaluation and emit markdown + JSON summaries."""
    root = Path(__file__).resolve().parent
    data_root = root / 'data' / 'part2'
    logs_root = root / 'logs'
    logs_root.mkdir(exist_ok=True)

    image_names = ['butterfly.jpg', 'einstein.jpg', 'fishes.jpg', 'sunflowers.jpg']

    methods = {
        # Current code path (balanced / conservative-precision)
        'current_balanced': BlobConfig(
            name='current_balanced',
            q_scale=0.965,
            q_global=0.988,
            top_k=1200,
            soft_border_margin=24.0,
            soft_alpha=0.04,
            border_margin=12.0,
            quality_min=0.995,
            iou_thresh=0.80,
            dist_overlap=0.22,
        ),
        # Earlier run that visually had higher recall
        'previous_high_recall': BlobConfig(
            name='previous_high_recall',
            q_scale=None,
            q_global=0.992,
            top_k=600,
            soft_border_margin=None,
            soft_alpha=0.0,
            border_margin=15.0,
            quality_min=None,
            iou_thresh=None,
            dist_overlap=0.55,
        ),
        # Over-pruned setting used briefly
        'over_pruned': BlobConfig(
            name='over_pruned',
            q_scale=0.985,
            q_global=0.992,
            top_k=900,
            soft_border_margin=30.0,
            soft_alpha=0.25,
            border_margin=15.0,
            quality_min=1.06,
            iou_thresh=0.30,
            dist_overlap=0.55,
        ),
    }

    per_image = {}
    totals = {k: {'tp': 0, 'fp': 0, 'fn': 0} for k in methods}

    for name in image_names:
        image = read_gray(data_root / name)
        scale_space, meta = build_scale_space(image, ksize=9, sigma=1.6, n=12, k=1.2)
        det = {}
        for method_name, cfg in methods.items():
            circles, scores = detect_from_scale_space(scale_space, meta, cfg)
            det[method_name] = circles

        silver_gt = synthesize_silver_gt(det, min_votes=2)
        image_result = {'silver_gt_count': int(silver_gt.shape[0]), 'methods': {}}
        for method_name in methods:
            tp, fp, fn, p, r, f1 = eval_tp_fp_fn(det[method_name], silver_gt)
            image_result['methods'][method_name] = {
                'count': int(det[method_name].shape[0]),
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'precision': p,
                'recall': r,
                'f1': f1,
            }
            totals[method_name]['tp'] += tp
            totals[method_name]['fp'] += fp
            totals[method_name]['fn'] += fn
        per_image[name] = image_result

    summary = {}
    for method_name, t in totals.items():
        tp, fp, fn = t['tp'], t['fp'], t['fn']
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        summary[method_name] = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    payload = {
        'note': (
            'Silver-ground-truth evaluation. TP/FP/FN are proxy metrics based on consensus across methods, '
            'not manual human annotations.'
        ),
        'methods': {k: vars(v) for k, v in methods.items()},
        'per_image': per_image,
        'overall': summary,
    }

    json_path = logs_root / 'part2_proxy_eval.json'
    json_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    md = []
    md.append('# Part 2 Proxy TP/FP Evaluation')
    md.append('')
    md.append('This uses a **silver ground truth** (consensus of at least 2 methods).')
    md.append('Metrics are relative/comparative, not absolute human-labeled accuracy.')
    md.append('')
    md.append('## Overall')
    md.append('')
    md.append('| method | TP | FP | FN | precision | recall | f1 |')
    md.append('|---|---:|---:|---:|---:|---:|---:|')
    for method_name, s in summary.items():
        md.append(
            f"| {method_name} | {s['tp']} | {s['fp']} | {s['fn']} | "
            f"{s['precision']:.4f} | {s['recall']:.4f} | {s['f1']:.4f} |"
        )
    md.append('')
    md.append('## Per Image')
    md.append('')
    for image_name, img_data in per_image.items():
        md.append(f'### {image_name}')
        md.append(f"- silver_gt_count: {img_data['silver_gt_count']}")
        md.append('| method | count | TP | FP | FN | precision | recall | f1 |')
        md.append('|---|---:|---:|---:|---:|---:|---:|---:|')
        for method_name, m in img_data['methods'].items():
            md.append(
                f"| {method_name} | {m['count']} | {m['tp']} | {m['fp']} | {m['fn']} | "
                f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |"
            )
        md.append('')

    md_path = logs_root / 'part2_proxy_eval.md'
    md_path.write_text('\n'.join(md), encoding='utf-8')
    print(f'Wrote {json_path}')
    print(f'Wrote {md_path}')


if __name__ == '__main__':
    main()

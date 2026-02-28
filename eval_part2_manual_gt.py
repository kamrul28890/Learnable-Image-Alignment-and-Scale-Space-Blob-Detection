"""Evaluate Part 2 detections against manually annotated center points."""

import csv
import json
from pathlib import Path

from eval_part2_proxy import BlobConfig, build_scale_space, detect_from_scale_space, read_gray


def load_json(path: Path):
    """Load JSON payload from disk."""
    return json.loads(path.read_text(encoding='utf-8'))


def match_points_to_circles(gt_points, pred_circles, dist_factor=0.45, min_tol=6.0):
    """
    GT: points [(x, y)]
    Pred: circles [(x, y, r)]
    I count a match if center distance <= max(min_tol, dist_factor * r_pred).
    """
    used = [False] * len(pred_circles)
    tp = 0
    for gx, gy in gt_points:
        best_idx = -1
        best_dist = 1e18
        for i, (px, py, pr) in enumerate(pred_circles):
            if used[i]:
                continue
            d = ((gx - px) ** 2 + (gy - py) ** 2) ** 0.5
            tol = max(min_tol, dist_factor * pr)
            if d <= tol and d < best_dist:
                best_dist = d
                best_idx = i
        if best_idx >= 0:
            used[best_idx] = True
            tp += 1
    fp = len(pred_circles) - tp
    fn = len(gt_points) - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return tp, fp, fn, precision, recall, f1


def write_csv(path: Path, rows: list[dict]):
    """Write a list of dictionaries to CSV with stable headers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    """Run manual-GT evaluation and write markdown/csv/json summaries."""
    root = Path(__file__).resolve().parent
    data_root = root / 'data' / 'part2'
    logs_root = root / 'logs'
    gt_path = logs_root / 'part2_manual_gt_points.json'
    if not gt_path.exists():
        raise FileNotFoundError(
            f'Manual GT not found: {gt_path}. Run manual_gt_workflow.py first.'
        )

    gt_payload = load_json(gt_path)
    gt_images = gt_payload.get('images', {})
    image_names = ['butterfly.jpg', 'einstein.jpg', 'fishes.jpg', 'sunflowers.jpg']

    methods = {
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

    per_image_rows = []
    totals = {m: {'tp': 0, 'fp': 0, 'fn': 0} for m in methods}

    for image_name in image_names:
        image = read_gray(data_root / image_name)
        scale_space, meta = build_scale_space(image, ksize=9, sigma=1.6, n=12, k=1.2)
        gt_points = gt_images.get(image_name, [])
        gt_points = [(float(p[0]), float(p[1])) for p in gt_points]
        for method_name, cfg in methods.items():
            pred_circles, _ = detect_from_scale_space(scale_space, meta, cfg)
            pred = [(float(c[0]), float(c[1]), float(c[2])) for c in pred_circles]
            tp, fp, fn, p, r, f1 = match_points_to_circles(gt_points, pred)
            per_image_rows.append({
                'image': image_name,
                'method': method_name,
                'gt_count': len(gt_points),
                'pred_count': len(pred),
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'precision': round(p, 6),
                'recall': round(r, 6),
                'f1': round(f1, 6),
            })
            totals[method_name]['tp'] += tp
            totals[method_name]['fp'] += fp
            totals[method_name]['fn'] += fn

    overall_rows = []
    for method_name, t in totals.items():
        tp, fp, fn = t['tp'], t['fp'], t['fn']
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        overall_rows.append({
            'method': method_name,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': round(p, 6),
            'recall': round(r, 6),
            'f1': round(f1, 6),
        })

    write_csv(logs_root / 'part2_manual_eval_per_image.csv', per_image_rows)
    write_csv(logs_root / 'part2_manual_eval_overall.csv', overall_rows)

    md = []
    md.append('# Part 2 Manual GT Evaluation')
    md.append('')
    md.append(f'GT file: `{gt_path.name}`')
    md.append('')
    md.append('## Overall')
    md.append('')
    md.append('| method | TP | FP | FN | precision | recall | f1 |')
    md.append('|---|---:|---:|---:|---:|---:|---:|')
    for r in overall_rows:
        md.append(
            f"| {r['method']} | {r['tp']} | {r['fp']} | {r['fn']} | "
            f"{r['precision']:.6f} | {r['recall']:.6f} | {r['f1']:.6f} |"
        )
    md.append('')
    md.append('## Per Image')
    md.append('')
    md.append('| image | method | gt_count | pred_count | TP | FP | FN | precision | recall | f1 |')
    md.append('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for r in per_image_rows:
        md.append(
            f"| {r['image']} | {r['method']} | {r['gt_count']} | {r['pred_count']} | "
            f"{r['tp']} | {r['fp']} | {r['fn']} | {r['precision']:.6f} | {r['recall']:.6f} | {r['f1']:.6f} |"
        )

    (logs_root / 'part2_manual_eval.md').write_text('\n'.join(md), encoding='utf-8')
    payload = {
        'gt_file': str(gt_path),
        'overall': overall_rows,
        'per_image': per_image_rows,
    }
    (logs_root / 'part2_manual_eval.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'Wrote {logs_root / "part2_manual_eval.md"}')


if __name__ == '__main__':
    main()

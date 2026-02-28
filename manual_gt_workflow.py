"""Interactive workflow to bootstrap and refine manual point annotations for Part 2."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from PIL import Image

from eval_part2_proxy import (
    BlobConfig,
    build_scale_space,
    detect_from_scale_space,
    read_gray,
    synthesize_silver_gt,
)


def load_json(path: Path, default):
    """Load JSON if available, otherwise return the provided default payload."""
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return default


def save_json(path: Path, payload):
    """Save JSON payload with indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def annotate_points(image_path: Path, existing_points: list[list[float]]) -> list[list[float]]:
    """Open an interactive window to add/remove center points for one image."""
    image = np.array(Image.open(image_path).convert('L'))
    points = [list(map(float, p[:2])) for p in existing_points]

    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.set_title(
        f'{image_path.name}\n'
        'Left-click: add point | Right-click: remove nearest | Enter: next image | q: quit/save'
    )

    artists = []

    def redraw():
        nonlocal artists
        for a in artists:
            a.remove()
        artists = []
        for i, (x, y) in enumerate(points):
            c = Circle((x, y), 4, color='r', fill=False, linewidth=1.2)
            ax.add_patch(c)
            t = ax.text(x + 3, y + 3, str(i), color='yellow', fontsize=7)
            artists.extend([c, t])
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            points.append([float(event.xdata), float(event.ydata)])
        elif event.button == 3 and points:
            arr = np.array(points, dtype=np.float32)
            d = np.hypot(arr[:, 0] - float(event.xdata), arr[:, 1] - float(event.ydata))
            idx = int(np.argmin(d))
            points.pop(idx)
        redraw()

    def on_key(event):
        if event.key == 'enter':
            plt.close(fig)
        elif event.key == 'q':
            plt.close(fig)
            raise SystemExit(0)
        elif event.key == 'backspace' and points:
            points.pop()
            redraw()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    redraw()
    plt.show()
    return points


def run_annotation(images_dir: Path, gt_path: Path):
    """Run the interactive annotation loop over all assignment images."""
    image_names = ['butterfly.jpg', 'einstein.jpg', 'fishes.jpg', 'sunflowers.jpg']
    gt = load_json(gt_path, {'format': 'part2_manual_points_v1', 'images': {}})
    gt.setdefault('images', {})

    for name in image_names:
        img_path = images_dir / name
        existing = gt['images'].get(name, [])
        pts = annotate_points(img_path, existing)
        gt['images'][name] = pts
        save_json(gt_path, gt)
        print(f'Saved {name}: {len(pts)} points')

    print(f'Annotation complete: {gt_path}')


def bootstrap_from_proxy(proxy_path: Path, gt_path: Path, images_dir: Path, source: str='silver'):
    """Initialize manual GT points from detector output consensus."""
    gt = {'format': 'part2_manual_points_v1', 'source': f'{source}_bootstrap', 'images': {}}
    current_cfg = BlobConfig(
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
    )
    prev_cfg = BlobConfig(
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
    )
    pruned_cfg = BlobConfig(
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
    )

    for image_name in ['butterfly.jpg', 'einstein.jpg', 'fishes.jpg', 'sunflowers.jpg']:
        image = read_gray(images_dir / image_name)
        scale_space, meta = build_scale_space(image, ksize=9, sigma=1.6, n=12, k=1.2)
        if source == 'current':
            circles, _ = detect_from_scale_space(scale_space, meta, current_cfg)
        else:
            det = {}
            for cfg in [current_cfg, prev_cfg, pruned_cfg]:
                circles_cfg, _ = detect_from_scale_space(scale_space, meta, cfg)
                det[cfg.name] = circles_cfg
            circles = synthesize_silver_gt(det, min_votes=2)
        gt['images'][image_name] = [[float(c[0]), float(c[1])] for c in circles]
        gt.setdefault('bootstrap_counts', {})[image_name] = int(circles.shape[0])
    save_json(gt_path, gt)
    print(f'Bootstrapped template: {gt_path}')
    print('Now run annotation mode to add/adjust points.')


def main():
    """CLI entry for bootstrap and annotation modes."""
    parser = argparse.ArgumentParser(description='Manual GT workflow for Part 2 points')
    parser.add_argument('--mode', choices=['annotate', 'bootstrap'], required=True)
    parser.add_argument(
        '--images_dir',
        default='data/part2',
        help='Directory containing part2 images'
    )
    parser.add_argument(
        '--gt_path',
        default='logs/part2_manual_gt_points.json',
        help='Manual GT points JSON file'
    )
    parser.add_argument(
        '--proxy_path',
        default='logs/part2_proxy_eval.json',
        help='Proxy eval JSON (for bootstrap mode)'
    )
    parser.add_argument(
        '--bootstrap_source',
        choices=['silver', 'current'],
        default='silver',
        help='Bootstrap source for initial points'
    )
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    gt_path = Path(args.gt_path)
    proxy_path = Path(args.proxy_path)

    if args.mode == 'bootstrap':
        bootstrap_from_proxy(proxy_path, gt_path, images_dir, source=args.bootstrap_source)
    else:
        run_annotation(images_dir, gt_path)


if __name__ == '__main__':
    main()

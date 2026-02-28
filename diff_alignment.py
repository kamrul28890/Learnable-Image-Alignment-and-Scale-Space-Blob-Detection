"""Implements a differentiable alignmnet alg."""

import csv
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from alignment_model import AlignmentModel
from metrics import ncc, mse

class DiffAlignment(AlignmentModel):
  def __init__(self, image_name, metric='mse', padding='border'):
    super().__init__(image_name, metric, padding)

  def align(self):
    start_time = time.time()
    self.img = self._load_image()
    self.b, self.g, self.r = self._crop_and_divide_image()
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.b = self.b.to(self.device)
    self.g = self.g.to(self.device)
    self.r = self.r.to(self.device)

    # I use green as the reference channel because it is usually the most stable/least noisy plate.
    b_idx, b_shifts, b_debug = self._align_pairs(self.g, self.b, pair_name='B->G')
    print('-------------------')
    r_idx, r_shifts, r_debug = self._align_pairs(self.g, self.r, pair_name='R->G')
    self.b_idx = b_idx
    self.r_idx = r_idx

    # I use differentiable warping for the final composition to avoid circular wrap artifacts.
    self.b_aligned = self._warp_with_shifts(self.b, b_shifts, padding_mode=self.padding)
    self.r_aligned = self._warp_with_shifts(self.r, r_shifts, padding_mode=self.padding)
    self.rgb = torch.stack([self.r_aligned, self.g, self.b_aligned], dim=0).detach().cpu()
    self._log_alignment_run(
      runtime_sec=time.time() - start_time,
      b_idx=b_idx,
      r_idx=r_idx,
      b_debug=b_debug,
      r_debug=r_debug
    )

  def _align_pairs(self, img1, img2, pair_name='pair', max_shift_norm=0.18):
    if self.metric == 'ncc':
      loss_fn = ncc
    elif self.metric == 'mse':
      loss_fn = mse
    else:
      raise ValueError(f'Unsupported metric for differentiable alignment: {self.metric}')

    # I normalize both channels so loss values are less biased by illumination differences.
    fixed = self._normalize_for_loss(img1)
    moving = self._normalize_for_loss(img2)
    fixed_edge = self._sobel_magnitude(fixed)
    moving_edge = self._sobel_magnitude(moving)

    # I run a 3-stage pyramid and start each stage on edge-space before intensity-space.
    stages = [
      {'scale': 0.25, 'steps': 500, 'lr': 0.02, 'edge_ratio': 0.75, 'restarts': 3},
      {'scale': 0.5, 'steps': 700, 'lr': 0.01, 'edge_ratio': 0.55, 'restarts': 1},
      {'scale': 1.0, 'steps': 1200, 'lr': 0.004, 'edge_ratio': 0.30, 'restarts': 1},
    ]
    shifts_init = torch.zeros(2, device=img1.device, dtype=img1.dtype)
    stage_debug = []

    for stage in stages:
      fixed_stage = self._downsample_if_needed(fixed, stage['scale'])
      moving_stage = self._downsample_if_needed(moving, stage['scale'])
      fixed_edge_stage = self._downsample_if_needed(fixed_edge, stage['scale'])
      moving_edge_stage = self._downsample_if_needed(moving_edge, stage['scale'])

      crop_ratio = 0.08 if stage['scale'] == 1.0 else 0.06
      fixed_stage = self._center_crop_by_ratio(fixed_stage, crop_ratio)
      moving_stage = self._center_crop_by_ratio(moving_stage, crop_ratio)
      fixed_edge_stage = self._center_crop_by_ratio(fixed_edge_stage, crop_ratio)
      moving_edge_stage = self._center_crop_by_ratio(moving_edge_stage, crop_ratio)

      restart_inits = [shifts_init]
      if stage['restarts'] > 1:
        seed = self._stable_seed(f'{self.image_name}_{self.metric}_{pair_name}')
        gen = torch.Generator(device='cpu')
        gen.manual_seed(seed)
        for _ in range(stage['restarts'] - 1):
          jitter = (torch.rand(2, generator=gen) - 0.5) * 0.08
          restart_inits.append((shifts_init.detach().cpu() + jitter).to(img1.device, img1.dtype))

      best_restart_loss = float('inf')
      best_restart_shifts = shifts_init
      best_steps_taken = 0
      edge_steps = max(1, int(stage['steps'] * stage['edge_ratio']))
      patience = max(80, int(0.18 * stage['steps']))

      for restart_id, init_shift in enumerate(restart_inits):
        stage_net = AlignNet(
          fixed_stage.size(),
          max_shift_norm=max_shift_norm,
          init_shifts=init_shift
        ).to(img1.device)
        optimizer = torch.optim.AdamW(stage_net.parameters(), lr=stage['lr'])
        best_local_loss = float('inf')
        best_local_shifts = stage_net.get_clamped_shifts().detach()
        no_improve_steps = 0
        steps_taken = stage['steps']

        for k in range(stage['steps']):
          optimizer.zero_grad()
          if k < edge_steps:
            moved = stage_net(moving_edge_stage)
            target = fixed_edge_stage
          else:
            moved = stage_net(moving_stage)
            target = fixed_stage

          align_loss = loss_fn(target, moved)
          reg_loss = 5e-3 * torch.sum(stage_net.get_clamped_shifts() ** 2)
          loss = align_loss + reg_loss
          loss.backward()
          optimizer.step()

          cur_loss = float(align_loss.item())
          if cur_loss < best_local_loss - 1e-7:
            best_local_loss = cur_loss
            best_local_shifts = stage_net.get_clamped_shifts().detach()
            no_improve_steps = 0
          else:
            no_improve_steps += 1

          if k % 300 == 0 or k == stage['steps'] - 1:
            cur_idx = self._normalized_to_pixel_shift(stage_net.get_clamped_shifts(), img1.shape)
            phase = 'edge' if k < edge_steps else 'intensity'
            print(
              f'{pair_name} | scale={stage["scale"]:.2f}, restart={restart_id}, '
              f'phase={phase}, step={k:4d}, shift={cur_idx}, loss={cur_loss:.6f}'
            )

          if k >= edge_steps and no_improve_steps >= patience:
            steps_taken = k + 1
            break

        if best_local_loss < best_restart_loss:
          best_restart_loss = best_local_loss
          best_restart_shifts = best_local_shifts
          best_steps_taken = steps_taken

      shifts_init = best_restart_shifts
      stage_debug.append({
        'scale': stage['scale'],
        'lr': stage['lr'],
        'steps_budget': stage['steps'],
        'steps_taken': best_steps_taken,
        'edge_steps': edge_steps,
        'restarts': stage['restarts'],
        'best_stage_loss': best_restart_loss,
        'shift_px': self._normalized_to_pixel_shift(shifts_init, img1.shape),
      })

    shifts = shifts_init
    align_idx = self._normalized_to_pixel_shift(shifts, img1.shape)
    debug = {
      'pair_name': pair_name,
      'metric': self.metric,
      'shift_px': align_idx,
      'shift_norm': [float(shifts[0].item()), float(shifts[1].item())],
      'stages': stage_debug,
    }
    return align_idx, shifts, debug

  @staticmethod
  def _normalized_to_pixel_shift(shifts, img_shape):
    h, w = img_shape[-2], img_shape[-1]
    # I convert normalized affine translations back to integer pixel shifts.
    dy = -int(torch.round(shifts[1] * (h / 2)).item())
    dx = -int(torch.round(shifts[0] * (w / 2)).item())
    return [dy, dx]

  @staticmethod
  def _normalize_for_loss(img):
    return (img - img.mean()) / (img.std() + 1e-6)

  @staticmethod
  def _stable_seed(text):
    return int(sum(bytearray(text.encode('utf-8')))) % (2 ** 31 - 1)

  @staticmethod
  def _downsample_if_needed(img, scale):
    if scale >= 1.0:
      return img
    return F.interpolate(
      img.unsqueeze(0).unsqueeze(0),
      scale_factor=scale,
      mode='bilinear',
      align_corners=False,
      antialias=True
    )[0, 0]

  @staticmethod
  def _center_crop_by_ratio(img, ratio):
    if ratio <= 0.0:
      return img
    h, w = img.shape
    mh = int(round(h * ratio))
    mw = int(round(w * ratio))
    mh = min(mh, (h - 2) // 2)
    mw = min(mw, (w - 2) // 2)
    if mh <= 0 and mw <= 0:
      return img
    return img[mh:h - mh, mw:w - mw]

  @staticmethod
  def _sobel_magnitude(img):
    kx = torch.tensor(
      [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
      device=img.device,
      dtype=img.dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
      [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
      device=img.device,
      dtype=img.dtype,
    ).view(1, 1, 3, 3)
    x = img.unsqueeze(0).unsqueeze(0)
    gx = F.conv2d(x, kx, padding=1)[0, 0]
    gy = F.conv2d(x, ky, padding=1)[0, 0]
    return torch.sqrt(gx * gx + gy * gy + 1e-8)

  def _log_alignment_run(self, runtime_sec, b_idx, r_idx, b_debug, r_debug):
    os.makedirs('logs', exist_ok=True)
    csv_path = os.path.join('logs', 'part1_alignment_runs.csv')
    json_path = os.path.join(
      'logs',
      f'part1_debug_{os.path.basename(self.image_name).split(".")[0]}_{self.metric}.json'
    )

    row = {
      'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
      'image_name': self.image_name,
      'metric': self.metric,
      'runtime_sec': round(float(runtime_sec), 4),
      'b_shift_dy': int(b_idx[0]),
      'b_shift_dx': int(b_idx[1]),
      'r_shift_dy': int(r_idx[0]),
      'r_shift_dx': int(r_idx[1]),
      'b_final_loss': round(float(b_debug['stages'][-1]['best_stage_loss']), 8),
      'r_final_loss': round(float(r_debug['stages'][-1]['best_stage_loss']), 8),
    }
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
      writer = csv.DictWriter(f, fieldnames=list(row.keys()))
      if write_header:
        writer.writeheader()
      writer.writerow(row)

    debug_payload = {
      'row': row,
      'b_debug': b_debug,
      'r_debug': r_debug,
    }
    with open(json_path, 'w', encoding='utf-8') as f:
      json.dump(debug_payload, f, indent=2)

  @staticmethod
  def _warp_with_shifts(img, shifts, padding_mode='border'):
    theta = torch.eye(2, 3, device=img.device, dtype=img.dtype).unsqueeze(0)
    theta[0, 0, 2] = shifts[0]
    theta[0, 1, 2] = shifts[1]
    img_4d = img.unsqueeze(0).unsqueeze(0)
    grid = F.affine_grid(theta, size=img_4d.size(), align_corners=False)
    warped = F.grid_sample(
      img_4d, grid, mode='bilinear', padding_mode=padding_mode, align_corners=False
    )[0, 0]
    return warped

class AlignNet(nn.Module):
  def __init__(self, img_size, max_shift_norm=0.2, init_shifts=None):
    super(AlignNet, self).__init__()
    self.img_size = img_size
    self.max_shift_norm = max_shift_norm
    # Only have parameters for the shiftsA
    ## Your CODE HERE ##
    # I optimize [tx, ty] directly in normalized image coordinates.
    if init_shifts is None:
      init_shifts = torch.zeros(2, dtype=torch.float32)
    self.shifts = nn.Parameter(init_shifts.detach().clone().float())

  def get_clamped_shifts(self):
    return torch.clamp(self.shifts, -self.max_shift_norm, self.max_shift_norm)

  def forward(self, img):
    ## Your CODE HERE ##
    shifts = self.get_clamped_shifts()
    theta = torch.eye(2, 3, device=img.device, dtype=img.dtype).unsqueeze(0)
    theta[0, 0, 2] = shifts[0]
    theta[0, 1, 2] = shifts[1]
    img_4d = img.unsqueeze(0).unsqueeze(0)
    grid = F.affine_grid(theta, size=img_4d.size(), align_corners=False)
    # I use border padding here to reduce edge artifacts while optimizing the shift.
    transformed_img = F.grid_sample(
      img_4d, grid, mode='bilinear', padding_mode='border', align_corners=False
    )[0, 0]
    return transformed_img

"""Implements a differentiable alignmnet alg."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from alignment_model import AlignmentModel
from metrics import ncc, mse
from utils.misc_helper import custom_shifts

class DiffAlignment(AlignmentModel):
  def __init__(self, image_name, metric='mse', padding='circular'):
    super().__init__(image_name, metric, padding)

  def align(self):
    self.img = self._load_image()
    self.b, self.g, self.r = self._crop_and_divide_image()
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.b = self.b.to(self.device)
    self.g = self.g.to(self.device)
    self.r = self.r.to(self.device)
    lr = 0.005
    steps = 2500
    g_idx, _ = self._align_pairs(self.b, self.g, lr=lr, steps=steps)
    print('-------------------')
    r_idx, _ = self._align_pairs(self.b, self.r, lr=lr, steps=steps)
    self.g_idx = g_idx
    self.r_idx = r_idx
    self.g_aligned = custom_shifts(self.g, g_idx, dims=(0,1), padding=self.padding)
    self.r_aligned = custom_shifts(self.r, r_idx, dims=(0,1), padding=self.padding)
    self.rgb = torch.stack([self.r_aligned, self.g_aligned, self.b], dim=0).detach().cpu()

  def _align_pairs(self, img1, img2, lr=0.1, steps=2000, max_shift_norm=0.2):
    if self.metric == 'ncc':
      loss_fn = ncc
    elif self.metric == 'mse':
      loss_fn = mse
    else:
      raise ValueError(f'Unsupported metric for differentiable alignment: {self.metric}')
    # Create alignment module
    align_net = AlignNet(img1.size(), max_shift_norm=max_shift_norm).to(img1.device)

    # Create optimizer
    optimizer = torch.optim.AdamW(align_net.parameters(), lr=lr)

    # Loss function
    for k in range(steps):
      optimizer.zero_grad()
      img2_shifted = align_net(img2)
      loss = loss_fn(img1, img2_shifted)
      loss.backward()
      optimizer.step()
      if k % 500 == 0 or k == steps - 1:
        shifts = align_net.get_clamped_shifts()
        align_idx = self._normalized_to_pixel_shift(shifts, img1.shape)
        print(f'step={k:4d}, shift={align_idx}, loss={loss.item():.6f}')
    shifts = align_net.get_clamped_shifts()
    align_idx = self._normalized_to_pixel_shift(shifts, img1.shape)
    return align_idx, align_net

  @staticmethod
  def _normalized_to_pixel_shift(shifts, img_shape):
    h, w = img_shape[-2], img_shape[-1]
    # I convert normalized affine translations back to integer pixel shifts.
    dy = -int(torch.round(shifts[1] * (h / 2)).item())
    dx = -int(torch.round(shifts[0] * (w / 2)).item())
    return [dy, dx]

class AlignNet(nn.Module):
  def __init__(self, img_size, max_shift_norm=0.2):
    super(AlignNet, self).__init__()
    self.img_size = img_size
    self.max_shift_norm = max_shift_norm
    # Only have parameters for the shiftsA
    ## Your CODE HERE ##
    # I optimize [tx, ty] directly in normalized image coordinates.
    self.shifts = nn.Parameter(torch.zeros(2, dtype=torch.float32))

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

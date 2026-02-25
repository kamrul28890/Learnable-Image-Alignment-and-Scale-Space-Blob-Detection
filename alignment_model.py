"""Implements the alignment algorithm."""

import torch
import torchvision
from metrics import ncc, mse, ssim
from utils.misc_helper import custom_shifts


class AlignmentModel:
  def __init__(self, image_name, metric='ssim', padding='circular'):
    # Image name
    self.image_name = image_name
    # Metric to use for alignment
    self.metric = metric
    # Padding mode for custom_shifts
    self.padding = padding

  def save(self, output_name):
    torchvision.utils.save_image(self.rgb, output_name)

  def align(self):
    """Aligns the image using the metric specified in the constructor.
       Experiment with the ordering of the alignment.

       Finally, outputs the rgb image in self.rgb.
    """
    self.img = self._load_image()
    self.b, self.g, self.r = self._crop_and_divide_image()
    # I align green/red to blue because blue is the top channel in this dataset format.
    g_idx = self._align_pairs(self.b, self.g, delta=20)
    r_idx = self._align_pairs(self.b, self.r, delta=20)
    self.g_aligned = custom_shifts(self.g, g_idx, dims=(0, 1), padding=self.padding)
    self.r_aligned = custom_shifts(self.r, r_idx, dims=(0, 1), padding=self.padding)
    self.rgb = torch.stack([self.r_aligned, self.g_aligned, self.b], dim=0)

    ## Your alignment code here ##

  def _load_image(self):
    """Load the image from the image_name path,
       typecast it to float, and normalize it.

       Returns: torch.Tensor of shape (H, W)
    """
    ret = None
    ## Your CODE HERE ##
    ret = torchvision.io.read_image(
      self.image_name, mode=torchvision.io.ImageReadMode.GRAY
    ).float() / 255.0
    ret = ret.squeeze(0)

    return ret

  def _crop_and_divide_image(self):
    """Crop the image boundary and divide the image into three parts, padded to the same size.

       Feel free to be creative about this.
       You can eyeball the boundary values, or write code to find approximate cut-offs.
       Hint: Plot out the average values per row / column and visualize it!

       Returns: B, G, R torch.Tensor of shape (roughly H//3, W)
    """
    b_channel = None
    g_channel = None
    r_channel = None 
    ## Your CODE HERE ##
    h, w = self.img.shape
    h3 = h // 3
    # I drop the remainder row so each channel has consistent size.
    stacked = self.img[:3 * h3, :]
    b_channel = stacked[:h3, :]
    g_channel = stacked[h3:2 * h3, :]
    r_channel = stacked[2 * h3:3 * h3, :]

    # I crop small borders to suppress strong frame edges before matching.
    crop_h = min(max(5, int(0.05 * h3)), h3 // 4)
    crop_w = min(max(5, int(0.05 * w)), w // 4)
    if crop_h > 0 and crop_w > 0:
      b_channel = b_channel[crop_h:-crop_h, crop_w:-crop_w]
      g_channel = g_channel[crop_h:-crop_h, crop_w:-crop_w]
      r_channel = r_channel[crop_h:-crop_h, crop_w:-crop_w]


    return b_channel, g_channel, r_channel

  def _align_pairs(self, img1, img2, delta):
    """
    Aligns two images using the metric specified in the constructor.
    Returns: Tuple of (u, v) shifts that minimizes the metric.
    """
    align_idx = (0,0) 
    ## Your CODE HERE ##
    if self.metric == 'ncc':
      loss_fn = ncc
    elif self.metric == 'mse':
      loss_fn = mse
    elif self.metric == 'ssim':
      loss_fn = ssim
    else:
      raise ValueError(f'Unsupported metric: {self.metric}')

    best_loss = float('inf')
    # I brute-force offsets in a local window and keep the lowest loss.
    for dy in range(-delta, delta + 1):
      for dx in range(-delta, delta + 1):
        shifted = custom_shifts(img2, (dy, dx), dims=(0, 1), padding=self.padding)
        loss = loss_fn(img1, shifted)
        if torch.is_tensor(loss):
          loss = loss.item()
        if loss < best_loss:
          best_loss = loss
          align_idx = (dy, dx)

    return align_idx

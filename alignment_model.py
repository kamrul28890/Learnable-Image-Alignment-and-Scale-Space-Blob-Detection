"""Implements the base alignment algorithm using exhaustive search.

This module provides the AlignmentModel class, which loads a Prokudin-Gorsky
tri-plate image, crops film borders, splits into B/G/R channels, and aligns
them via brute-force shift search with a coarse-to-fine image pyramid for
efficiency on high-resolution plates.
"""

import torch
import torch.nn.functional as F
import torchvision
from metrics import ncc, mse, ssim
from utils.misc_helper import custom_shifts


class AlignmentModel:
    """Aligns Prokudin-Gorsky tri-plate images using exhaustive shift search.

    The three grayscale plates (Blue, Green, Red from top to bottom) are
    extracted from a single stacked image, film borders are cropped, and
    each channel is aligned to the Blue plate by searching over a grid of
    pixel displacements and minimising a chosen similarity metric.

    Args:
        image_name (str): Path to the stacked tri-plate JPEG.
        metric (str): Similarity metric — one of 'ncc', 'mse', or 'ssim'.
        padding (str): Padding mode for shifts — 'circular' or 'zero'.
    """

    def __init__(self, image_name: str, metric: str = 'ssim',
                 padding: str = 'circular') -> None:
        self.image_name = image_name
        self.metric = metric
        self.padding = padding
        self.rgb: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def align(self) -> None:
        """Full alignment pipeline.

        Loads the image, crops borders, splits into channels, aligns G and R
        to B using a coarse-to-fine pyramid search, and stores the result in
        self.rgb as a (3, H, W) float tensor in [0, 1].
        """
        self.img = self._load_image()
        self.b, self.g, self.r = self._crop_and_divide_image()

        # Coarse-to-fine pyramid search — handles large displacements efficiently.
        g_shift = self._pyramid_align(self.b, self.g)
        r_shift = self._pyramid_align(self.b, self.r)

        print(f"[{self.image_name}] G shift: {g_shift}, R shift: {r_shift}")

        self.g_aligned = custom_shifts(self.g, g_shift,
                                       dims=(0, 1), padding=self.padding)
        self.r_aligned = custom_shifts(self.r, r_shift,
                                       dims=(0, 1), padding=self.padding)

        # Stack as (R, G, B) — torchvision save_image expects RGB order.
        self.rgb = torch.stack([self.r_aligned, self.g_aligned, self.b], dim=0)

    def save(self, output_name: str) -> None:
        """Save the aligned RGB image to disk.

        Args:
            output_name (str): Destination file path.
        """
        if self.rgb is None:
            raise RuntimeError("Call align() before save().")
        torchvision.utils.save_image(self.rgb, output_name)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_image(self) -> torch.Tensor:
        """Load image from disk, cast to float32, and normalise to [0, 1].

        Returns:
            torch.Tensor: Grayscale image of shape (H, W).
        """
        img = torchvision.io.read_image(
            self.image_name,
            mode=torchvision.io.ImageReadMode.GRAY
        )
        # read_image returns (1, H, W) uint8 — squeeze channel, normalise.
        return img.squeeze(0).float() / 255.0

    def _crop_and_divide_image(self) -> tuple:
        """Crop film borders and split the stacked image into three channels.

        Prokudin-Gorsky plates have thick black film borders that corrupt
        alignment metrics. We crop ~6 % from each edge, then divide the
        remaining image into three equal horizontal strips:
            Top    → Blue channel
            Middle → Green channel
            Bottom → Red channel

        Returns:
            Tuple[Tensor, Tensor, Tensor]: (B, G, R) each of shape (~H//3, W).
        """
        H, W = self.img.shape

        # Remove film borders: 6 % on height edges, 6 % on width edges.
        crop_h = int(H * 0.06)
        crop_w = int(W * 0.06)
        cropped = self.img[crop_h: H - crop_h, crop_w: W - crop_w]

        # Divide into three equal strips.
        ch = cropped.shape[0] // 3
        b_channel = cropped[0:ch, :]
        g_channel = cropped[ch: 2 * ch, :]
        r_channel = cropped[2 * ch: 3 * ch, :]

        return b_channel, g_channel, r_channel

    def _get_metric_fn(self):
        """Return the loss callable corresponding to self.metric."""
        if self.metric == 'ncc':
            return ncc
        elif self.metric == 'mse':
            return mse
        elif self.metric == 'ssim':
            return ssim
        else:
            raise ValueError(f"Unknown metric '{self.metric}'. "
                             "Choose from: ncc, mse, ssim.")

    def _align_pairs(self, img1: torch.Tensor, img2: torch.Tensor,
                     delta: int = 15) -> tuple:
        """Brute-force search for the best integer (row, col) shift.

        Searches all displacements in [-delta, +delta] x [-delta, +delta]
        and returns the one that minimises the chosen metric.

        Args:
            img1 (Tensor): Reference channel, shape (H, W).
            img2 (Tensor): Moving channel, shape (H, W).
            delta (int): Half-window of the search range in pixels.

        Returns:
            Tuple[int, int]: Best (row_shift, col_shift).
        """
        loss_fn = self._get_metric_fn()
        best_loss = float('inf')
        best_shift = (0, 0)

        for dr in range(-delta, delta + 1):
            for dc in range(-delta, delta + 1):
                shifted = custom_shifts(img2, [dr, dc],
                                        dims=(0, 1), padding='circular')
                loss = loss_fn(img1, shifted)
                loss_val = loss.item() if hasattr(loss, 'item') else float(loss)
                if loss_val < best_loss:
                    best_loss = loss_val
                    best_shift = (dr, dc)

        return best_shift

    def _pyramid_align(self, img1: torch.Tensor,
                       img2: torch.Tensor) -> tuple:
        """Coarse-to-fine pyramid alignment for large-displacement images.

        Progressively refines the shift estimate across three resolution
        levels. Coarser levels cover large displacements cheaply; finer
        levels correct the residual with a small local search.

            Level 0 (1/4 res): search ±15 px  → coarse estimate
            Level 1 (1/2 res): search ± 5 px  → medium refinement
            Level 2 (full res): search ± 2 px  → fine correction

        Args:
            img1 (Tensor): Reference channel, shape (H, W).
            img2 (Tensor): Moving channel, shape (H, W).

        Returns:
            Tuple[int, int]: Cumulative (row_shift, col_shift) at full res.
        """
        scales = [4, 2, 1]   # Downscale factors (largest → smallest).
        deltas = [15, 5, 2]  # Search window per level.

        cumulative = [0, 0]

        for scale, delta in zip(scales, deltas):
            # Add batch + channel dims for avg_pool2d.
            t1 = img1.unsqueeze(0).unsqueeze(0)
            t2 = img2.unsqueeze(0).unsqueeze(0)

            if scale > 1:
                t1 = F.avg_pool2d(t1, scale)
                t2 = F.avg_pool2d(t2, scale)

            s1 = t1.squeeze()
            s2 = t2.squeeze()

            # Pre-apply the accumulated shift (scaled to current resolution).
            prev_shift = [cumulative[0] // scale, cumulative[1] // scale]
            s2 = custom_shifts(s2, prev_shift, dims=(0, 1), padding='circular')

            # Find residual shift at this level.
            dr, dc = self._align_pairs(s1, s2, delta=delta)

            # Convert residual back to full-resolution coordinates.
            cumulative[0] += dr * scale
            cumulative[1] += dc * scale

        return (cumulative[0], cumulative[1])

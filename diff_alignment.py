"""Implements a differentiable alignment algorithm using spatial transformers.

This module replaces brute-force shift search with gradient-descent via a
differentiable spatial transformer (affine_grid + grid_sample).

COORDINATE CONVENTIONS (critical for correctness):
----------------------------------------------------
affine_grid uses NORMALISED coordinates in [-1, 1]:
    tx > 0  →  sampling grid shifts RIGHT  →  content moves LEFT in output
    ty > 0  →  sampling grid shifts DOWN   →  content moves UP in output

To convert learned (tx, ty) → pixel shifts for custom_shifts (torch.roll):
    row_shift = -round(ty * H / 2)   [negative because ty>0 moves content UP]
    col_shift = -round(tx * W / 2)   [negative because tx>0 moves content LEFT]

custom_shifts([+dr, +dc]) uses torch.roll:
    +dr → content shifts DOWN  (positive row)
    +dc → content shifts RIGHT (positive col)

KEY DESIGN DECISIONS:
---------------------
1. No clamp inside forward(): clamping kills gradients when params saturate.
   Instead, clamp ONLY at the final extraction step.
2. padding_mode='border': avoids zero-filled regions that destabilise NCC.
3. Higher LR (0.01) + cosine annealing: escapes the flat NCC basin near (0,0).
4. Gradient clipping: prevents explosive updates with NCC loss.
5. Tight clamp at extraction (±0.15 normalised = ±22px for typical image):
   prevents the R channel from jumping to large wrong local minima.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from alignment_model import AlignmentModel
from metrics import ncc, mse
from utils.misc_helper import custom_shifts


# ---------------------------------------------------------------------------
# Device selection — GPU if available, CPU otherwise. Fully transparent.
# ---------------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[DiffAlignment] Running on device: {DEVICE}")


class AlignNet(nn.Module):
    """Differentiable 2-D translation via a spatial transformer.

    Learns a single (tx, ty) pair in normalised coordinates [-1, 1].
    Applies the translation with sub-pixel accuracy via affine_grid +
    grid_sample, which provides smooth, everywhere-nonzero gradients.

    IMPORTANT: We do NOT clamp inside forward() because torch.clamp stops
    gradients from flowing when parameters hit the boundary. Instead, we
    let the parameters move freely during training and only clamp at the
    final pixel-offset extraction step.

    Args:
        img_size (tuple): Shape of the input image (H, W).
    """

    def __init__(self, img_size: tuple) -> None:
        super().__init__()
        self.img_size = img_size  # (H, W)
        # Initialise to zero = identity transform (no shift).
        self.shifts = nn.Parameter(torch.zeros(2))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Apply the learned translation to img.

        Args:
            img (Tensor): Grayscale channel of shape (H, W), on DEVICE.

        Returns:
            Tensor: Translated image of shape (H, W).
        """
        # affine_grid requires (N, C, H, W) — add batch and channel dims.
        img_4d = img.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)

        # shifts[0] = tx (horizontal), shifts[1] = ty (vertical).
        # NO clamp here — clamping kills gradients at the boundary.
        tx = self.shifts[0]
        ty = self.shifts[1]

        # Affine matrix for pure translation:
        #   [[1, 0, tx],
        #    [0, 1, ty]]
        # Must be shape (N, 2, 3) and on the same device as img.
        theta = torch.zeros(1, 2, 3, device=img.device, dtype=img.dtype)
        theta[0, 0, 0] = 1.0   # x-scale (no scaling)
        theta[0, 1, 1] = 1.0   # y-scale (no scaling)
        theta[0, 0, 2] = tx    # x-translation
        theta[0, 1, 2] = ty    # y-translation

        H, W = self.img_size
        grid = F.affine_grid(theta, (1, 1, H, W), align_corners=False)

        # 'border' padding replicates edge pixels instead of filling with
        # zeros. This prevents NCC from being destabilised by large zero
        # regions when channels are far from alignment.
        shifted = F.grid_sample(
            img_4d, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )
        return shifted.squeeze(0).squeeze(0)   # Back to (H, W)


class DiffAlignment(AlignmentModel):
    """Gradient-descent image alignment via a learnable spatial transformer.

    Inherits _load_image, _crop_and_divide_image, and save() from
    AlignmentModel. Overrides align() and _align_pairs() with a differentiable
    optimisation approach using AdamW + cosine annealing.

    Args:
        image_name (str): Path to the stacked tri-plate JPEG.
        metric (str): Loss function — 'mse' or 'ncc'.
        padding (str): Border mode for final roll — 'circular' or 'zero'.
        lr (float): Initial AdamW learning rate.
        n_steps (int): Number of gradient-descent iterations.
    """

    def __init__(self, image_name: str, metric: str = 'mse',
                 padding: str = 'circular', lr: float = 0.01,
                 n_steps: int = 5000) -> None:
        super().__init__(image_name, metric, padding)
        self.lr = lr
        self.n_steps = n_steps

    def align(self) -> None:
        """Full differentiable alignment pipeline."""
        self.img = self._load_image()
        self.b, self.g, self.r = self._crop_and_divide_image()

        # Move to GPU/CPU as appropriate.
        self.b = self.b.to(DEVICE)
        self.g = self.g.to(DEVICE)
        self.r = self.r.to(DEVICE)

        print(f"\n[{self.image_name}] Aligning G → B  (metric={self.metric})")
        g_idx, _ = self._align_pairs(self.b, self.g)

        print(f"\n[{self.image_name}] Aligning R → B  (metric={self.metric})")
        r_idx, _ = self._align_pairs(self.b, self.r)

        print(f"\n[{self.image_name}] Learned shifts — "
              f"G: {g_idx} px,  R: {r_idx} px")

        # Move back to CPU for torch.roll in custom_shifts.
        b_cpu = self.b.cpu()
        g_cpu = self.g.cpu()
        r_cpu = self.r.cpu()

        self.g_aligned = custom_shifts(g_cpu, g_idx,
                                       dims=(0, 1), padding=self.padding)
        self.r_aligned = custom_shifts(r_cpu, r_idx,
                                       dims=(0, 1), padding=self.padding)

        # torchvision.save_image expects (C, H, W) in RGB order.
        self.rgb = torch.stack([self.r_aligned, self.g_aligned, b_cpu], dim=0)

    def _align_pairs(self, img1: torch.Tensor,
                     img2: torch.Tensor) -> tuple:
        """Optimise the translation that aligns img2 onto img1.

        Uses AdamW with a cosine annealing learning rate schedule and gradient
        clipping to robustly minimise the chosen loss over img2's translation.

        Conversion from normalised shift to pixel offset:
            row_shift = -round(ty * H / 2)
            col_shift = -round(tx * W / 2)

        The negative signs arise because in affine_grid:
            ty = +0.1 shifts the sampling grid DOWN → content moves UP in output
        In custom_shifts / torch.roll:
            row_shift = -7 → content moves UP by 7 rows ✓

        Args:
            img1 (Tensor): Reference channel (Blue), on DEVICE, shape (H, W).
            img2 (Tensor): Moving channel (G or R), on DEVICE, shape (H, W).

        Returns:
            Tuple[(int, int), AlignNet]:
                - (row_shift, col_shift) in integer pixels.
                - Trained AlignNet instance (for inspection/logging).
        """
        if self.metric == 'ncc':
            loss_fn = ncc
        elif self.metric == 'mse':
            loss_fn = mse
        else:
            raise ValueError(
                f"DiffAlignment supports 'ncc' and 'mse', got '{self.metric}'."
            )

        H, W = img1.shape
        align_net = AlignNet(img1.shape).to(DEVICE)

        # AdamW with cosine annealing: starts at lr, decays smoothly to 0.
        # This prevents oscillation near convergence while still escaping
        # the flat basin around (0, 0) with a larger initial step size.
        optimizer = torch.optim.AdamW(align_net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_steps, eta_min=1e-5
        )

        for step in range(self.n_steps):
            optimizer.zero_grad()
            img2_shifted = align_net(img2)
            loss = loss_fn(img1, img2_shifted)
            loss.backward()

            # Clip gradients: NCC can produce large gradients near boundaries.
            torch.nn.utils.clip_grad_norm_(align_net.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            if step % 1000 == 0:
                with torch.no_grad():
                    tx_v = align_net.shifts[0].item()
                    ty_v = align_net.shifts[1].item()
                    row_px = -round(ty_v * H / 2)
                    col_px = -round(tx_v * W / 2)
                print(f"  Step {step:5d} | loss: {loss.item():.6f} "
                      f"| shift (row, col): ({row_px}, {col_px})")

        # ----------------------------------------------------------------
        # Extract final pixel shift.
        # Clamp to ±0.15 normalised ≈ ±22 px on a ~300-px-tall plate.
        # This is the expected range for inter-plate displacement and
        # prevents the optimiser from landing at large wrong local minima.
        # Clamping ONLY here (not in forward) so training gradients are free.
        # ----------------------------------------------------------------
        with torch.no_grad():
            tx_final = float(torch.clamp(align_net.shifts[0], -0.15, 0.15))
            ty_final = float(torch.clamp(align_net.shifts[1], -0.15, 0.15))
            row_shift = -round(ty_final * H / 2)
            col_shift = -round(tx_final * W / 2)

        return (row_shift, col_shift), align_net

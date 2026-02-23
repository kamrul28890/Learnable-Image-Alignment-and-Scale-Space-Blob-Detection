"""Image similarity metrics used as loss functions for channel alignment.

All metrics return a SCALAR where lower = better alignment, so they can
be minimised directly by gradient descent.

Functions:
    ncc  — Negative Normalised Cross-Correlation (differentiable).
    mse  — Mean Squared Error (differentiable).
    ssim — Negative Structural Similarity (non-differentiable; for
           exhaustive-search use only).
"""

import torch
from skimage.metrics import structural_similarity


def ncc(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Negative Normalised Cross-Correlation (NCC).

    NCC = sum(a_norm * b_norm), where a_norm = (a - mean(a)) / ||a - mean(a)||.
    Range: [-1, 0]. Perfect alignment = -1.

    This is the ORIGINAL formulation from the starter code, kept because it
    provides a clean, interpretable loss with correct gradient sign.

    Fully differentiable w.r.t. both inputs.

    Args:
        img1 (Tensor): Reference image, any shape.
        img2 (Tensor): Moving image, same shape as img1.

    Returns:
        Tensor: Scalar loss in [-1, 0].
    """
    img1 = img1 - img1.mean()
    img2 = img2 - img2.mean()
    norm1 = torch.linalg.vector_norm(img1).clamp(min=1e-8)
    norm2 = torch.linalg.vector_norm(img2).clamp(min=1e-8)
    return -torch.sum(img1 / norm1 * img2 / norm2)


def mse(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error (MSE).

    MSE = mean((img1 - img2)^2). Range: [0, ∞). Perfect alignment = 0.
    MSE has stronger, more uniform gradients than NCC and is better suited
    for gradient descent when the images are far from alignment.

    Fully differentiable w.r.t. both inputs.

    Args:
        img1 (Tensor): Reference image, any shape.
        img2 (Tensor): Moving image, same shape as img1.

    Returns:
        Tensor: Scalar loss ≥ 0.
    """
    return torch.mean((img1 - img2) ** 2)


def ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Negative Structural Similarity Index (SSIM).

    SSIM jointly measures luminance, contrast, and structure. Range: [-1, 0].
    NOT differentiable — use only with exhaustive search (AlignmentModel),
    not with gradient descent (DiffAlignment).

    Reference:
        Wang et al. "Image quality assessment: from error visibility to
        structural similarity." IEEE TIP, 2004.

    Args:
        img1 (Tensor): 2-D grayscale reference image (H, W).
        img2 (Tensor): 2-D grayscale moving image (H, W).

    Returns:
        float: Negative SSIM in [-1, 0].
    """
    a = img1.detach().numpy()
    b = img2.detach().numpy()
    data_range = float(a.max() - b.min())
    return -structural_similarity(a, b, data_range=data_range)

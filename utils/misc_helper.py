"""Miscellaneous helper functions for image processing."""

import torch


def custom_shifts(
    input: torch.Tensor,
    shifts: list | tuple,
    dims: tuple = (0, 1),
    padding: str = 'circular'
) -> torch.Tensor:
    """Shift a 2-D tensor by (dr, dc) pixels with configurable border padding.

    Args:
        input (Tensor): 2-D image tensor of shape (H, W).
        shifts (list | tuple): [row_shift, col_shift] — positive values shift
            the content down / right respectively.
        dims (tuple): Dimensions to shift along. Default: (0, 1) for (H, W).
        padding (str): Border handling mode.
            - 'circular': Wrap-around (torch.roll behaviour). Useful when the
              plate borders are similar across channels.
            - 'zero': Fill exposed borders with 0. More realistic for alignment
              but slightly penalises boundary regions in the loss.

    Returns:
        Tensor: Shifted image, same shape as input.
    """
    ret = torch.roll(input, shifts, dims)

    if padding == 'zero':
        dr, dc = shifts[0], shifts[1]
        # Zero out the rows/columns that were wrapped around.
        if dr > 0:
            ret[:dr, :] = 0
        elif dr < 0:
            ret[dr:, :] = 0
        if dc > 0:
            ret[:, :dc] = 0
        elif dc < 0:
            ret[:, dc:] = 0

    return ret



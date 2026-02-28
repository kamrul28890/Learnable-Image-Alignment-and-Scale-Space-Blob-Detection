"""I/O helpers for reading and saving tensors as images."""

from torch import Tensor
import torchvision

def torch_read_image(filename: str, gray=True) -> Tensor:
    """Read an image from disk and normalize pixel values to [0, 1]."""
    if gray:
        mode = torchvision.io.ImageReadMode.GRAY
    else:
        mode = torchvision.io.ImageReadMode.UNCHANGED
    return torchvision.io.read_image(
        filename, mode) / 255.

def torch_save_image(image: Tensor, filename: str) -> None:
    """Save a tensor image to disk."""
    torchvision.utils.save_image(image, filename)

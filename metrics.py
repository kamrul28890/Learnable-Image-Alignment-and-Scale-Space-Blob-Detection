"""Implements different image metrics."""

import torch
from skimage.metrics import structural_similarity

def ncc(img1, img2):
  """Takes two image and compute the negative normalized cross correlation.
     Lower the value, better the alignment.
  """
  img1 = img1 - img1.mean()
  img2 = img2 - img2.mean()
  img1_norm = torch.linalg.vector_norm(img1)
  img2_norm = torch.linalg.vector_norm(img2)
  return -torch.sum(img1/img1_norm * img2/img2_norm)

def mse(img1, img2): 
  """Takes two image and compute the mean squared error.
     Lower the value, better the alignment.
  """
  return torch.mean((img1 - img2)**2)

def ssim(img1, img2):
  """Takes two image and compute the negative structural similarity.

  This function is given to you, nothing to do here.

  Please refer to the classic paper by Wang et al. of Image quality 
  assessment: from error visibility to structural similarity.
  """
  img1 = img1.numpy()
  img2 = img2.numpy()
  return -structural_similarity(img1, img2, data_range=img1.max() - img2.min())
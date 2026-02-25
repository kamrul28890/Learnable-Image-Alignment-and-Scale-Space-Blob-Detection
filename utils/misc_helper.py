"""Implements Misc helper functions."""

import torch

def custom_shifts(input, shifts, dims=None, padding='circular'):
  """Shifts the input tensor by the specified shifts along the specified dimensions.
     Supports circular and zero padding.
  """
  ret = torch.roll(input, shifts, dims)
  if padding == 'zero':
    ret[:shifts[0], :shifts[1]] = 0
  return ret 



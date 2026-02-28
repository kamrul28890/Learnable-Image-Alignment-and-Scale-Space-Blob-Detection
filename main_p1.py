"""Implements alignment algorithm."""

# Implements argparse to parse command line arguments.
import argparse
import os
from diff_alignment import DiffAlignment

def main():
  """Main function to run the alignment model."""
  parser = argparse.ArgumentParser(description='CS59300CVD Assignment 2 Part 1')
  parser.add_argument('-i', '--image_name', required=True, type=str, help='Input image path')
  parser.add_argument('-m', '--metric', default='ncc', type=str, help='Metric to use for alignment')
  args = parser.parse_args()
  print(args.image_name)
  if args.image_name == 'all':
    run_all(metric=args.metric)
    return

  model = DiffAlignment(args.image_name, metric=args.metric)
  model.align()
  model.save('outputs/%s_%s_aligned_DIFF.png' % (args.image_name.split('.')[0], args.metric))

def run_all(metric=None):
  """Run the alignment model on all images."""
  os.makedirs('outputs', exist_ok=True)
  metrics = ['ncc', 'mse'] if metric is None else [metric]
  for metric in metrics:
    for image_id in range(1, 7):
      image_name = 'data/part1/%d.jpg' % image_id
      model = DiffAlignment(image_name, metric=metric)
      model.align()
      model.save('outputs/%d_%s_aligned_DIFF.png' % (image_id, metric))

if __name__ == '__main__':
  main()
  

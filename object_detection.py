import os, sys
from copy import deepcopy
import math
import numpy as np
from scipy.stats import norm
from dataparser import Dataloader
import matplotlib.pyplot as plt
from skimage import measure, color


def objects(grays):
  '''
  Takes a list of three grayscale images.
  Outputs a single numpy array representing a binary image, in which clusters of
  1s are candidate moving objects.
  '''
  g1, g2, g3 = grays
  # difference
  dif1, dif2 = np.abs(g1-g2), np.abs(g2-g3)
  # mean
  m1, m2 = np.mean(dif1), np.mean(dif2)
  # threshold
  th1, th2 = -math.log(0.05) * m1, -math.log(0.05) * m2
  # mask
  b1, b2 = dif1 > th1, dif2 > th2
  # intersection
  return np.logical_and(b1, b2)


def region_growing_v2(gray, binary):
  '''
  Implementation a new region growing function that is applied to centroids of candidate clusters
  '''
  binary = deepcopy(binary)
  height, width = gray.shape
  labeled_image = measure.label(binary, background=0)
  blobs = measure.regionprops(labeled_image)
  
  for blob in blobs:
    ctr_row, ctr_col = blob.centroid
    ctr_row, ctr_col = int(ctr_row), int(ctr_col)
    if ctr_row - 5 >= 0 and ctr_row + 5 < height and ctr_col - 5 >= 0 and ctr_col + 5 < width: 
      gray_region = gray[ctr_row - 5:ctr_row + 6, ctr_col - 5:ctr_col + 6]
      
      mean = np.mean(gray_region)
      sd = np.std(gray_region)
      t1 = norm.ppf(0.005, loc=mean, scale=sd)
      t2 = norm.ppf(0.995, loc=mean, scale=sd)
      
      new_bin_region = np.logical_or((gray_region > t1 and gray_region < t2), 
                                      binary[ctr_row - 5:ctr_row + 6, ctr_col - 5:ctr_col + 6])
      binary[ctr_row - 5:ctr_row + 6, ctr_col - 5:ctr_col + 6] = new_bin_region

  return binary
      

if __name__ == '__main__':

  path = sys.argv[1] # path to and including mot

  # TESTING
  current_dir = os.path.dirname(os.path.realpath(__file__))
  os.chdir(current_dir)
  dataloader = Dataloader(f'{path}/car/001', img_file_pattern='*.jpg', frame_range=(1, 100))
  
  frames = list(dataloader.preloaded_frames.values())[1:4]
  
  gs = [color.rgb2gray(img) for _, img, _ in frames]
  plt.imshow(gs[1], cmap='gray')

  b = objects(gs)
  plt.figure()
  plt.imshow(b, cmap='gray')

  b = region_growing_v2(gs[1], b)
  plt.figure()
  plt.imshow(b, cmap='gray')

  plt.show(block=True)

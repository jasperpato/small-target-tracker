import os, sys
from copy import deepcopy
import math
import numpy as np
from scipy.stats import norm
from dataparser import Dataloader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import measure, color

def objects(grays):
  '''
  Takes a list of three grayscale images.
  Outputs a single numpy array representing a binary image, in which clusters of
  1s are candidate moving objects.
  '''
  g1, g2, g3 = grays
  height, width = grays[1].shape
  binary = np.zeros((height, width), dtype=np.uint8)
  
  # misses the interval between the last multiple of 30 and the width and height
  for i in range(0, height, 30):
    for j in range(0, width, 30):  
      g1_region = g1[i:i+30, j:j+30]
      g2_region = g2[i:i+30, j:j+30]
      g3_region = g3[i:i+30, j:j+30]
      
      # difference
      dif1, dif2 = np.abs(g1_region - g2_region), np.abs(g2_region - g3_region)
      # mean
      m1, m2 = np.mean(dif1), np.mean(dif2)
      # threshold
      th1, th2 = -math.log(0.05) * m1, -math.log(0.05) * m2
      # mask
      b1, b2 = dif1 > th1, dif2 > th2
      # intersection
      binary[i:i+30, j:j+30] = np.logical_and(b1, b2)
  
  return binary


def region_growing(gray, binary):
  '''
  Implementation a new region growing function that is applied to centroids of candidate clusters
  '''
  binary = deepcopy(binary)
  height, width = gray.shape
  labeled_image = measure.label(binary, background=0)
  blobs = measure.regionprops(labeled_image)

  for blob in blobs:
    blob_rows, blob_cols = zip(*blob.coords)
    if len(blob.coords) < 10:
      binary[blob_rows, blob_cols] = 0
      continue

    ctr_row, ctr_col = blob.centroid
    ctr_row, ctr_col = int(ctr_row), int(ctr_col)
    
    # 11 x 11 box bounds
    l = ctr_col - 5 if ctr_col - 5 >= 0 else 0
    r = ctr_col + 5 if ctr_col + 5 < width else width
    t = ctr_row - 5 if ctr_row - 5 >= 0 else 0
    b = ctr_row + 5 if ctr_row + 5 < height else height

    gray_region = gray[l:r, b:t]
    blob_grays = gray[blob_rows, blob_cols]
      
    mean = np.mean(blob_grays)
    sd = np.std(blob_grays)
    t1 = norm.ppf(0.1, loc=mean, scale=sd)
    t2 = 2 * mean - t1
    binary[l:r, b:t] = np.logical_or(np.logical_and(gray_region > t1, gray_region < t2), binary[l:r, b:t])

  return binary
      

if __name__ == '__main__':

  # dataset_path = sys.argv[1].rstrip('/')
  dataset_path = '/home/allenator/UWA/fourth_year/CITS4402/VISO/mot'

  # TESTING
  print(f'{dataset_path}/car/001')
  dataloader = Dataloader(f'{dataset_path}/car/001', img_file_pattern='*.jpg', frame_range=(1, 100))
  frames = list(dataloader.preloaded_frames.values())
  i0 = 10
  
  for i in range(i0, len(frames) - i0):
    f1 = frames[i - i0]
    f2 = frames[i]
    f3 = frames[i + i0]
  
    gs = (color.rgb2gray(f1[1]), color.rgb2gray(f2[1]), color.rgb2gray(f3[1]))
    # plt.imshow(f2[1], cmap='gray')

    b = objects(gs)
    fig, ax = plt.subplots()
    plt.imshow(b, cmap='gray')

    b = region_growing(gs[1], b)
    fig2, ax2 = plt.subplots()
    ax2.imshow(b, cmap='gray')

    for box in f2[2]:
      rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='r', facecolor='none')
      rect2 = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='r', facecolor='none')
      ax.add_patch(rect)
      ax2.add_patch(rect2)

    plt.show(block=True)
    break

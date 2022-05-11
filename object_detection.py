import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from copy import deepcopy
from scipy.stats import norm
from scipy import signal
from dataparser import Dataloader
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
  labeled_image = measure.label(binary, background=0, connectivity=2)
  blobs = measure.regionprops(labeled_image)

  for blob in blobs:
    blob_rows, blob_cols = zip(*blob.coords)
    ctr_row, ctr_col = int(blob.centroid[0]), int(blob.centroid[1])
    blob_grays = gray[blob_rows, blob_cols]
    mean = np.mean(blob_grays)
    sd = np.std(blob_grays)
    
    if not sd: continue

    if len(blob.coords) < 3:
      binary[blob_rows, blob_cols] = 0
      continue
      
    t1 = norm.ppf(5e-3, loc=mean, scale=sd)
    t2 = 2 * mean - t1  
    if t2 - t1 > 0.2: continue
    
    # 11 x 11 box bounds
    l = ctr_col - 5 if ctr_col - 5 >= 0 else 0
    r = ctr_col + 6 if ctr_col + 5 < width else width
    b = ctr_row - 5 if ctr_row - 5 >= 0 else 0
    t = ctr_row + 6 if ctr_row + 5 < height else height
    
    gray_region = gray[b:t, l:r]
    candidates = np.logical_and(gray_region > t1, gray_region < t2)
    binary[b:t, l:r] = np.logical_or(candidates, binary[b:t, l:r])
    
  return binary
    
      
if __name__ == '__main__':

  index = 1 if len(sys.argv) == 2 else 2
  dataset_path = sys.argv[index].rstrip('/')
  # dataset_path = '/home/allenator/UWA/fourth_year/CITS4402/VISO/mot'

  # TESTING
  loader = Dataloader(f'{dataset_path}/car/001', img_file_pattern='*.jpg', frame_range=(1, 100))
  preloaded_frames = list(loader.preloaded_frames.values())
  i0 = 10
  
  for i in range(i0, len(preloaded_frames) - i0):
    frames = [preloaded_frames[i+j*i0] for j in (-1,0,1)]
    grays = [color.rgb2gray(f[1]) for f in frames]
    
    plt.figure()
    plt.imshow(grays[1], cmap='gray')

    b = objects(grays)
    _, ax1 = plt.subplots()
    ax1.imshow(b, cmap='gray')

    grown = region_growing(grays[1], b)
    _, ax2 = plt.subplots()
    ax2.imshow(grown, cmap='gray')

    if sys.argv[1] == '--boxes':
      for box in frames[1][2]:
        ax1.add_patch(patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='r', facecolor='none'))
        ax2.add_patch(patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='r', facecolor='none'))

    plt.show(block=True)
    break
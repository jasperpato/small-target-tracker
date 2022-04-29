import os
from copy import deepcopy
import math
import numpy as np
from scipy.stats import norm
from dataparser import Dataloader
import matplotlib.pyplot as plt
from skimage import measure


def frame2gray(frame):
  '''
  Input: list of frame tuples with numpy img array as second element
  Output: list of grayscale numpy img arrays
  '''
  return np.round(np.dot(frame[1][...,:3], [0.2989, 0.5870, 0.1140]))


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
  binary = deepcopy(binary)
  height, width = gray.shape
  labeled_image = measure.label(binary, background=0)
  blobs = measure.regionprops(labeled_image)
  
  for blob in blobs:
    ctr_row, ctr_col = blob.centroid
    if ctr_row - 5 >= 0 and ctr_row + 5 < height and ctr_col - 5 >= 0 and ctr_col + 5 > width: 
      gray_region = gray[ctr_row - 5:ctr_row + 6, ctr_col - 5:ctr_col + 6]
      
      mean = np.mean(gray_region)
      sd = np.std(gray_region)
      t1 = norm.ppf(0.005, loc=mean, scale=sd)
      t2 = norm.ppf(0.995, loc=mean, scale=sd)
      
      new_bin_region = np.bitwise_or((gray_region > t1 and gray_region < t2), 
                                      binary[ctr_row - 5:ctr_row + 6, ctr_col - 5:ctr_col + 6])
      binary[ctr_row - 5:ctr_row + 6, ctr_col - 5:ctr_col + 6] = new_bin_region
  return binary
      
        
def region_growing(gray, binary):
  '''
  Input: a grayscale image and corresponding binary image containing moving candidate pixels
  Output: new binary image with region growing applied

  This function currently applies region growing to every candidate pixel in binary,
  but it should apply it only to th centroid of candidate clusters. Unsure how to find centroids yet.
  '''
  bin = deepcopy(binary)

  # for each pixel
  for i in range(len(gray)):
    for j in range(len(gray[i])):
      if binary[i][j]:

        # calculate mean
        sum, num = 0, 0
        # for each surround pixel in 11x11 box
        for p in range(i-5, i+6):
          for q in range(j-5, j+6):
            # if pixel is inside image bounds
            if not (p < 0 or p >= len(gray) or q < 0 or q >= len(gray[i])): 
              sum += gray[p][q]
              num += 1
        mean = sum / num

        # calculate SD
        sum, num = 0, 0
        for p in range(i-5, i+6):
          for q in range(j-5, j+6):
            if not (p < 0 or p >= len(gray) or q < 0 or q >= len(gray[i])): 
              sum += (gray[p][q] - mean) ** 2
              num += 1
        sd = (sum / num) ** 0.5

        # get thresholds from ppf
        t1, t2 = 0, 0
        if sd:
          ppf = norm.ppf(0.95, loc=mean, scale=sd)
          t1, t2 = (2 * mean - ppf, ppf)
        else: t1, t2 = mean, mean

        # apply thresholding to grow regions in binary image
        for p in range(i-5, i+6):
          for q in range(j-5, j+6):
            if not (p < 0 or p >= len(gray) or q < 0 or q >= len(gray[i])):
              # if grey level pixel is in quantile, set to 1
              if gray[p][q] >= t1 and gray[p][q] <= t2: bin[p][q] = True 

      # if not i % 100 and not j % 100: print(i,j) 

  return bin


if __name__ == '__main__':

    # TESTING
    current_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(current_dir)
    dataloader = Dataloader('/Users/jasperpaterson/Local/object-tracking/car/001', img_file_pattern='*.jpg', frame_range=(1, 100))
    
    frames = list(dataloader.preloaded_frames.values())[:3]
    
    gs = [frame2gray(f) for f in frames]
    plt.imshow(gs[1], cmap='gray')

    b = objects(gs)
    plt.figure()
    plt.imshow(b, cmap='gray')

    b = region_growing(gs[1], b)
    plt.figure()
    plt.imshow(b, cmap='gray')

    plt.show(block=True)

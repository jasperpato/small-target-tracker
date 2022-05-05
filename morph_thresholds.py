import os, sys
from copy import deepcopy
import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import measure, color

from dataparser import Dataloader
from object_detection import objects, region_growing

g
def plot_morph_cues(binary, gt):
  '''
  Plot morphological features for true positives and false positives
  gt list of ground truths bounding box data
  '''
  binary = deepcopy(binary)
  labeled_image = measure.label(binary, background=0)
  blobs = measure.regionprops(labeled_image)

  for blob in blobs:
    area = blob.area_filled
    extent = area / blob.area_bbox
    a_len = blob.axis_major_length
    ecc = blob.eccentricity

    # find corresponding bounding box 
    for t, l, w, h in gt:
      # compute intersection
      pass


if __name__ == '__main__':

  dataset_path = sys.argv[1].rstrip('/')
  dataloader = Dataloader(f'{dataset_path}/car/001', img_file_pattern='*.jpg', frame_range=(1, 100))
  frames = list(dataloader.preloaded_frames.values())
  i0 = 5
  
  imgs = (frames[0], frames[i0], frames[2 * i0])
  grays = [color.rgb2gray(i[1]) for i in imgs]
    
  binary = objects(grays)
  grown = region_growing(grays[1], binary)
  
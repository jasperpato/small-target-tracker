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
from evaluation import intersection_over_union


def plot_morph_cues(binary, gt, ax):
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
    bbox = blob.bbox

    # find closest bounding box 
    max_iou = 0
    for g in gt:
      # l, t, w, h = g
      # # compute intersection
      # in_b = max(t-h, bbox[0])
      # in_t = min(t, bbox[2])
      # in_l = max(l, bbox[1])
      # in_r = min(l+w, bbox[3])
      # in_area = (in_t - in_b) * (in_r - in_l)
      # iou = in_area / (h * w + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) - in_area)

      iou = intersection_over_union(g, (bbox[1], bbox[0], bbox[3]-bbox[1], bbox[2]-bbox[0]))
      if iou > max_iou: max_iou = iou

    if max_iou > 0.5:
      ax.add_patch(patches.Rectangle((bbox[1], bbox[0]), bbox[3]-bbox[1], bbox[2]-bbox[0], linewidth=1, edgecolor='b', facecolor='none'))
      print(f'bbox {bbox} area {area} extent {extent:0.2f} a_len {a_len:0.2f} ecc {ecc:0.2f} iou {max_iou:0.2f}')



if __name__ == '__main__':

  dataset_path = sys.argv[1].rstrip('/')
  dataloader = Dataloader(f'{dataset_path}/car/001', img_file_pattern='*.jpg', frame_range=(1, 100))
  frames = list(dataloader.preloaded_frames.values())
  i0 = 5
  
  imgs = (frames[0], frames[i0], frames[2 * i0])
  grays = [color.rgb2gray(i[1]) for i in imgs]
    
  binary = objects(grays)
  grown = region_growing(grays[1], binary)

  fig, ax = plt.subplots()
  ax.imshow(grown, cmap='gray')

  plot_morph_cues(grown, frames[i0][2], ax)
  for box in imgs[1][2]:
    ax.add_patch(patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='r', facecolor='none'))
  plt.show(block=True)
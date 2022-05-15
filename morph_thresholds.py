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
from evaluation import intersection_over_union, Box


def morph_cues(binary, gt_boxes, iou_threshold=0.5):
  '''
  Plot morphological features for true positives and false positives
  gt list of ground truths bounding box data
  '''

  binary = deepcopy(binary)
  labeled_image = measure.label(binary, background=0, connectivity=1)
  blobs = measure.regionprops(labeled_image)
  
  npos = len(gt_boxes)
  seen_gts = np.zeros(npos)
  tp_areas = []
  tp_extents = []
  tp_a_lengths = []
  tp_eccentricities = []

  for blob in blobs:
    tl_x, tl_y, w, h = blob.bbox[1], blob.bbox[0], blob.bbox[2]-blob.bbox[0], blob.bbox[3]-blob.bbox[1]
    
    max_iou = 0.0
    for i in range(npos):
      iou = intersection_over_union(Box(tl_x, tl_y, w, h), Box(*gt_boxes[i]))
      if iou > max_iou:
        max_iou = iou
        gt_idx = i  # index of the ground truth box with the highest IoU
    
    if max_iou >= iou_threshold and seen_gts[gt_idx] == 0:
      seen_gts[gt_idx] = 1    # mark as detected
      tp_areas.append(blob.area_filled)
      tp_extents.append(blob.area_filled / blob.area_bbox)
      tp_a_lengths.append(blob.axis_major_length)
      tp_eccentricities.append(blob.eccentricity)
      # ax.add_patch(patches.Rectangle((tl_x, tl_y), w, h, linewidth=1, edgecolor='b', facecolor='none'))

  area_avg, area_sd = np.mean(tp_areas) if tp_areas else -1, np.std(tp_areas) if tp_areas else -1
  ext_avg, ext_sd = np.mean(tp_extents) if tp_extents else -1, np.std(tp_extents) if tp_extents else -1
  alen_avg, alen_sd = np.mean(tp_a_lengths) if tp_a_lengths else -1, np.std(tp_a_lengths) if tp_a_lengths else -1
  ecc_avg, ecc_sd = np.mean(tp_eccentricities) if tp_eccentricities else -1, np.std(tp_eccentricities) if tp_eccentricities else -1

  return len(tp_areas), area_avg, area_sd, ext_avg, ext_sd, alen_avg, alen_sd, ecc_avg, ecc_sd


if __name__ == '__main__':

  dataset_path = sys.argv[1].rstrip('/')
  dataloader = Dataloader(f'{dataset_path}/car/001', img_file_pattern='*.jpg', frame_range=(1, 100))
  frames = list(dataloader.preloaded_frames.values())
  print(len(frames))
  step = 5

  for i in range(0, len(frames)-2*step+1, step):
  
    imgs = (frames[i], frames[i+step], frames[i+2*step])
    grays = [color.rgb2gray(i[1]) for i in imgs]
      
    binary = objects(grays)
    grown = region_growing(grays[1], binary)

    print(i+step, list(np.round(morph_cues(grown, frames[step][2], 0.4),2)))
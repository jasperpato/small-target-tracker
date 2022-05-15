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
    pred_box = Box(xtl=blob.bbox[1], ytl=blob.bbox[0],
                   w=blob.bbox[3] - blob.bbox[1],
                   h=blob.bbox[2] - blob.bbox[0])
    max_iou = 0.0
    for i in range(npos):
      gt_box = Box(*gt_boxes[i])
      iou = intersection_over_union(pred_box, gt_box)
      if iou > max_iou:
        max_iou = iou
        gt_idx = i  # index of the ground truth box with the highest IoU

    if max_iou >= iou_threshold and seen_gts[gt_idx] == 0:
      seen_gts[gt_idx] = 1    # mark as detected
      tp_areas.append(blob.area_filled)
      tp_extents.append(blob.area_filled / blob.area_bbox)
      tp_a_lengths.append(blob.axis_major_length)
      tp_eccentricities.append(blob.eccentricity)
      ax.add_patch(patches.Rectangle((pred_box.xtl - 0.5, pred_box.ytl - 0.5),
                                     pred_box.xbr - pred_box.xtl + 1,
                                     pred_box.ybr - pred_box.ytl + 1,
                                     linewidth=1, edgecolor='g', facecolor='none'))


if __name__ == '__main__':

  dataset_path = sys.argv[1].rstrip('/')
  dataloader = Dataloader(f'{dataset_path}/car/001', img_file_pattern='*.jpg', frame_range=(1, 100))
  frames = list(dataloader.preloaded_frames.values())
  i0 = 10

  imgs = (frames[0], frames[i0], frames[2 * i0])
  grays = [color.rgb2gray(i[1]) for i in imgs]

  binary = objects(grays)
  grown = region_growing(grays[1], binary)

  fig, ax = plt.subplots()
  ax.imshow(grown, cmap='gray')
  
  for box in imgs[1][2]:
    ax.add_patch(patches.Rectangle((box[0] - 0.5, box[1] - 0.5),
                                   box[2], box[3], linewidth=1,
                                   edgecolor='y', facecolor='none'))

  morph_cues(grown, imgs[1][2], ax)
  plt.show(block=True)

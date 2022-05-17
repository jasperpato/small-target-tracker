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
      # ax.add_patch(patches.Rectangle((tl_x, tl_y), w, h, linewidth=1, edgecolor='b', facecolor='none'))

  area_avg, area_std = np.mean(tp_areas) if tp_areas else -1, np.std(tp_areas) if tp_areas else -1
  ext_avg, ext_std = np.mean(tp_extents) if tp_extents else -1, np.std(tp_extents) if tp_extents else -1
  alen_avg, alen_std = np.mean(tp_a_lengths) if tp_a_lengths else -1, np.std(tp_a_lengths) if tp_a_lengths else -1
  ecc_avg, ecc_std = np.mean(tp_eccentricities) if tp_eccentricities else -1, np.std(tp_eccentricities) if tp_eccentricities else -1

  return len(tp_areas), area_avg, area_std, ext_avg, ext_std, alen_avg, alen_std, ecc_avg, ecc_std


def find_thresholds(dataset_path, num_folders, num_frames):
  area_avg, ext_avg, alen_avg, ecc_avg = 0, 0, 0, 0
  area_std, ext_std, alen_std, ecc_std = 0, 0, 0, 0
  total_cands = 0

  for j in range(1,num_folders+1):
    print(f'Folder {j:03}')

    dataloader = Dataloader(f'{dataset_path}/car/{j:03}', img_file_pattern='*.jpg', frame_range=(1,num_frames))
    frames = list(dataloader.preloaded_frames.values())
    step = 5

    for i in range(step-1, len(frames)-step+1, step):

      grays = [ color.rgb2gray(f[1]) for f in (frames[i-step+1], frames[i], frames[i+step-1]) ]
        
      binary = objects(grays)
      grown = region_growing(grays[1], binary)

      ncands, ar_avg, ar_std, ex_avg, ex_std, al_avg, al_std, ec_avg, ec_std = morph_cues(binary, frames[i][2], 0.4)

      # cumulative average
      area_avg = (area_avg * total_cands + ar_avg * ncands) / (total_cands + ncands)
      ext_avg = (ext_avg * total_cands + ex_avg * ncands) / (total_cands + ncands)
      alen_avg = (alen_avg * total_cands + al_avg * ncands) / (total_cands + ncands)
      ecc_avg = (ecc_avg * total_cands + ec_avg * ncands) / (total_cands + ncands)

      # cumulative sd
      area_std = ((area_std ** 2 * total_cands + ar_std ** 2 * ncands) / (total_cands + ncands)) ** 0.5
      ext_std = ((ext_std ** 2 * total_cands + ex_std ** 2 * ncands) / (total_cands + ncands)) ** 0.5
      alen_std = ((alen_std ** 2 * total_cands + al_std ** 2 * ncands) / (total_cands + ncands)) ** 0.5
      ecc_std = ((ecc_std ** 2 * total_cands + ec_std ** 2 * ncands) / (total_cands + ncands)) ** 0.5

      total_cands += ncands

  print(area_avg, area_std, ext_avg, ext_std, alen_avg, alen_std, ecc_avg, ecc_std)

  with open('cue_results.txt', 'w') as f:
    f.write(f'{num_folders}, {num_frames}, {area_avg}, {area_std}, {ext_avg}, {ext_std}, {alen_avg}, {alen_std}, {ecc_avg}, {ecc_std}')


def cue_filtering(binary, range=0.99):
  with open('cue_results.txt', 'r') as f:
    area_avg, area_std, ext_avg, ext_std, alen_avg, alen_std, ecc_avg, ecc_std = [float(n) for n in f.read().split(',')[2:]] # 17.694700095207864, 6.344815406670817, 0.652458410014255, 0.10566117930359069, 6.437163565738847, 1.5565604896009309, 0.7698992306070194, 0.13094394850836155
  print(area_avg, ecc_std)

  binary = deepcopy(binary)
  labeled_image = measure.label(binary, background=0, connectivity=1)
  blobs = measure.regionprops(labeled_image)

  t1_area = norm.ppf((1-range)/2, loc=area_avg, scale=area_std)
  t2_area = 2 * area_avg - t1_area
  t1_ext = norm.ppf((1-range)/2, loc=ext_avg, scale=ext_std)
  t2_ext = 2 * ext_avg - t1_ext
  t1_alen = norm.ppf((1-range)/2, loc=alen_avg, scale=alen_std)
  t2_alen = 2 * alen_avg - t1_alen
  t1_ecc = norm.ppf((1-range)/2, loc=ecc_avg, scale=ecc_std)
  t2_ecc = 2 * ecc_avg - t1_ecc

  for blob in blobs:
    blob_rows, blob_cols = zip(*blob.coords)
    if blob.area_filled < t1_area or blob.area_filled > t2_area: binary[blob_rows, blob_cols] = 0
    if blob.extent < t1_ext or blob.extent > t2_ext: binary[blob_rows, blob_cols] = 0
    if blob.axis_major_length < t1_alen or blob.axis_major_length > t2_alen: binary[blob_rows, blob_cols] = 0
    if blob.eccentricity < t1_ecc or blob.eccentricity > t2_ecc: binary[blob_rows, blob_cols] = 0

  return binary


if __name__ == '__main__':

  dataset_path = sys.argv[1].rstrip('/')
  num_folders = int(sys.argv[2])
  num_frames = int(sys.argv[3])

  # find_thresholds(dataset_path, num_folders, num_frames)

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

    filtered = cue_filtering(grown)
    _, ax3 = plt.subplots()
    ax3.imshow(filtered, cmap='gray')

    for box in frames[1][2]:
      # ax1.add_patch(patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='r', facecolor='none'))
      ax3.add_patch(patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='r', facecolor='none'))

    plt.show(block=True)
    break

  
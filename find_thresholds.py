import os
import sys
import numpy as np
import argparse

from scipy.stats import norm
from skimage import measure, color

from dataparser import Dataloader
from object_detection import objects, grow
from evaluation import intersection_over_union, Box


parser = argparse.ArgumentParser(description='Find thresholds for cue detection')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to the VISO/mot dataset')
parser.add_argument('--change_thresholds', action='store_true', default=False, help='Change thresholds stored in' + 
                    'results/cue_thresholds.txt based on the statistics in results/cue_results.txt')
parser.add_argument('--iou_threshold', type=float, default=0.7, help='IoU threshold for cue detection')
parser.add_argument('--prob_range', type=float, default=0.9, help='Probability range to define the thresholds for cue filtering')
parser.add_argument('--step', type=int, default=10, help='Interframe difference to use for cue detection')
parser.add_argument('--num_folders', type=int, default=-1, help='Number of folders to process from dataset.' + \
                    'Type -1 to process all folders')


def get_morph_cues(binary, gt_boxes, **kwargs):
  '''
  Returns morphological feature averages and stds for true positives in one image
  '''

  iou_threshold = kwargs.get('iou_threshold', 0.7)
  
  labeled_image = measure.label(binary, background=0, connectivity=1)
  blobs = measure.regionprops(labeled_image)
  npos = len(gt_boxes)
  seen_gts = np.zeros(npos)
  tp_areas, tp_extents, tp_a_lengths, tp_eccentricities = [], [], [], []

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
        gt_idx = i # index of the ground truth box with the highest IoU

    if max_iou >= iou_threshold and seen_gts[gt_idx] == 0:
      seen_gts[gt_idx] = 1 # mark as detected
      tp_areas.append(min(blob.area_filled, gt_box.area)) # if blob is larger than gt box, take gt box area
      tp_extents.append(blob.area_filled / blob.area_bbox)
      tp_a_lengths.append(blob.axis_major_length)
      tp_eccentricities.append(blob.eccentricity)

  area_avg, area_std = np.mean(tp_areas) if tp_areas else -1, np.std(tp_areas) if tp_areas else -1    # what does the -1 do?
  ext_avg, ext_std = np.mean(tp_extents) if tp_extents else -1, np.std(tp_extents) if tp_extents else -1
  alen_avg, alen_std = np.mean(tp_a_lengths) if tp_a_lengths else -1, np.std(tp_a_lengths) if tp_a_lengths else -1
  ecc_avg, ecc_std = np.mean(tp_eccentricities) if tp_eccentricities else -1, np.std(tp_eccentricities) if tp_eccentricities else -1

  return len(tp_areas), area_avg, area_std, ext_avg, ext_std, alen_avg, alen_std, ecc_avg, ecc_std


def find_detection_stats(dataset_path, **kwargs):
  '''
  Loops through folders to find global averages of morph cues
  '''

  iou_threshold = kwargs.get('iou_threshold', 0.7)
  step = kwargs.get('step', 10)
  num_folders = kwargs.get('num_folders', -1) # -1 means all folders

  area_avg, ext_avg, alen_avg, ecc_avg = 0, 0, 0, 0
  area_std, ext_std, alen_std, ecc_std = 0, 0, 0, 0
  total_cands = 0

  for folder in os.listdir(f'{dataset_path}/car')[:num_folders]:
    print(f'---------- Folder {folder} ------------')
    dataloader = Dataloader(f'{dataset_path}/car/{folder}', img_file_pattern='*.jpg')
    frame_nums = dataloader.frames

    for i in range(step, len(frame_nums) - step, step):
      print(f'Progress: {(i - step) / (len(frame_nums) - step) * 100:.2f} %', end='\r')
      f0, f1, f2 = [dataloader(frame_nums[i+j*step]) for j in (-1, 0, 1)]
      grays = [color.rgb2gray(im) for _, im, _ in (f0, f1, f2)]

      binary = objects(grays)
      binary = grow(grays[1], binary)
      ncands, ar_avg, ar_std, ex_avg, ex_std, al_avg, al_std, ec_avg, ec_std = \
        get_morph_cues(binary, f1[2], iou_threshold=iou_threshold)
        
      if ncands > 0:
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

  with open('results/cue_results.txt', 'w') as f:
    f.write(f'{area_avg}, {area_std}, {ext_avg}, {ext_std}, {alen_avg}, {alen_std}, {ecc_avg}, {ecc_std}')
    

def find_thresholds(prob_range=0.9):
  '''
  Use cue results stores in results/cue_results.txt to write thresholds to 
  results/cue_thresholds.txt.
  '''
  
  with open('results/cue_results.txt', 'r') as f:
    area_avg, area_std, ext_avg, ext_std, alen_avg, alen_std, ecc_avg, ecc_std = \
      [float(n) for n in f.read().split(',')[2:]]

  t1_area = norm.ppf((1-prob_range)/2, loc=area_avg, scale=area_std)
  t2_area = 2 * area_avg - t1_area
  t1_ext = norm.ppf((1-prob_range)/2, loc=ext_avg, scale=ext_std)
  t2_ext = 2 * ext_avg - t1_ext
  t1_alen = norm.ppf((1-prob_range)/2, loc=alen_avg, scale=alen_std)
  t2_alen = 2 * alen_avg - t1_alen
  t1_ecc = norm.ppf((1-prob_range)/2, loc=ecc_avg, scale=ecc_std)
  t2_ecc = 2 * ecc_avg - t1_ecc

  with open('results/cue_thresholds.txt', 'w') as f:
    f.write(f'{t1_area}, {t2_area}, {t1_ext}, {t2_ext}, {t1_alen}, {t2_alen}, {t1_ecc}, {t2_ecc}')


if __name__ == '__main__':
  args = parser.parse_args()
  if not args.change_thresholds:
    find_detection_stats(args.dataset_path, iou_threshold=args.iou_threshold, 
                         step=args.step, 
                         num_folders=args.num_folders)
  find_thresholds(args.prob_range)


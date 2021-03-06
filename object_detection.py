import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from copy import deepcopy
from scipy.stats import norm
from dataparser import Dataloader
from skimage import measure, color, filters

from evaluation import *
  

def objects(grays, thresh):
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
      th1, th2 = -math.log(thresh) * m1, -math.log(thresh) * m2
      # mask
      b1, b2 = dif1 > th1, dif2 > th2
      # intersection
      binary[i:i+30, j:j+30] = np.logical_and(b1, b2)
  
  return binary


def grow(gray, binary, **hyperparams):
  '''
  Implement a region growing function that is applied at the centroids of candidate clusters.
  '''
  thresh = hyperparams.get('thresh', 5e-3)
  diff_thresh = hyperparams.get('diff_thresh', 0.5)
  if hyperparams.get('copy', False): binary = deepcopy(binary)
  
  gray = filters.unsharp_mask(gray, radius=2.0, amount=3.0)
  height, width = gray.shape
  blobs = measure.regionprops(
          measure.label(binary, background=0, connectivity=1))

  for blob in blobs:
    blob_rows, blob_cols = blob.coords.T
    ctr_row, ctr_col = int(blob.centroid[0]), int(blob.centroid[1])
    blob_grays = gray[blob_rows, blob_cols]
    
    if len(blob.coords) < 3:
      binary[blob_rows, blob_cols] = 0
      continue
    
    mean = np.mean(blob_grays)
    sd = np.std(blob_grays)
    if not sd: continue
      
    t1 = norm.ppf(thresh, loc=mean, scale=sd)
    t2 = 2 * mean - t1  
    if t2 - t1 > diff_thresh: continue
    
    # 11 x 11 box bounds
    l = ctr_col - 5 if ctr_col - 5 >= 0 else 0
    r = ctr_col + 6 if ctr_col + 5 < width else width
    b = ctr_row - 5 if ctr_row - 5 >= 0 else 0
    t = ctr_row + 6 if ctr_row + 5 < height else height
    
    gray_region = gray[b:t, l:r]
    candidates = np.logical_and(gray_region > t1, gray_region < t2)
    new_labels = measure.label(candidates, background=0, connectivity=1)
    candidates = np.logical_and(candidates, new_labels == new_labels[4, 4])
    binary[b:t, l:r] = np.logical_or(candidates, binary[b:t, l:r])

  return binary


def filter(binary, thresholds, **kwargs):
  '''
  Filter based on pre-defined morphological cue thresholds.
  '''
  if kwargs.get('copy', False): binary = deepcopy(binary)
  labeled_image = measure.label(binary)
  blobs = measure.regionprops(labeled_image)

  for blob in blobs:
    blob_rows, blob_cols = zip(*blob.coords)

    t1_area, t2_area = thresholds['area']
    t1_ext, t2_ext = thresholds['ext']
    t1_alen, t2_alen = thresholds['alen']
    t1_ecc, t2_ecc = thresholds['ecc']

    if blob.area_filled < t1_area or blob.area_filled > t2_area: 
      binary[blob_rows, blob_cols] = 0
    if blob.extent < t1_ext or blob.extent > t2_ext: 
      binary[blob_rows, blob_cols] = 0
    if blob.axis_major_length < t1_alen or blob.axis_major_length > t2_alen: 
      binary[blob_rows, blob_cols] = 0
    if blob.eccentricity < t1_ecc or blob.eccentricity > t2_ecc: 
      binary[blob_rows, blob_cols] = 0
 
  return binary


def get_thresholds():
  '''
  Load morphological cue thresholds from file. Should be called only once.
  Currently uses backup data because region growing is untrustworthy.
  '''

  thresholds = { 'area': (0,0), 'ext': (0,0), 'alen': (0,0), 'ecc': (0,0), }
  with open('thresholds/cue_thresholds.txt', 'r') as f:
    data = [float(n) for n in f.read().split(',')]
    thresholds['area'] = (data[0], data[1])
    thresholds['ext'] = (data[2], data[3])
    thresholds['alen'] = (data[4], data[5])
    thresholds['ecc'] = (data[6], data[7])
  return thresholds


if __name__ == '__main__':

  index = 1 if len(sys.argv) == 2 else 2
  dataset_path = sys.argv[index].rstrip('/')

  # TESTING
  loader = Dataloader(f'{dataset_path}/car/001', img_file_pattern='*.jpg', frame_range=(1, 50))
  preloaded_frames = list(loader.preloaded_frames.values())
  i0 = 10

  frames = [preloaded_frames[10+j*i0] for j in (-1,0,1)]
  ctr_frame = frames[1]
  grays = [color.rgb2gray(im) for im, _ in frames]
  ctr_gray = grays[1]

  plt.figure("Gray Image")
  plt.imshow(ctr_gray, cmap='gray')

  binary = objects(grays, 0.05)
  _, ax1 = plt.subplots()
  ax1.set_title("Candidate Objects Detection")
  ax1.imshow(binary, cmap='gray')

  grown_binary = grow(ctr_gray, binary)
  _, ax2 = plt.subplots()
  ax2.set_title("Candidate Object Growing")
  ax2.imshow(grown_binary, cmap='gray')
  
  filtered_binary = filter(grown_binary, get_thresholds())
  _, ax3 = plt.subplots()
  ax3.set_title("Candidate Match Discrimation")
  ax3.imshow(filtered_binary, cmap='gray')

  if sys.argv[1] == '--boxes':
    for gt_box in ctr_frame[1]:
      ax2.add_patch(patches.Rectangle((gt_box[0] - 0.5, gt_box[1] - 0.5), 
                                      gt_box[2], gt_box[3], 
                                      linewidth=1, edgecolor='r', facecolor='none'))
      ax3.add_patch(patches.Rectangle((gt_box[0] - 0.5, gt_box[1] - 0.5),
                                      gt_box[2], gt_box[3], 
                                      linewidth=1, edgecolor='r', facecolor='none'))
  
  candidate_clusters = measure.label(filtered_binary, background=0, connectivity=1)
  pred_bboxes = [Box(xtl=blob.bbox[1], ytl=blob.bbox[0],
                    w=blob.bbox[3] - blob.bbox[1],
                    h=blob.bbox[2] - blob.bbox[0]) for blob in measure.regionprops(candidate_clusters)]
  gt_bboxes = [Box(*gt_box) for gt_box in ctr_frame[1]]
  
  print(f'==============================')
  print(f'EVALUATION RESULTS {evaluation_metrics(pred_bboxes, gt_bboxes, iou_threshold=0.7)}')
  print(f'==============================')
  
  plt.show(block=True)
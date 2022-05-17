from scipy.optimize import linear_sum_assignment
import numpy as np
import math
from kalman_filter import KF
import random
import sys
from morph_thresholds import cue_filtering
from object_detection import objects, region_growing
from skimage import color
from dataparser import Dataloader
from skimage.measure import label, regionprops


def association(region, tracks):
    psuedo_col = []
    psuedo_row = []

    if len(region) > len(tracks):
        for i in range(len(region), len(region) + (len(region) - len(tracks))):
            psuedo_col.append(i)
    if len(tracks) > len(region):
        for i in range(len(tracks), len(tracks) + (len(tracks) - len(region))):
            psuedo_row.append(i)

    cost = np.full((len(tracks) + len(psuedo_row), len(region) + len(psuedo_col)), 1000)
    # print("done")

    # Create cost matrix
    for i in range(len(tracks)):
        tracks[i].predict()
        # print("done")
        for j in range(len(region)):
            point1 = np.array((tracks[i].x[0], tracks[i].x[1]))
            point2 = np.array((region[j].centroid[0],region[j].centroid[1]))
            cost[i, j] = np.linalg.norm(point1 - point2)

    row , col = linear_sum_assignment(cost)
    count = 0
    if len(region) > len(tracks):
        for c in col:
            if c in psuedo_col:
                tracks.append(KF(region[c].centroid[0], region[c].centroid[1], 0.1))
            else:
                tracks[row[c]].update(region[c].centroid[0], region[c].centroid[1])
            count += 1

    for i in range(len(row)):
        tracks[row[i]].update(region[col[i]].centroid[0], region[col[i]].centroid[1])

    return

if __name__ == "__main__":
    index = 1 if len(sys.argv) == 2 else 2
    dataset_path = sys.argv[index].rstrip('/')

    # TESTING
    loader = Dataloader(f'{dataset_path}/car/001', img_file_pattern='*.jpg', frame_range=(1, 50))
    preloaded_frames = list(loader.preloaded_frames.values())
    i0 = 10

    tracks = []
    for i in range(i0, len(preloaded_frames) - i0):
        frames = [preloaded_frames[i+j*i0] for j in (-1,0,1)]
        grays = [color.rgb2gray(f[1]) for f in frames]

        b = objects(grays)
        g = region_growing(grays[1], b)
        m = cue_filtering(g)

        label_b = label(m)
        blobs = regionprops(label_b)
        print(len(blobs))
        print(len(tracks))
        if i == i0:
            for b in blobs:
                tracks.append(KF(b.centroid[0], b.centroid[1], 0.1))
        else:
            association(blobs, tracks)
    
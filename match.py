from scipy.optimize import linear_sum_assignment
import numpy as np
import math
from kalman_filter import KalmanFilter
import random
import sys
from object_detection import objects, grow, filter, get_thresholds
from skimage import color
from dataparser import Dataloader
from skimage.measure import label, regionprops


def association(region, tracks, previous_frame, current_frame):
    psuedo_col = []
    psuedo_row = []
    previous_KF = tracks 
    changed_track = 0 # For GUI statistic
    
    if len(region) > len(tracks):
        for i in range(len(region), len(region) + len(region) - len(tracks)):
            psuedo_row.append(i-(len(region) - len(tracks)))

    if len(tracks) > len(region):
        for i in range(len(tracks), len(tracks) + len(tracks) - len(region)):
            psuedo_col.append(i-(len(tracks) - len(region)))

    cost = np.full((len(tracks) + len(psuedo_row), len(region) + len(psuedo_col)), 1000000)
    
    # Create cost matrix
    for i in range(len(tracks)):
        tracks[i].predict()

        for j in range(len(region)):
            point1 = np.array((tracks[i].x[0], tracks[i].x[1]))
            point2 = np.array((region[j].centroid[0],region[j].centroid[1]))
            cost[i, j] = np.linalg.norm(point1 - point2)

    print(cost)
    row , col = linear_sum_assignment(cost)
    count = 0

    # Assign new tracks to unassigned hypothesis
    if len(region) > len(tracks):
        for r in row:
            if r in psuedo_row:
                tracks.append(KalmanFilter(region[count].centroid[0], region[count].centroid[1], 0.1))
            else:
                tracks[r].update(region[count].centroid[0], region[count].centroid[1])
            count += 1
    
    # Send additional unassigned tracks to search engine
    delete_track = []
    count = 0
    if len(tracks) > len(region):
        for c in col:
            if c in psuedo_col:
                tracks[count].predict()
                bpos = search_previous(previous_frame, tracks[count].x[:2])
                if bpos is not None:
                    tracks[count].update(*bpos)
                else:
                    delete_track.append(count)
            count += 1

    return delete_track

def search_previous(previous_props, pos):
    """
    Search local area (Havent found algorithm to do it)
    return True or False
    """
    min_dist = 999999
    min_cent = None
    for blob in previous_props:
        bpos = np.array([blob.centroid[1], blob.centroid[0]])
        d = np.linalg.norm(bpos - np.array(pos))
        if d < min_dist and d < 100:
            min_dist = d
            min_cent = bpos
    return min_cent
    

if __name__ == "__main__":
    index = 1 if len(sys.argv) == 2 else 2
    dataset_path = sys.argv[index].rstrip('/')

    # TESTING
    loader = Dataloader(f'{dataset_path}/car/001', img_file_pattern='*.jpg', frame_range=(1, 10))
    preloaded_frames = list(loader.preloaded_frames.values())
    step = 1
    thresholds = get_thresholds()

    tracks = []
    previous_props = None

    for i in range(step, len(preloaded_frames)-step, step):
        frames = (preloaded_frames[i-step], preloaded_frames[i], preloaded_frames[i+step])
        grays = [color.rgb2gray(f[1]) for f in frames]

        current = grays

        b = objects(grays)
        g = grow(grays[1], b)
        f = filter(b, thresholds)

        label_f = label(f)
        blobs = regionprops(label_f)
        print('Number of hypothesis : {}'.format(len(blobs)))
        
        if i == step:
            for b in blobs:
                tracks.append(KalmanFilter(b.centroid[0], b.centroid[1], 0.1))
        else:
            delete_track = association(blobs, tracks, previous_props, current)
            new_tracks = []
            for i in range(len(tracks)):
                if i not in delete_track: new_tracks.append(tracks[i])
            tracks = new_tracks
        
        previous_props = blobs
        
        print('Number of tracks : {}'.format(len(tracks)))
    
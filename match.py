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
        for i in range(len(region), len(region) + (len(region) - len(tracks))):
            psuedo_row.append(i-(len(region) - len(tracks)))

    if len(tracks) > len(region):
        for i in range(len(tracks), len(tracks) + (len(tracks) - len(region))):
            psuedo_col.append(i-(len(tracks) - len(region)))

    cost = np.full((len(tracks) + len(psuedo_row), len(region) + len(psuedo_col)), 1000)
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
                tracks.append(KalmanFilter(region[count].centroid[0], region[count].centroid[1],0.1))
            else:
                tracks[r].update(region[count].centroid[0], region[count].centroid[1])
            count += 1
    
    # Send additional unassigned tracks to search engine
    delete_track = []
    if len(tracks) > len(region):
        for c in col:
            if c in psuedo_col:
                search_nearest(previous_frame, current_frame, previous_KF, count)
                """
                If true:
                    tracks[count] = KF(tracks[count].x[0], tracks[count].x[1], 0.1) # assuming object stopped
                else:
                    delete_track.append(count) # Delete this row later so that it does not mess up the other association
                
                """
            else:
                tracks[count].update(region[c].centroid[0], region[c].centroid[1])
            count += 1

        if len(delete_track) != 0:
            temp = []
            for i in len(tracks):
                if i not in delete_track:
                    temp.append(tracks[i])
            tracks = temp

    return changed_track

def search_nearest(previous_frame, current_frame, previous_KF, row):
    current_KF = previous_KF[row].predict()
    """
    Search local area (Havent found algorithm to do it)
    return True or False
    """
    
    pass

if __name__ == "__main__":
    index = 1 if len(sys.argv) == 2 else 2
    dataset_path = sys.argv[index].rstrip('/')

    # TESTING
    loader = Dataloader(f'{dataset_path}/car/001', img_file_pattern='*.jpg', frame_range=(1, 10))
    preloaded_frames = list(loader.preloaded_frames.values())
    step = 1
    thresholds = get_thresholds()

    tracks = []
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
            association(blobs, tracks, previous, current)
        
        previous = current
        
        print('Number of tracks : {}'.format(len(tracks)))
    
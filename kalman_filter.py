import numpy as np
import sys
from dataparser import Dataloader
from evaluation import Box
from skimage import color
from object_detection import objects
from skimage.measure import label, regionprops
from skimage.measure._regionprops import RegionProperties
from skimage.metrics import structural_similarity
from scipy.optimize import linear_sum_assignment

"""
Class to initialise a Kalman Filter
"""

PROPORTION_OF_PSEUDOS = 0.1

class KalmanFilter:
    def __init__(self, blob: RegionProperties, covar: float = 0.001):
        '''
        A new track is initialized at the location of an object 
        with a speed and acceleration of 0
        '''
        init_x, init_y = int(blob.centroid[1]), int(blob.centroid[0])
        
        # Initial mean vector of the state vector. Format : [x, y, v of x, v of y, a of x, a of y]
        self.x = np.array([init_x, init_y, 0, 0, 0, 0])
        
        # Motion correspondence
        self.F = np.array ([[1.,0.,1.,0.,0.5,0.],
                            [0.,1.,0.,1.,0.,0.5],
                            [0.,0.,1.,0.,1.,0.],
                            [0.,0.,0.,1.,0.,1.],
                            [0.,0.,0.,0.,1.,0.],
                            [0.,0.,0.,0.,0.,1.]])
        # Observation correspondence
        self.H = np.array ([[1.,0.,0.,0.,0.,0.],
                            [0.,1.,0.,0.,0.,0.]])
        
        # Covariance of motion model
        self.Q = covar * np.eye(6, dtype=float)
        # Covariance of observation model
        self.R = covar * np.eye(2,dtype=float)
        # Initial state of P is Q
        self.P = self.Q

        self.update_regprops(blob)
        
        
    @classmethod
    def assign_detections_to_tracks(cls, hypotheses: np.ndarray, tracks: np.ndarray, 
                                    previous_hypotheses: np.ndarray) -> list:
        '''
        Uses the hungarian algorithm to assign hypotheses to tracks and update each trach according '
        to the assigned hypothesis. Tracks are initialised for unassigned hypotheses and unassigned 
        tracks are matched to the closest previous hypothesis.
        '''
        new_tracks = []
        pseudo_track_ind = len(tracks)
        pseudo_hyp_ind = len(hypotheses)
        pseudo_cost = 10
        
        max_dim = max(len(tracks), len(hypotheses))
        cost_matrix = np.zeros((max_dim + int(PROPORTION_OF_PSEUDOS * max_dim), 
                                max_dim + int(PROPORTION_OF_PSEUDOS * max_dim)))
        cost_matrix[pseudo_track_ind:, :] = pseudo_cost         # cost of assigning pseudo-track to hypothesis
        cost_matrix[:, pseudo_hyp_ind:] = pseudo_cost           # cost of assigning pseudo-hypothesis to track
        cost_matrix[pseudo_track_ind:, pseudo_hyp_ind:] = 0     # theoretical cost of assigning pseudo-track to pseudo-hypothesis
        
        hyp_x = np.array([hyp.centroid[1] for hyp in hypotheses]).reshape(1, -1)
        hyp_y = np.array([hyp.centroid[0] for hyp in hypotheses]).reshape(1, -1)
        track_x = np.array([track.x[0] for track in tracks]).reshape(-1, 1)
        track_y = np.array([track.x[1] for track in tracks]).reshape(-1, 1)
        
        # use array broadcasting to compute the cost matrix
        euclidian_distances = np.sqrt((hyp_x - track_x) ** 2 + (hyp_y - track_y) ** 2)
        cost_matrix[:pseudo_track_ind, :pseudo_hyp_ind] = euclidian_distances

        rows, cols = linear_sum_assignment(cost_matrix)
        
        # assign hypotheses according to linear_sum_assignment
        assigned_inds = np.logical_and(rows < pseudo_track_ind, cols < pseudo_hyp_ind)
        assigned_pairs = zip(tracks[rows[assigned_inds]], hypotheses[cols[assigned_inds]])
        for track, hyp in assigned_pairs:
            track.update(hyp)
            new_tracks.append(track)
            
        unassigned_track_inds = rows[np.logical_and(cols >= pseudo_hyp_ind, rows < pseudo_track_ind)]
        unassigned_hypothesis_inds = cols[np.logical_and(rows >= pseudo_track_ind, cols < pseudo_hyp_ind)]
        
        # find nearest hypothesis for unassigned tracks (if it exists)
        for row in unassigned_track_inds:
            track = tracks[row]
            nearest_hyp = track.nearest_search(previous_hypotheses)
            if nearest_hyp is not None:
                track.update(nearest_hyp)
                new_tracks.append(track)

        # initialise tracks for unassigned hypotheses
        for col in unassigned_hypothesis_inds:
            hyp = hypotheses[col]
            new_tracks.append(cls(hyp))
            
        return new_tracks
    
    
    def nearest_search(self, previous_hypotheses: np.ndarray, search_radius: int = 5, 
                       ssim_thresh: float = 0.8) -> RegionProperties:
        '''
        Finds the previous hypothesis with the lowest ssim to the currently tracked vehicle
        where the hypothesis has an L2 distance from the track less than search_radius
        '''
        hyp_ctrs = np.array([hyp.centroid[::-1] for hyp in previous_hypotheses])
        euclidian_distances = np.sqrt((hyp_ctrs[:, 0] - self.x[0]) ** 2 + (hyp_ctrs[:, 1] - self.x[1]) ** 2)
        filtered_hyps = previous_hypotheses[euclidian_distances < search_radius]
        max_ssim = 0.0
        best_hyp = None
        
        for hyp in filtered_hyps:
            hyp_box = Box(xtl=hyp.bbox[1], ytl=hyp.bbox[0],
                            w=hyp.bbox[3] - hyp.bbox[1],
                            h=hyp.bbox[2] - hyp.bbox[0])
            bound_h, bound_w = [max(self.bbox.h, hyp_box.h), max(self.bbox.w, hyp_box.w)]
            
            # find images of the hypothesis and tracked object
            im_hyp = np.zeros((bound_h, bound_w))
            im_hyp[get_relative_coords(hyp.coords)] = 1
            
            # center pad hypothesis image
            y_pad = (bound_h - hyp_box.h) // 2
            x_pad = (bound_w - hyp_box.w) // 2
            im_hyp = np.pad(im_hyp, ((y_pad,), (x_pad,)), constant_values=0)
            
            # center pad track image
            y_pad = (bound_h - self.bbox.h) // 2
            x_pad = (bound_w - self.bbox.w) // 2
            im_track = np.pad(self.im_cue, ((y_pad,), (x_pad,)), constant_values=0)

            # Need to build a 7x7 or bigger matrix to pass to ssim (Need to review)
            im_track = np.pad(im_track, pad_width=3)
            im_hyp = np.pad(im_hyp, pad_width=3)
            
            # compute ssim
            ssim = structural_similarity(im_track, im_hyp, data_range=1.0)
            if ssim > max_ssim and ssim > ssim_thresh:
                max_ssim = ssim
                best_hyp = hyp
                
        return best_hyp
            
            
    def predict(self):
        '''
        Between frames, the predicted (a priori) state vector and the predicted (a priori) 
        estimate covariance is updated using the motion model.
        '''
        self.x = self.F.dot(self.x)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q


    def update(self, blob: RegionProperties):
        '''
        After the tracks and hypotheses have been associated with one another, the final 
        step before moving to the next frame is to update the state estimate of the Kalman filter
        '''
        new_x, new_y = int(blob.centroid[1]), int(blob.centroid[0])
        z = np.array([new_x, new_y])
        # Innovation calculation
        y = z - self.H.dot(self.x)
        # Innovation covariance calculation
        S = self.H.dot(self.P).dot(self.H.T) + self.R 
        # Kalman gain calculation
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        # Posteriori state estimate calculation
        self.x = self.x + K.dot(y)
        # Posteriori state covariance calculation
        self.P = (np.eye(6,dtype=float) - K.dot(self.H)).dot(self.P)

        self.update_regprops(blob)
        return self
        
        
    def update_regprops(self, blob: RegionProperties):
        '''
        Updates the state estimate of the Kalman filter with a blob
        '''
        self.coords = get_relative_coords(blob.coords)
        self.bbox = Box(xtl=blob.bbox[1], ytl=blob.bbox[0],
                        w=blob.bbox[3] - blob.bbox[1],
                        h=blob.bbox[2] - blob.bbox[0])
        self.im_cue = np.zeros((self.bbox.h, self.bbox.w))
        self.im_cue[self.coords] = 1
        

def get_relative_coords(blob_coords: np.ndarray):
    rows, cols = zip(*blob_coords)
    min_row, min_col = min(rows), min(cols)
    rows = np.array(rows) - min_row
    cols = np.array(cols) - min_col
    return rows, cols
        

# Test to see if state vector and P are updated correctly
if __name__ == '__main__':
    index = 1 if len(sys.argv) == 2 else 2
    dataset_path = sys.argv[index].rstrip('/')

    # TESTING
    # Assumes that the first centroid is related to the same track
    loader = Dataloader(f'{dataset_path}/car/001', img_file_pattern='*.jpg', frame_range=(1, 100))
    preloaded_frames = list(loader.preloaded_frames.values())
    i0 = 10
    count = 0
    for i in range(i0, len(preloaded_frames) - i0):
        frames = [preloaded_frames[i+j*i0] for j in (-1,0,1)]
        grays = [color.rgb2gray(im) for im, _ in frames]

        b = objects(grays)

        label_b = label(b)
        blobs = regionprops(label_b)

        if count == 0:
            f = KalmanFilter(blobs[0].centroid[0], blobs[0].centroid[1], 0.001)
            print("Initial x : {:f}, Initial y : {:f}, Inital velocity : {:f}".format(f.x[0],f.x[1], (f.x[2]**2+f.x[3]**2)**0.5))
            print("The P matrix : \n")
            print(f.P)
        else:
            f.predict()
            f.update(blobs[0].centroid[0], blobs[0].centroid[1])
        
        count +=1
    
    print("Final x : {:f}, Final y : {:f}, Final velocity : {:f}".format(f.x[0],f.x[1], (f.x[2]**2+f.x[3]**2)**0.5))
    print("The P matrix : \n")
    print(f.P)
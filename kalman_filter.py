import numpy as np
from skimage.measure import label, regionprops
from dataparser import Dataloader
import sys
from skimage import measure, color
from object_detection import objects
"""
Class to initialise a Kalman Filter class with the necessary vectors
"""
class KF:
    def __init__(self, init_x, init_y, covar) -> None:
        # Initial mean vector of the state vector. Format : [x, y, v of x, v of y, a of x, a of y]
        self.x = np.array([init_x,init_y,0,0,0,0])

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

    """
    Predict assumes the change in time is 1 second
    """
    def predict(self):
        self.x = self.F.dot(self.x)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q

    def update(self, meas_x, meas_y):
        z = np.array([meas_x, meas_y])

        # Innovation calculation
        y = z - self.H.dot(self.x)

        # Innovation covariance calculation
        S = self.H.dot(self.P).dot(self.H.T) + self.R 
        
        # Kalman Gain
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))

        self.x = self.x + K.dot(y)
        self.P = (np.eye(6,dtype=float) - K.dot(self.H)).dot(self.P)

#Test to see if state vector and P are updated correctly
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
        grays = [color.rgb2gray(f[1]) for f in frames]

        b = objects(grays)

        label_b = label(b)
        centroids = regionprops(label_b)

        if count == 0:
            f = KF(centroids[0].centroid[0], centroids[0].centroid[1], 0.001)
            print("Initial x : {:f}, Initial y : {:f}, Inital velocity : {:f}".format(f.x[0],f.x[1], (f.x[2]**2+f.x[3]**2)**0.5))
            print("The P matrix : \n")
            print(f.P)
        else:
            f.predict()
            f.update(centroids[0].centroid[0], centroids[0].centroid[1])
        
        count +=1
    
    print("Final x : {:f}, Final y : {:f}, Final velocity : {:f}".format(f.x[0],f.x[1], (f.x[2]**2+f.x[3]**2)**0.5))
    print("The P matrix : \n")
    print(f.P)
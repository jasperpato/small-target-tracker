import numpy as np
from filterpy.kalman import KalmanFilter
from object_detection import objects, region_growing
from dataparser import Dataloader
import sys
from skimage import measure, color

"""
Kalman filter taken from:
       https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html

Takes the x and y centroid coordinate of an object in a frame
Returns a kalmanfilter model that can update and predict movement
"""
def createFilter(x,y,var):
    """
    All array dimensions and values are based on project PDF
    var needs to be chosen and specified 
    """
    f = KalmanFilter(dim_x = 6, dim_z = 2)

    f.x = np.array ([[x],[y],[0],[0],[0],[0]])

    f.F = np.array ([[1,0,1,0,0.5,0],
                        [0,1,0,1,0,0.5],
                        [0,0,1,0,1,0],
                        [0,0,0,1,0,1],
                        [0,0,0,0,1,0],
                        [0,0,0,0,0,1]])

    f.H = np.array ([[1,0,0,0,0,0],
                    [0,1,0,0,0,0]])

    f.Q = np.array ([[var,0,0,0,0,0],
                    [0,var,0,0,0,0],
                    [0,0,var,0,0,0],
                    [0,0,0,var,0,0],
                    [0,0,0,0,var,0],
                    [0,0,0,0,0,var]])

    f.R = np.array ([[var,0],[0,var]])
    f.P = f.Q

    return f


if __name__ == '__main__':
    dataset_path = sys.argv[1].rstrip('/')

    # TESTING by using gtdata of an assumed first object
    loader = Dataloader(f'{dataset_path}/car/001', img_file_pattern='*.jpg', frame_range=(1, 100))

    first = True
    for frame, img, gtdata in loader:
        if first:
            first = False
            for e in gtdata:
                f = createFilter(e[0],e[1],1000)
                break
            print(f.x) # Print initial object state
        else:
            for e in gtdata:
                f.predict()
                f.update(np.array([e[0],e[1]]))
                break
    
    print(f.x) # Print final object state
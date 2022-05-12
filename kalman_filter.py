import numpy as np

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
    f = KF(0.0, 0.0,0.001)
    print(f.x)
    print(f.P)
    f.predict()
    print(f.x)
    print(f.P)
    f.update(1.0, 0.0)
    print(f.x)
    print(f.P)
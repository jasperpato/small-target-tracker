import math
import numpy as np
import matplotlib.pyplot as plt


def frame2gray(frame):
  '''
  Input: list of frame tuples with numpy img array as second element
  Output: list of grayscale numpy img arrays
  '''
  return np.round(np.dot(frame[1][...,:3], [0.2989, 0.5870, 0.1140]))


def objects(frames):
  '''
  Takes a list of three frames.
  Outputs a single numpy array representing a binary image, in which clusters of
  1s are candidate moving objects.
  '''
  # grayscale
  g1, g2, g3 = [frame2gray(f) for f in frames]

  # difference
  dif1, dif2 = np.abs(g1-g2), np.abs(g2-g3)

  # mean
  m1, m2 = np.mean(dif1), np.mean(dif2)

  # threshold
  th1, th2 = -math.log(0.05) * m1, -math.log(0.05) * m2

  # mask
  b1, b2 = dif1 > th1, dif2 > th2

  # intersection
  return np.logical_and(b1, b2)
import cv2
import numpy
import matplotlib.pyplot as plt


def frames2grays(frames):
  '''
  Input: list of frame tuples with numpy img array as second element
  Output: list of grayscale numpy img arrays
  '''
  g = []
  for f in frames:
    g.append(cv2.cvtColor(f[1], cv2.COLOR_RGB2GRAY))
  return g


def object_detection(frames):
  '''
  Takes a list of three frames.
  Outputs a single numpy array representing a binary image, in which clusters of
  1s are candidate moving objects.
  '''
  g1, g2, g3 = frames2grays(frames)
  dif1, dif2 = numpy.absolute(g1-g2), numpy.absolute(g2-g3)
  m1, m2 = numpy.mean(dif1), numpy.mean(dif2)

  print(m1, m2)
  plt.show(block=True)
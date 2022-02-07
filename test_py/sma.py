"""
  Simple Moving Average using numpy
"""

import numpy as np

dataset = [1,5,2,9,3,7,10,20,6,9,3,4,8,14,15]
dataset2 = [1,5,7,2,6,7,8,2,5,2,6,8,2,6,13]
def movingaverge(values, window):
  weights = np.repeat(1.0, window) / window
  smas = np.convolve(values, weights, "valid")
  return smas

ma = movingaverge(dataset2, 3)



print(ma)
import cv2
import numpy as np
import math
from numpy.linalg import inv



d_mat = np.array([])

def dct(img):
    height, weight = img.shape[:2]

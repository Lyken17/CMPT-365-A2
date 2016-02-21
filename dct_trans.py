import cv2
import numpy as np
import math
from numpy.linalg import inv


dct_size = 0
dct_mat = np.array([])
idct_mat = np.array([])

def dct_2d(img):
    return dct(dct(img, axis=0), axis=1)

def idct_2d(img):
    return idct(idct(img, axis=0), axis=1)

def dct(img, axis = 0):
    N = img.shape[1] if axis == 0 else img.shape[0]

    global dct_mat, idct_mat, dct_size
    if N != dct_size:
        dct_mat = np.zeros((N, N))
        for x in xrange(N):
            for y in xrange(N):
                    if x == 0:
                        dct_mat[x, y] = math.sqrt(0.5)
                    else:
                        dct_mat[x, y] = math.cos(math.pi / float(N) * (y + 0.5) * x)

        dct_mat *= math.sqrt(2.0 / N)
        idct_mat = inv(dct_mat)
        dct_size = N

    return np.dot(img, dct_mat.T) if axis == 0 else np.dot(dct_mat, img)

def idct(img, axis = 0):
    N = img.shape[1] if axis == 0 else img.shape[0]

    global dct_mat, idct_mat, dct_size
    if N != dct_size:
        dct_mat = np.zeros((N, N))
        for x in xrange(N):
            for y in xrange(N):
                    if x == 0:
                        dct_mat[x, y] = math.sqrt(0.5)
                    else:
                        dct_mat[x, y] = math.cos(math.pi / float(N) * (y + 0.5) * x)

        dct_mat *= math.sqrt(2.0 / N)
        idct_mat = inv(dct_mat)
        dct_size = N

    return np.dot(img, idct_mat.T) if axis == 0 else np.dot(idct_mat, img)


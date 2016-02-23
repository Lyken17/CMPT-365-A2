import cv2
import numpy as np
import math
from numpy.linalg import inv


dct_size = 0
dct_mat = np.array([])
idct_mat = np.array([])
quantization_table = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]])

def dct_web_api(file, quality):
    pass


def dct_quantization(img, quality=100):
    scaling_factor = float(100 - quality) / 50 if quality >= 50 else float(50) / quality
    data = np.zeros_like(img, dtype=float)
    data[:] = img
    global quantization_table
    if scaling_factor != 0.0:
        q_table = np.rint(quantization_table * scaling_factor)
    else:
        q_table = quantization_table
    # from scipy.fftpack

    for c in xrange(3):
        for x in xrange(0, 512, 8):
            for y in xrange(0, 512, 8):
                data[x:x + 8, y:y + 8, c] = dct_2d(data[x:x + 8, y:y + 8, c])
                data[x:x + 8, y:y + 8, c] = np.rint(data[x:x + 8, y:y + 8, c] / q_table) * q_table
                data[x:x + 8, y:y + 8, c] = idct_2d(data[x:x + 8, y:y + 8, c])

    img[:] = data
    print
    return img

def dct_2d(img):
    return __dct__(__dct__(img, axis=0), axis=1)

def idct_2d(img):
    return __idct__(__idct__(img, axis=0), axis=1)

def __dct__(img, axis = 0):
    # N = img.shape[1] if axis == 0 else img.shape[0]
    N = 8
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

def __idct__(img, axis = 0):
    # N = img.shape[1] if axis == 0 else img.shape[0]
    N = 8
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


import cv2
import numpy as np
import math, os, sys
from numpy.linalg import inv

dct_size = 0
img_dir = "static/uploads/"
dct_mat = np.array([])
idct_mat = np.array([])

luma_table = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                       [12, 12, 14, 19, 26, 58, 60, 55],
                       [14, 13, 16, 24, 40, 57, 69, 56],
                       [14, 17, 22, 29, 51, 87, 80, 62],
                       [18, 22, 37, 56, 68, 109, 103, 77],
                       [24, 35, 55, 64, 81, 104, 113, 92],
                       [49, 64, 78, 87, 103, 121, 120, 101],
                       [72, 92, 95, 98, 112, 100, 103, 99]], dtype=float)

chroma_table = np.array([[17, 18, 24, 47, 99, 128, 192, 256],
                    [18, 21, 26, 66, 99, 128, 192, 256],
                    [24, 26, 56, 99, 128, 192, 256, 512],
                    [47, 66, 99, 128, 192, 256, 512, 1024],
                    [99, 99, 128, 192, 256, 512, 1024, 2048],
                    [128, 128, 192, 256, 512, 1024, 3072, 4096],
                    [192, 192, 256, 512, 1024, 3072, 6144, 7168],
                    [256, 256, 512, 1024, 2048, 4096, 7168, 8192]], dtype=float)

from PIL import Image



def dct_web_api(file):
    print os.path.join(img_dir, file)
    img = cv2.imread(os.path.join(img_dir, file))
    img = cv2.resize(img, (512, 512))
    original = file.split('.')[0] + '.jpg'
    cv2.imwrite(os.path.join(img_dir, original), img)

    trans = dct_YUV(img)
    for q in xrange(3):
        temp_name = file.split('.')[0] + '_dct_' + str(q+1) + '.jpg'
        cv2.imwrite(os.path.join(img_dir, temp_name), trans[:,:,q])

    for q in xrange(100, 0, -25):
        temp_name = file.split('.')[0] + '_' + str(q) + '.jpg'
        temp_img = dct_quantization(img, q)
        cv2.imwrite(os.path.join(img_dir, temp_name), temp_img)

    for q in xrange(100, 0, -25):
        temp_name = file.split('.')[0] + '_' + str(q) + '_chroma.jpg'
        temp_img = dct_quantization(img, q, choice=1)
        cv2.imwrite(os.path.join(img_dir, temp_name), temp_img)

    '''
    img = Image.open(os.path.join(img_dir, file))
    for q in xrange(100, 0, -25):
        temp_name = file.split('.')[0] + '_' + str(q) + '.jpg'
        img.save(os.path.join(img_dir, temp_name),'JPEG', quality=q)
    '''

    return "success"

def dct_YUV(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    data = np.zeros_like(img, dtype=float)
    data[:] = img

    for c in xrange(3):
        for x in xrange(0, 512, 8):
            for y in xrange(0, 512, 8):
                data[x:x + 8, y:y + 8, c] = dct_2d(data[x:x + 8, y:y + 8, c])

    img[:] = data
    img = cv2.cvtColor(img, cv2.COLOR_YCR_CB2RGB)
    return img

def dct_quantization(img, quality=100, choice=0):
    scaling_factor = float(100 - quality) / 50 if quality >= 50 else float(50) / quality

    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    data = np.zeros_like(img, dtype=float)
    data[:] = img

    global luma_table, chroma_table

    q_table = luma_table / 12 if choice == 0 else chroma_table / 16

    if quality != 100:
        if scaling_factor != 0.0:
            q_table = np.round(luma_table * scaling_factor)

    for c in xrange(3):
        for x in xrange(0, 512, 8):
            for y in xrange(0, 512, 8):
                data[x:x + 8, y:y + 8, c] = dct_2d(data[x:x + 8, y:y + 8, c])
                data[x:x + 8, y:y + 8, c] = np.around(data[x:x + 8, y:y + 8, c] / q_table) * q_table
                data[x:x + 8, y:y + 8, c] = idct_2d(data[x:x + 8, y:y + 8, c])

    img[:] = data
    img = cv2.cvtColor(img, cv2.COLOR_YCR_CB2RGB)
    return img


def dct_2d(img):
    from scipy.fftpack import dct
    # return dct(dct(img.T, norm='ortho').T, norm='ortho')
    return __dct__(__dct__(img, axis=0), axis=1)


def idct_2d(img):
    from scipy.fftpack import idct
    # return idct(idct(img.T, norm='ortho').T, norm='ortho')
    return __idct__(__idct__(img, axis=0), axis=1)


def __dct__(img, axis=0):
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


def __idct__(img, axis=0):
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

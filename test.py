import cv2
import numpy as np
import math
from numpy.linalg import inv

from image_process import *
from dct_trans import *

img_dir = "static/img.jpg"


def main():
    data = cv2.imread(img_dir)
    data = cv2.resize(data, (512, 512))

    img = np.zeros_like(data, dtype=float)
    # print(img.dtype)

    img[:] = data
    img = subsample(img)
    img = dct_quantization(img)
    data[:] = img
    cv2.imshow("image", data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def dct_quantization(img):
    quantization_table = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                [12, 12, 14, 19, 26, 58, 60, 55],
                                [14, 13, 16, 24, 40, 57, 69, 56],
                                [14, 17, 22, 29, 51, 87, 80, 62],
                                [18, 22, 37, 56, 68, 109, 103, 77],
                                [24, 35, 55, 64, 81, 104, 113, 92],
                                [49, 64, 78, 87, 103, 121, 120, 101],
                                [72, 92, 95, 98, 112, 100, 103, 99]])
    # from scipy.fftpack
    '''
    for c in xrange(3):
        for x in xrange(0, 512, 8):
            for y in xrange(0, 512, 8):
                img[x:x + 8, y:y + 8, c] = dct_2d(img[x:x + 8, y:y + 8, c])
                img[x:x + 8, y:y + 8, c] = np.rint(img[x:x + 8, y:y + 8, c] / quantization_table) * quantization_table
                img[x:x + 8, y:y + 8, c] = idct_2d(img[x:x + 8, y:y + 8, c])
    '''

    print
    return img


def test():
    img = cv2.imread(img_dir)
    arr = np.random.random((4, 3))
    print(idct_2d(dct_2d(arr)) - arr)


if __name__ == "__main__":
    main()
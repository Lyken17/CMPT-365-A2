import cv2
import numpy as np
import math
from numpy.linalg import inv

from image_process import *
from dct_trans import *

img_dir = "static/saber.jpg"


def main():
    data = cv2.imread(img_dir)

    # img.astype(float)
    # print(data.shape)

    img = np.zeros_like(data, dtype=float)
    # print(img.dtype)

    img[:] = data
    img = RGB2YUV(img)
    img = YUV2RGB(img)

    data[:] = img

    print(data)
    print(img)
    cv2.imshow("image", data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test():
    img = cv2.imread(img_dir)
    arr = np.random.random((4,3))
    print(idct_2d(dct_2d(arr)) - arr)

if __name__ == "__main__":
    main()

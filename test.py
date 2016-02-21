import cv2
import numpy as np
import math
from numpy.linalg import inv

from image_process import *
from dct_trans import *

img_dir = "static/saber.jpg"


def main():
    img = cv2.imread(img_dir)

    img_array = np.asarray(img)
    img.astype(float)

    img = RGB2YUV(img)
    img = YUV2RGB(img)

    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test():
    img = cv2.imread(img_dir)
    arr = np.random.random((4,3))
    # arr = np.array([[4., 3., 5., 10., 18, 20], [4, 3., 5., 10., 18, 20]])
    print(arr)
    print(dct_2d(arr))
    print(idct_2d(dct_2d(arr)))

if __name__ == "__main__":
    main()

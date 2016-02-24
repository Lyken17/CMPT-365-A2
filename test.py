import cv2
import numpy as np
import math
from numpy.linalg import inv
from image_process import *

from image_process import *
from dct_trans import *

img_dir = "static/uploads/temp.jpg"
img_dir = "static/temp.jpg"


def main():
    data = cv2.imread(img_dir)
    data = cv2.resize(data, (512, 512))

    # img[:] = data
    data = subsample(data)
    # data = dct_quantization(data)

    # cv2.imshow("image", data)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()



def test():
    img = cv2.imread(img_dir)
    img = cv2.resize(img, (512, 512))
    img = dct_quantization(img, choice=1)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

@img_preprocess
def lalala(img):
    pass

if __name__ == "__main__":
   test()
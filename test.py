import cv2
import numpy as np
from scipy.fftpack import dct

from image_process import subsample

img_dir = "static/saber.jpg"


def main():
    img = cv2.imread(img_dir)
    img_array = np.asarray(img)
    img = subsample(img)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test():
    img = cv2.imread(img_dir)
    arr = np.array([4., 3., 5., 10., 18, 20])
    print img.dtype
    img = np.int(img)
    d1 = dct(dct(img[:,:,1], 2, axis=0),2, axis=1)
    # d2 = dct(d1, 3) / 2 / len(arr)



if __name__ == "__main__":

    test()

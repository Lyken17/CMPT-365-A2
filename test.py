import cv2
import numpy as np
import math
from numpy.linalg import inv

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
    # d1 = dct(arr, 2)
    # d2 = idct(arr)
    # print d1
    # print d2

    N = 6
    dct_mat = np.zeros((N, N))
    for x in xrange(N):
        for y in xrange(N):
                if x == 0:
                    dct_mat[x, y] = math.sqrt(0.5)
                else:
                    dct_mat[x, y] = math.cos(math.pi / float(N) * (y + 0.5) * x)

    dct_mat *= math.sqrt(2.0 / N)
    idct_mat = inv(dct_mat)
    # print dct_mat * math.sqrt(0.5)
    ans = np.dot(dct_mat, arr)
    print ans
    print np.dot(idct_mat, ans)


if __name__ == "__main__":
    test()

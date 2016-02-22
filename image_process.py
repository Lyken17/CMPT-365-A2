import numpy as np
import cv2
from numpy.linalg import inv


class YUV(object):
    def __init__(self, f):
        # print("YUV decorator has initialised")
        self.f = f

    def __call__(self, *args, **kwargs):
        print("enter func %s") % self.f.__name__
        self.f(*args, **kwargs)
        print("exit func %s") % self.f.__name__


R_Y_mat = np.array([[0.299,0.587,0.114],
                    [-0.14713, -0.28886, 0.436],
                    [0.615, -0.51499, -0.10001]])
Y_R_mat = np.array([[1, 0, 1.13983],
                    [1, -0.39465, -0.58060],
                    [1, 2.03211, 0]])

def RGB2YUV(img):
    height, weight = img.shape[:2]


    # for x in xrange(height):
    #     for y in xrange(weight):
    #       img[x,y] = np.dot(R_Y_mat, img[x,y])

    # print(img[15, 15])
    # img[15, 15] =  np.dot(R_Y_mat, img[15, 15])
    # print(img[15, 15])
    # img[15, 15] =  np.dot(Y_R_mat, img[15, 15])
    # print(img[15, 15])

    img = np.einsum('lk,ijl->ijk', R_Y_mat, img)

    return img

def YUV2RGB(img):
    height, weight = img.shape[:2]

    # for x in xrange(height):
    #     for y in xrange(weight):
    #       img[x,y] = np.dot(Y_R_mat, img[x,y])

    # img[15, 15] =  np.dot(R_Y_mat, img[15, 15])
    # print(img[15,15])
    img = np.einsum('lk,ijl->ijk', Y_R_mat, img)

    return img


def subsample(img):
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    img = RGB2YUV(img)
    img = chroma_4_2_0_subsampling(img)
    img = YUV2RGB(img)
    # img = cv2.cvtColor(img, cv2.COLOR_YCR_CB2RGB)
    return img


def chroma_4_2_0_subsampling(img):

    height, weight = img.shape[:2]

    # Subsample U by half
    # print img[:,:,1]

    for x in xrange(height):
        for y in xrange(weight):
            img[x, y, 1] = img[x - x % 2, y - y % 2 , 1]

    # Subsample V by zero
    # print img[:,:,2]

    for x in xrange(height):
        for y in xrange(weight):
            img[x, y, 2] = img[x - x % 2, y - y % 2 , 2]

    # print img[:,:,2]
    return img



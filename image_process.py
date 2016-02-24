import numpy as np
import cv2
from numpy.linalg import inv
import os, sys

WRITE2FILE = True

class img_preprocess(object):
    def __init__(self, f):
        # print("YUV decorator has initialised")
        self.f = f

    def __call__(self, *args, **kwargs):
        print("enter func %s") % self.f.__name__

        self.f(*args, **kwargs)

        print("exit func %s") % self.f.__name__

img_dir = "static/uploads/"

R_Y_mat = np.array([[0.299,0.587,0.114],
                    [-0.14713, -0.28886, 0.436],
                    [0.615, -0.51499, -0.10001]])
Y_R_mat = np.array([[1, 0, 1.13983],
                    [1, -0.39465, -0.58060],
                    [1, 2.03211, 0]])


def RGB2YUV(img):
    return np.einsum('lk,ijl->ijk', R_Y_mat, img)


def YUV2RGB(img):
    return np.einsum('lk,ijl->ijk', Y_R_mat, img)



def subsample_web_api(file):
    img = cv2.imread(os.path.join(img_dir, file))
    img = cv2.resize(img, (512, 512))
    file = file.split('.')[0] + '.jpg'
    cv2.imwrite(os.path.join(img_dir, file), img)

    subsample(img)


def subsample(img):

    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    # img = RGB2YUV(img)

    if WRITE2FILE:
        cv2.imwrite('static/uploads/temp_Y.jpg',img[:,:,0])
        cv2.imwrite('static/uploads/temp_U.jpg',img[:,:,1])
        cv2.imwrite('static/uploads/temp_V.jpg',img[:,:,2])


    data = np.zeros_like(img, dtype=float)
    data[:] = img

    d1 = chroma_4_2_0_subsampling(data)
    d2 = chroma_4_2_2_subsampling(data)

    img[:] = d1
    # img = YUV2RGB(img)
    img = cv2.cvtColor(img, cv2.COLOR_YCR_CB2RGB)

    if WRITE2FILE:
        cv2.imwrite('static/uploads/temp_4_2_0.jpg',img)

    img[:] = d2
    # img = YUV2RGB(img)
    img = cv2.cvtColor(img, cv2.COLOR_YCR_CB2RGB)

    if WRITE2FILE:
        cv2.imwrite('static/uploads/temp_4_2_2.jpg',img)

    cv2.destroyAllWindows()
    return img

def chroma_4_2_2_subsampling(img):

    height, weight = img.shape[:2]

    # Subsample U by half
    # print img[:,:,1]

    for x in xrange(height):
        for y in xrange(weight):
            img[x, y, 1] = img[x - x % 2, y , 1]

    # Subsample V by zero
    # print img[:,:,2]

    for x in xrange(height):
        for y in xrange(weight):
            img[x, y, 2] = img[x - x % 2, y, 2]

    # print img[:,:,2]
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



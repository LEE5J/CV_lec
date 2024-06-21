import cv2
import numpy as np
import math
import copy
from PIL import Image

def show_img(image, title="image"):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def func_trans(src):
    negative = copy.deepcopy(src)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            for k in range(src.shape[2]):
                negative[i,j,k] = 255-src[i,j,k]
    show_img(negative,"negative")
    log = copy.deepcopy(src)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            for k in range(src.shape[2]):
                log[i, j, k] = 255 * math.log(float(src[i, j, k] / 255)*(math.e-1)+1)
    show_img(log,"log")
    pow_small = copy.deepcopy(src)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            for k in range(src.shape[2]):
                pow_small[i,j,k] = 255*math.pow(float(src[i, j, k]/255),0.6)
    show_img(pow_small,"powsmall")
    pow_big = copy.deepcopy(src)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            for k in range(src.shape[2]):
                pow_big[i, j, k] = 255 * math.pow(float(src[i, j, k] / 255), 3)
    show_img(pow_big,"powbig")

pic = "HW1-2  Fig0525(b)(aerial_view_turb_c_0pt0025) .tif"
src = cv2.imread(pic, cv2.IMREAD_COLOR)
show_img(src)
func_trans(src)

def histo_equal(src):
    histo = copy.deepcopy(src)
    sum = np.zeros((256))
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            sum[src[i, j, 0]] += 1 #흑백
    total = 0
    lookup = np.zeros((256))
    nomalize = float(src.shape[0]*src.shape[1]/255)
    for i in range(256):
        total += sum[i]
        lookup[i] = int(0.5+total / nomalize)

    for i in range(histo.shape[0]):
        for j in range(histo.shape[1]):
            histo[i,j,0] = lookup[src[i,j,0]]
            histo[i,j,1] = lookup[src[i,j,0]]
            histo[i,j,2] = lookup[src[i,j,0]]
    show_img(histo,"histo_equal")

pic = "HW1-1 Fig0316(1)(top_left).jpg"
src = cv2.imread(pic, cv2.IMREAD_COLOR)
show_img(src)
histo_equal(src)


def show_hsv(imgae, title="HSV_image"):
    show_img(cv2.cvtColor(imgae, cv2.COLOR_HSV2BGR), title)


def func_trans_color(src):
    hsi = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    negative = copy.deepcopy(hsi)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            negative[i,j,2] = 255-hsi[i,j,2]
    show_hsv(negative,"negetive_color")
    log = copy.deepcopy(hsi)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            log[i, j, 2] = 255 * math.log(float(src[i, j, 2] / 255) * (math.e - 1) + 1)
    show_hsv(log, "log")
    pow_small = copy.deepcopy(hsi)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            pow_small[i, j, 2] = 255 * math.pow(float(src[i, j, 2] / 255), 0.7)
    show_hsv(pow_small, "powsmall")
    pow_big = copy.deepcopy(hsi)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            pow_big[i, j, 2] = 255 * math.pow(float(src[i, j, 2] / 255), 2)
    show_hsv(pow_big, "powbig")


def histo_equal_color(src):
    hsi = cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
    sum = np.zeros((256))
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            sum[hsi[i, j, 2]] += 1 #흑백
    total = 0
    lookup = np.zeros((256))
    normalize = float(src.shape[0]*src.shape[1]/255)
    for i in range(256):
        total += sum[i]
        lookup[i] = int(0.5+total / normalize)

    for i in range(hsi.shape[0]):
        for j in range(hsi.shape[1]):
            hsi[i,j,2] = lookup[hsi[i,j,2]]
    show_hsv(hsi,"histo_equal")


pic = "baboon.png"
src = cv2.imread(pic, cv2.IMREAD_COLOR)
show_img(src)
func_trans_color(src)
histo_equal_color(src)


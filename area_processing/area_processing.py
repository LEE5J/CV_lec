import cv2
import numpy as np
import math
import copy
from PIL import Image
import os
import statistics
from multiprocessing import Pool


def show_img(image, title="image"):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_hsv(imgae, title="HSV_image"):
    show_img(cv2.cvtColor(imgae, cv2.COLOR_HSV2BGR), title)


def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
        -((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2 * sigma ** 2)), (size, size))
    return kernel / np.sum(kernel)


def rgb2gray(image):
    height, width = image.shape[:2]
    gray_image = image.copy()
    for i in range(height):
        for j in range(width):
            r, g, b = image[i, j]
            gray_value = 0.299 * r + 0.587 * g + 0.114 * b
            gray_image[i, j] = (gray_value, gray_value, gray_value)
    return gray_image


def noise_filtering_median(image, size=3):
    result = image.copy()
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            targ_list = list()
            size = 7
            for dx in range(int(-size / 2), int((size + 1) / 2)):
                for dy in range(int(-size / 2), int((size + 1) / 2)):
                    try:
                        targ_list.append(image[x + dx, y + dy, 2])
                    except:
                        targ_list.append(0)
            result[x, y, 2] = statistics.median(targ_list)
    return result


def noise_filtering_gausian(image, size=7, sigma=1.5):
    result = image.copy()
    mask = gaussian_kernel(size, sigma)
    padded_image = np.pad(image, ((size // 2, size // 2), (size // 2, size // 2), (0, 0)),
                          mode='constant')
    # show_hsv(padded_image)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            patch = padded_image[x:x + size, y:y + size, 2]
            result[x, y, 2] = np.sum(patch * mask)
    return result


def HBF(image, alpha=0.7):
    LF_img = noise_filtering_gausian(image)
    # show_hsv(LF_img,"LF_img_hsv")
    high_boost_image = image.copy()
    high_boost_image[:, :, 2] = image[:, :, 2] * (1 + alpha) - LF_img[:, :, 2]
    high_boost_image = np.clip(high_boost_image, 0, 255)
    return high_boost_image.astype(np.uint8)


def edge_detection_sobel(image):
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])
    image = image.astype(np.float32)
    b, g, r = cv2.split(image)
    x_r = cv2.filter2D(r, -1, kernel_x)
    y_r = cv2.filter2D(r, -1, kernel_y)
    x_g = cv2.filter2D(g, -1, kernel_x)
    y_g = cv2.filter2D(g, -1, kernel_y)
    x_b = cv2.filter2D(b, -1, kernel_x)
    y_b = cv2.filter2D(b, -1, kernel_y)
    edges = cv2.magnitude(x_r, y_r) + cv2.magnitude(x_g, y_g) + cv2.magnitude(x_b, y_b)
    # show_img(edges)
    return edges.astype(np.uint8)


def edge_detection_prewitt(image):
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1],
                         [0, 0, 0],
                         [1, 1, 1]])
    image = image.astype(np.float32)
    b, g, r = cv2.split(image)
    x_r = cv2.filter2D(r, -1, kernel_x)
    y_r = cv2.filter2D(r, -1, kernel_y)
    x_g = cv2.filter2D(g, -1, kernel_x)
    y_g = cv2.filter2D(g, -1, kernel_y)
    x_b = cv2.filter2D(b, -1, kernel_x)
    y_b = cv2.filter2D(b, -1, kernel_y)
    edges = cv2.magnitude(x_r, y_r) + cv2.magnitude(x_g, y_g) + cv2.magnitude(x_b, y_b)
    # show_img(edges)
    return edges.astype(np.uint8)


def edge_detection_canny(image, gray=False):
    if not gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image, 100, 200)
    return edges


def edge_detection_log(image, size=5, gray=False):
    if not gray:
        image = rgb2gray(image)
    blurred = cv2.GaussianBlur(image, (size, size), 0)
    edges = cv2.Laplacian(blurred, cv2.CV_32F)
    return edges.astype(np.uint8)


def load_img(pic):
    src = cv2.imread(pic, cv2.IMREAD_COLOR)
    if src is None:
        pic = "./area_processing" + pic[1:]
        src = cv2.imread(pic, cv2.IMREAD_COLOR)
    return src


if __name__ == "__main__":
    print(os.getcwd())
    pic = "./Noise_Filtering/Fig0504(a)(gaussian-noise).jpg"
    src = load_img(pic)
    show_img(src, "noise_origin")
    gray = rgb2gray(src)
    gray_hsv = cv2.cvtColor(gray, cv2.COLOR_BGR2HSV)
    show_hsv(noise_filtering_median(gray_hsv, size=3), "noise_MD3")
    show_hsv(noise_filtering_median(gray_hsv), "noise_MD7")
    show_hsv(noise_filtering_gausian(gray_hsv, 3, 0.5), "noise_GS3,0.5")
    show_hsv(noise_filtering_gausian(gray_hsv, 3, 1.5), "noise_GS3,1.5")
    show_hsv(noise_filtering_gausian(gray_hsv, 7, 0.5), "noise_GS7,0.5")
    show_hsv(noise_filtering_gausian(gray_hsv, 7, 1.5), "noise_GS7,1.5")
    pic = "./Noise_Filtering/Lena_noise.png"
    src = load_img(pic)
    show_img(src, "noise_origin")
    src = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    #show_img(gray)
    show_img(cv2.medianBlur(src, 3),"noise_MD3RGB")
    show_img(cv2.medianBlur(src, 7), "noise_MD7RGB")
    show_hsv(noise_filtering_median(src, size=3), "noise_MD3")
    show_hsv(noise_filtering_median(src), "noise_MD7")
    show_hsv(noise_filtering_gausian(src, 3, 0.5), "noise_Gausian3,0.5")
    show_hsv(noise_filtering_gausian(src, 3, 1.5), "noise_Gausian3,1.5")
    show_hsv(noise_filtering_gausian(src, 7, 0.5), "noise_Gausian7,0.5")
    show_hsv(noise_filtering_gausian(src, 7, 1.5), "noise_Gausian7,1.5")
    pic = "./High-boost_Filtering/Fig0327(a)(tungsten_original).jpg"
    src = load_img(pic)
    show_img(src, "HBF_origin")
    gray = rgb2gray(src)
    gray_hsv = cv2.cvtColor(gray, cv2.COLOR_BGR2HSV)
    show_hsv(HBF(gray_hsv, 1), "HBF_1")  # 너무 밝아지는 경향
    show_hsv(HBF(gray_hsv, 0.2), "HBF_0.2")  # 어두워짐
    pic = "./High-boost_Filtering/Fig0525(a)(aerial_view_no_turb).jpg"
    src = load_img(pic)
    show_img(src, "HBF_origin")
    gray = rgb2gray(src)
    gray_hsv = cv2.cvtColor(gray, cv2.COLOR_BGR2HSV)
    show_hsv(HBF(gray_hsv, 1), "HBF_1")
    show_hsv(HBF(gray_hsv, 0.2), "HBF_0.2")
    pic = "./Edge_Detection/Fig0327(a)(tungsten_original).jpg"
    src = load_img(pic)
    show_img(src,"edge_origin")
    src = rgb2gray(src)
    show_img(edge_detection_sobel(src), "Edge SB")
    show_img(edge_detection_prewitt(src), "Edge PW")
    show_img(edge_detection_log(src), "Edge LoG")
    show_img(edge_detection_canny(src), "Edge CN")
    pic = "./Edge_Detection/lenna_color.bmp"
    src = load_img(pic)
    show_img(src,"edge_origin")
    src = rgb2gray(src)
    show_img(edge_detection_sobel(src), "Edge SB")
    show_img(edge_detection_prewitt(src), "Edge PW")
    show_img(edge_detection_log(src,size=9), "Edge LoG9")
    show_img(edge_detection_canny(src), "Edge CN")

import os
import random
import cv2
import numpy
import torch

import cv2
import os
import numpy as np
import random
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage import exposure
from skimage.util import img_as_uint,random_noise,img_as_ubyte
import threading
def Spnoise_func(image, prob=0.4):
    '''
    添加椒盐噪声
    prob:噪声比例
    '''
    output = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
    return output


def add_super_noise(img):
    # img = cv2.resize(img, (0, 0), fx=4, fy=4)
    # for i in range(5):
    img = random_noise(img,mean=0.2,var=25, mode='speckle')
    # img = img+noise
    # img = Spnoise_func(img)
    # img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

    return img
def BIG_pepper_and_salt_SMALL(image, percentage, size_):
    height, width = image.shape[0], image.shape[1]
    # 设置新的图片分辨率框架
    width_new = width * size_
    height_new = height * size_
    # 判断图片的长宽比率
    if width / height >= width_new / height_new:
        img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv2.resize(image, (int(width * height_new / height), height_new))

    num = int(percentage * img_new.shape[0] * img_new.shape[1])  # 椒盐噪声点数量
    random.randint(0, img_new.shape[0])
    img2 = img_new.copy()




    for i in range(num):
        X = random.randint(5, img2.shape[0] - 5)  # 从0到图像长度之间的一个随机整数,因为是闭区间所以-1
        Y = random.randint(5, img2.shape[1] - 5)
        if random.randint(0, 4) == 5:  # 黑白色概率55开
            img2[X, Y] = (int(random.randint(0, 255)), int(random.randint(0, 255)), int(random.randint(0, 255)))  # 随机色彩
            img2[X+1, Y+1] = (int(random.randint(0, 255)), int(random.randint(0, 255)), int(random.randint(0, 255)))  # 随机色彩
            img2[X, Y+1] = (int(random.randint(0, 255)), int(random.randint(0, 255)), int(random.randint(0, 255)))  # 随机色彩
            img2[X+1, Y] = (int(random.randint(0, 255)), int(random.randint(0, 255)), int(random.randint(0, 255)))  # 随机色彩


        else:
            img2[X, Y] = (0, 0, 0)  # 黑色
            img2[X+1, Y+1] = (0, 0, 0)  # 黑色
            img2[X+1, Y] = (0, 0, 0)  # 黑色
            img2[X, Y+1] = (0, 0, 0)  # 黑色

    height, width = img2.shape[0], img2.shape[1]


    import numpy as np
    gauss = np.random.randn(height_new, width_new, 3)
    # 给图片添加speckle噪声
    noisy_img = img2 + img2 * gauss
    # 归一化图像的像素值
    img2 = np.clip(noisy_img, a_min=0, a_max=255)


    # 设置新的图片分辨率框架
    width_new = width //(size_)
    height_new = height //(size_)
    # 判断图片的长宽比率
    if width / height >= width_new / height_new:
        img_new = cv2.resize(img2, (width_new, int(height * width_new / width)))
    else:
        img_new = cv2.resize(img2, (int(width * height_new / height), height_new))

    return img_new


img = cv2.imread('H:\\Desktop\\prepare\\2.png')
# cv2.imshow('2',img)
# img2 = BIG_pepper_and_salt_SMALL(img,0.2)
# cv2.imshow('1', img2)
cv2.waitKey(0)


dir_path = r"F:\\train\\data"
files = os.listdir(dir_path)
for file in files:
    for images in os.listdir(r"F:\\train\\data\\"+file):
        if 'b' in images:
            print(dir_path+"\\"+file +"\\" +images)
            img = cv2.imread(dir_path+"\\"+file +"\\" +images)
    # img2 = add_super_noise(img)
            img2 = BIG_pepper_and_salt_SMALL(img, 0.1, 4)

    # img = random_noise(img, mode='speckle')
    # img2 = Spnoise_func(img)
    # img2 = add_super_noise(img)
    # cv2.imshow(file, img2)
    #         file_new=file.replace('.','_new.')
            cv2.imwrite(dir_path+"\\"+file +"\\" +images, img2)
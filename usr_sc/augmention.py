import cv2
import random
import copy
import os
import math
import numpy as np
import time
from skimage.util import random_noise

img_path='/mnt/e/file/image/image_foc/3-2.jpg'

def show_pic(img):
    cv2.imwrite(img,'/mnt/e/file/ultralytics/usr_sc/out.jpg')

def _addNoise(img):
    return random_noise(img,mode='gaussian',seed=int(time.time()),clip=True)*255

def _changeLight(img):
    alpha=random.uniform(0.35,5)
    blank=np.zeros(img.shape,img.dtype)
    return cv2.addWeighted(img,alpha,blank,1-alpha,0)

def rotate_img_bbox(img,angle=5,scale=1.):
    w = img.shape[1]
    h = img.shape[0]
    # 角度变弧度
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    # the move only affects the translation, so update the translation
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    
    rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
    
    return rot_img
    
    # 平移
def shift_pic_bboxes( img):
    '''
    参考:https://blog.csdn.net/sty945/article/details/79387054
    平移后的图片要包含所有的框
    输入:
        img:图像array
        bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
    输出:
        shift_img:平移后的图像array
        shift_bboxes:平移后的bounding box的坐标list
    '''
    # ---------------------- 平移图像 ----------------------
    w = img.shape[1]
    h = img.shape[0]
    x_min = w  # 裁剪后的包含所有目标框的最小的框
    x_max = 0
    y_min = h
    y_max = 0
    bboxes=[[500,600,600,500]]
    for bbox in bboxes:
        x_min = min(x_min, bbox[0])
        y_min = min(y_min, bbox[1])
        x_max = max(x_max, bbox[2])
        y_max = max(y_max, bbox[3])

    d_to_left = x_min  # 包含所有目标框的最大左移动距离
    d_to_right = w - x_max  # 包含所有目标框的最大右移动距离
    d_to_top = y_min  # 包含所有目标框的最大上移动距离
    d_to_bottom = h - y_max  # 包含所有目标框的最大下移动距离

    x = random.uniform(-(d_to_left - 1) / 3, (d_to_right - 1) / 3)
    y = random.uniform(-(d_to_top - 1) / 3, (d_to_bottom - 1) / 3)

    M = np.float32([[1, 0, x], [0, 1, y]])  # x为向左或右移动的像素值,正为向右负为向左; y为向上或者向下移动的像素值,正为向下负为向上
    shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    return shift_img
img=cv2.imread(img_path)
img=shift_pic_bboxes(img)
cv2.imwrite('/mnt/e/file/ultralytics/usr_sc/out_2.jpg',img)
# show_pic(img)

    
    
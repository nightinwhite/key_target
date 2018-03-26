# coding:utf-8
import os
import numpy as np
import cv2
tmp_path = "/home/hp/Data/train_data/slice_imgs/"
img_names = os.listdir(tmp_path)
max_w = 0
max_h = 0
for name in img_names:
    tmp_data = cv2.imread(tmp_path + name)
    print(tmp_data.shape, name)
# print(max_w, max_h)# 1824, 950
# coding:utf-8
import os
import numpy as np
tmp_path = "/home/hp/Data/train_data/slice_box_anns/"
img_names = os.listdir(tmp_path)
max_w = 0
max_h = 0
for name in img_names:
    tmp_data = np.load(tmp_path + name)
    for tmp_box in tmp_data:
        if tmp_box[2] > max_w:
            max_w = tmp_box[2]
        if tmp_box[3] > max_h:
            max_h = tmp_box[3]
        print (max_w, max_h)
print(max_w, max_h)# 1824, 950
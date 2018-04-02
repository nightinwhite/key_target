#coding:utf-8
import tensorflow as tf
from tensorflow.python.platform.flags import FLAGS
from SSD_Net import *
from build_loss import *
import common_flags
import variables_collection
from read_data import data_reader
import numpy as np
import sys
from utils import *
import cv2

def box_scale(k):
    out_pixel_list = [32, 64, 128, 256, 1024, 2048, 2048]
    return out_pixel_list[k]



def build_list_map(img):
    #用cv2读，注意修改下之前的norm
    img_h = img.shape[0]
    img_w = img.shape[1]
    box_ratios = [1.0, 1.0, 2.0, 3.0, 1.0/2.0, 1.0/3.0]
    out_shape_methods = [get_out1_shape, get_out2_shape, get_out3_shape, get_out4_shape, get_out5_shape, get_out6_shape]
    res_list_map = []
    for deep_num, shape_method in enumerate(out_shape_methods):
        # print(deep_num)
        tmp_h, tmp_w = shape_method(img_h, img_w)
        for tmp_y in range(tmp_h):
            for tmp_x in range(tmp_w):
                s_k = box_scale(deep_num)
                s_k1 = box_scale(deep_num + 1)
                for b_idx, br in enumerate(box_ratios):
                    tmp_use_scale = s_k
                    if deep_num != 0 and b_idx == 0:
                        tmp_use_scale = np.sqrt(s_k * s_k1)

                    default_w = tmp_use_scale * br
                    default_h = tmp_use_scale / br
                    # print(tmp_use_scale, np.sqrt(s_k * s_k1), default_w, default_h)
                    c_x = (tmp_x + 0.5) / float(tmp_w) * img_w
                    c_y = (tmp_y + 0.5) / float(tmp_h) * img_h
                    # print([c_x, c_y, default_w, default_h, tmp_w, tmp_h], len(res_list_map))
                    res_list_map.append([c_x, c_y, default_w, default_h, tmp_w, tmp_h])
    return res_list_map


class_color_map = {0:  [255, 0, 0] # blue
                   ,1: [0, 255, 0]
                   ,2: [0, 0, 255] # red
                   ,3: [255, 255, 0] #liang lan
                   ,4: [255, 0, 255] #fen
                   ,5: [0, 255, 255] #huang
                   ,6: [255, 255, 255]# bai
                   ,7: [255, 255, 255]# bai
                   }
# 迷彩车辆 0 ，迷彩建筑 1， 迷彩其它 2，
# 非迷彩车辆 3 ，非迷彩建筑 4， 非迷彩其它 5， 自然背景 6


test_imgs = []
test_name = "mn388_8_0_"
test_img = cv2.imread("/home/hp/Data/train_data/slice_imgs/{0}.png".format(test_name))
test_ann = np.load("/home/hp/Data/train_data/slice_box_anns/{0}.npy".format(test_name))
test_class = np.load("/home/hp/Data/train_data/slice_class_anns/{0}.npy".format(test_name))
test_train_ann =np.load("/home/hp/Data/train_data/train_box_anns_new/{0}.npy".format(test_name))
test_train_class = np.load("/home/hp/Data/train_data/train_class_anns_new/{0}.npy".format(test_name))
test_imgs.append(test_img)

for i in range(len(test_imgs)):
    tmp_img = test_imgs[i]
    tmp_img_w = tmp_img.shape[1]
    tmp_img_h = tmp_img.shape[0]
    tmp_list_map = build_list_map(tmp_img)
    tmp_img2 = np.copy(tmp_img)
    for m, res_box in enumerate(test_ann):
        if test_class[m]!=7:
            tmp_color = class_color_map[test_class[m]]
            tmp_res_box = res_box[:]
            print(tmp_res_box)
            tmp_res_box[4] = tmp_res_box[4] * np.pi
            tmp_contour = mrec_centre_To_mrec_corners_L(res_box)
            tmp_contour = np.asarray(tmp_contour)
            tst_contours = [tmp_contour]
            tmp_img2 = cv2.drawContours(tmp_img2, tst_contours, -1, tmp_color, 2)
    cv2.imshow("1", tmp_img2)
    cv2.waitKey()
    cv2.destroyAllWindows()







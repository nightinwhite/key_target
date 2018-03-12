# coding:utf-8
import numpy as np
from tensorflow.python.platform.flags import FLAGS
from utils import *
import os
import cv2
import common_flags

def box_scale(k):
    s_min = 0.1
    s_max = 0.95
    m = 6.0

    s_k = s_min + (s_max - s_min)*(k - 1.0)/(m - 1.0)
    return s_k

def build_label_box(img, anns):
    #用cv2读，注意修改下之前的norm
    img_h = img.shape[0]
    img_w = img.shape[1]
    box_ratios = [1.0, 1.0, 2.0, 3.0, 1.0/2.0, 1.0/3.0]
    res_box_list = []
    res_class_list = []
    res_box_mask = []
    res_logist_length = 0
    class_num = FLAGS.class_num
    count_i = 0
    out_shape_methods = [get_out1_shape, get_out2_shape, get_out3_shape, get_out4_shape, get_out5_shape, get_out6_shape]
    for deep_num, shape_method in enumerate(out_shape_methods):
        # print(deep_num)
        tmp_h, tmp_w = shape_method(img_h, img_w)
        res_logist_length += tmp_w*tmp_h
        for tmp_y in range(tmp_h):
            for tmp_x in range(tmp_w):
                s_k = box_scale(deep_num + 1)
                s_k1 = box_scale(deep_num + 2)
                if deep_num == 0:
                    tmp_scale = 0.07
                else:
                    tmp_scale = s_k
                for b_idx, br in enumerate(box_ratios):
                    count_i += 1
                    tmp_use_scale = tmp_scale
                    if deep_num != 0 and b_idx == 0:
                        tmp_use_scale = np.sqrt(s_k * s_k1)

                    default_w = tmp_use_scale * np.sqrt(br)
                    default_h = tmp_use_scale / np.sqrt(br)
                    # print(tmp_use_scale, np.sqrt(s_k * s_k1), default_w, default_h)
                    c_x = (tmp_x + 0.5) / float(tmp_w)
                    c_y = (tmp_y + 0.5) / float(tmp_h)
                    #这里源代码坐标是反的
                    tmp_class = class_num
                    tmp_max_jac_value = 0
                    tmp_box = [0, 0, 0, 0]
                    for ann in anns:
                        a_class = ann[0]
                        a_x = ann[1][0]
                        a_y = ann[1][1]
                        a_w = ann[1][2]
                        a_h = ann[1][3]
                        a_x = (a_x + 0.5) / float(img_w)
                        a_y = (a_y + 0.5) / float(img_h)
                        a_w = (a_w) / float(img_w)
                        a_h = (a_h) / float(img_h)
                        #对坐标，长宽进行百分比处理
                        tmp_jac_value = calc_jaccard(rec_centre_To_rec_corner_L([c_x, c_y, default_w, default_h]), rec_centre_To_rec_corner_L([a_x, a_y, a_w, a_h]))
                        # print ([c_x, c_y, default_w, default_h],[a_x, a_y, a_w, a_h], tmp_jac_value)
                        if tmp_jac_value > tmp_max_jac_value:
                            tmp_max_jac_value = tmp_jac_value
                            tmp_class = a_class
                            tmp_box = [a_x, a_y, a_w, a_h]
                    # print(tmp_max_jac_value, c_x, c_y, default_w, default_h)
                    if tmp_max_jac_value < 0.5:
                        tmp_class = class_num
                        tmp_box = [0, 0, 0, 0]
                        tmp_mask = 1
                    else:
                        tmp_box = [tmp_box[0] - c_x, tmp_box[1] - c_y, tmp_box[2] - default_w, tmp_box[3] - default_h]
                        tmp_mask = 0
                    # print(tmp_box, tmp_mask, tmp_class)
                    res_box_list.append(tmp_box)
                    res_box_mask.append(tmp_mask)
                    res_class_list.append(tmp_class)
    return res_class_list, res_box_list, res_box_mask, res_logist_length

if __name__ == '__main__':
    common_flags.define()
    imgs_path = "/home/hp/Data/train_data/slice_imgs/"
    class_ann_path = "/home/hp/Data/train_data/slice_class_anns/"
    box_ann_path = "/home/hp/Data/train_data/slice_box_anns/"
    class_save_path = "/home/hp/Data/train_data/train_class_anns/"
    if os.path.exists(class_save_path) == False:
        os.mkdir(class_save_path)
    box_save_path = "/home/hp/Data/train_data/train_box_anns/"
    if os.path.exists(box_save_path) == False:
        os.mkdir(box_save_path)
    mask_save_path = "/home/hp/Data/train_data/train_box_masks/"
    if os.path.exists(mask_save_path) == False:
        os.mkdir(mask_save_path)
    length_save_path = "/home/hp/Data/train_data/train_logist_lengths/"
    if os.path.exists(length_save_path) == False:
        os.mkdir(length_save_path)
    img_names = os.listdir(imgs_path)
    for i, img_name in enumerate(img_names):
        print("{0}/{1}".format(i, len(img_names)))
        f_name = img_name.split(".")[0]
        tmp_img = cv2.imread(imgs_path + img_name)
        tmp_class_anns = np.load(class_ann_path + f_name + ".npy")
        tmp_box_anns = np.load(box_ann_path + f_name + ".npy")
        tmp_anns = []

        for i in range(len(tmp_box_anns)):
            tmp_anns.append([tmp_class_anns[i], tmp_box_anns[i]])
        res_class_list, res_box_list, res_box_mask, res_logist_length = build_label_box(tmp_img, tmp_anns)
        np.save(class_save_path + f_name + ".npy", np.asarray(res_class_list))
        np.save(box_save_path + f_name + ".npy", np.asarray(res_box_list))
        np.save(mask_save_path + f_name + ".npy", np.asarray(res_box_mask))
        np.save(length_save_path + f_name + ".npy", np.asarray(res_logist_length))



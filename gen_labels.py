# coding:utf-8
import numpy as np
from tensorflow.python.platform.flags import FLAGS
from utils import *
import os
import cv2
import common_flags
import pp
import sys



def box_scale(k):
    out_pixel_list = [32, 64, 128, 256, 1024, 2048, 2048]
    return out_pixel_list[k]

def build_label_box(img, anns, img_name):
    #用cv2读，注意修改下之前的norm
    img_h = img.shape[0]
    img_w = img.shape[1]
    need_jac_val = 0.5
    box_ratios = [1.0, 1.0, 2.0, 3.0, 1.0/2.0, 1.0/3.0]
    res_logist_length = 0
    class_num = 7
    out_shape_methods = [get_out1_shape, get_out2_shape, get_out3_shape, get_out4_shape, get_out5_shape, get_out6_shape]
    for shape_method in out_shape_methods:
        tmp_h, tmp_w = shape_method(img_h, img_w)
        res_logist_length += tmp_w * tmp_h * 6
    res_box_list = [None for i in range(res_logist_length)]
    res_class_list = [None for i in range(res_logist_length)]
    res_box_mask = [None for i in range(res_logist_length)]
    res_jac_value_list = [0 for i in range(res_logist_length)]
    for ann in anns:
        a_class = ann[0]
        a_x = ann[1][0]
        a_y = ann[1][1]
        a_w = ann[1][2]
        a_h = ann[1][3]
        a_angle = ann[1][4]
        tmp_max_jac_value = 0
        tmp_index = 0
        tmp_max_index = 0
        tmp_tst_info = []
        for deep_num, shape_method in enumerate(out_shape_methods):
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
                        c_x = (tmp_x + 0.5) / tmp_w * img_w /float(default_w)
                        c_y = (tmp_y + 0.5) / tmp_w * img_w /float(default_h)
                        # 这里源代码坐标是反的
                        tmp_class = class_num
                        tmp_a_x = a_x /float(default_w)
                        tmp_a_y = a_y /float(default_h)
                        tmp_a_w = (a_w) / float(default_w)
                        tmp_a_h = (a_h) / float(default_h)
                        # 对坐标，长宽进行百分比处理
                        tmp_jac_value = PIOU([c_x, c_y, 1, 1, 0], [tmp_a_x, tmp_a_y, tmp_a_w, tmp_a_h, a_angle])
                        # print(tmp_x, tmp_y, tmp_w, tmp_h,img_w, img_h)
                        # print([c_x*default_w, c_y*default_h, default_w, default_h],[a_x, a_y, a_w, a_h], tmp_jac_value, tmp_use_scale)
                        if tmp_jac_value > tmp_max_jac_value:
                            tmp_max_jac_value = tmp_jac_value
                            tmp_max_index = tmp_index
                            tmp_max_box = [tmp_a_x - c_x, tmp_a_y - c_y,tmp_a_w - 1, tmp_a_h - 1, a_angle/np.pi]
                            tmp_tst_info = [[c_x*default_w, c_y*default_h, default_w, default_h], [a_x, a_y, a_w, a_h, a_angle/np.pi]]
                        # print(tmp_max_jac_value, c_x, c_y, default_w, default_h)
                        tmp_box = [0, 0, 0, 0, 0]
                        if tmp_jac_value < need_jac_val:
                            tmp_class = class_num
                            tmp_box = [0, 0, 0, 0, 0]
                            tmp_mask = 1
                        else:
                            # print("test-------------------")
                            # print(tmp_box)
                            tmp_class = a_class
                            tmp_box = [tmp_a_x - c_x, tmp_a_y - c_y, tmp_a_w - 1, tmp_a_h - 1, a_angle/np.pi]
                            tmp_mask = 0

                            # print(tmp_box)
                            # print([c_x, c_y, 1, 1])
                            # print("test-------------------")
                        if res_jac_value_list[tmp_index] == 0 or res_jac_value_list[tmp_index] < tmp_jac_value:
                            # print(tmp_box,res_jac_value_list[tmp_index])
                            res_box_list[tmp_index] = [tmp_box[0], tmp_box[1], tmp_box[2], tmp_box[3], tmp_box[4]]
                            res_box_mask[tmp_index] = tmp_mask
                            res_class_list[tmp_index] = tmp_class
                            res_jac_value_list[tmp_index] = tmp_jac_value
                        tmp_index += 1
        if tmp_max_jac_value < need_jac_val:
            # print(tmp_max_jac_value, tmp_max_box)
            # print(tmp_tst_info)
            # 有可能会覆盖掉其他类
            res_box_list[tmp_max_index] = tmp_max_box
            res_box_mask[tmp_max_index] = 0
            res_class_list[tmp_max_index] = a_class
        print(res_box_list[tmp_max_index], res_class_list[tmp_max_index], res_jac_value_list[tmp_max_index])
        # print(tmp_tst_info)
    class_save_path = "/home/hp/Data/train_data/train_class_anns_new_angle/"
    box_save_path = "/home/hp/Data/train_data/train_box_anns_new_angle/"
    mask_save_path = "/home/hp/Data/train_data/train_box_masks_new_angle/"
    length_save_path = "/home/hp/Data/train_data/train_logist_lengths_new_angle/"
    np.save(class_save_path + img_name + ".npy", np.asarray(res_class_list))
    np.save(box_save_path + img_name + ".npy", np.asarray(res_box_list))
    np.save(mask_save_path + img_name + ".npy", np.asarray(res_box_mask))
    np.save(length_save_path + img_name + ".npy", np.asarray(res_logist_length))
    return res_class_list, res_box_list, res_box_mask, res_logist_length

if __name__ == '__main__':
    ppservers = ()
    job_server = pp.Server(8, ppservers=ppservers)

    common_flags.define()
    imgs_path = "/home/hp/Data/train_data/slice_imgs_angle/"
    class_ann_path = "/home/hp/Data/train_data/slice_class_anns_angle/"
    box_ann_path = "/home/hp/Data/train_data/slice_box_anns_angle/"
    class_save_path = "/home/hp/Data/train_data/train_class_anns_new_angle/"
    if os.path.exists(class_save_path) == False:
        os.mkdir(class_save_path)
    box_save_path = "/home/hp/Data/train_data/train_box_anns_new_angle/"
    if os.path.exists(box_save_path) == False:
        os.mkdir(box_save_path)
    mask_save_path = "/home/hp/Data/train_data/train_box_masks_new_angle/"
    if os.path.exists(mask_save_path) == False:
        os.mkdir(mask_save_path)
    length_save_path = "/home/hp/Data/train_data/train_logist_lengths_new_angle/"
    if os.path.exists(length_save_path) == False:
        os.mkdir(length_save_path)
    img_names = os.listdir(imgs_path)
    jobs = []
    # print(img_names)
    # threads = []
    # for i in range(thread_num):
    #     tmp_thread = label_thread("{0}_thread".format(i))
    #     tmp_thread.start()
    #     threads.append(tmp_thread)
    for i, img_name in enumerate(img_names[12222:]):
        print("{0}/{1}".format(i, len(img_names)))
        f_name = img_name.split(".")[0]
        tmp_img = cv2.imread(imgs_path + img_name)
        tmp_class_anns = np.load(class_ann_path + f_name + ".npy")
        tmp_box_anns = np.load(box_ann_path + f_name + ".npy")
        tmp_anns = []

        for i in range(len(tmp_box_anns)):
            tmp_anns.append([tmp_class_anns[i], tmp_box_anns[i]])
        jobs.append(job_server.submit((build_label_box),(tmp_img, tmp_anns, f_name),(get_out1_shape, get_out2_shape, get_out3_shape, get_out4_shape, get_out5_shape, get_out6_shape,PIOU,box_scale), ("numpy as np", "shapely.geometry")))
        # res_class_list, res_box_list, res_box_mask, res_logist_length = build_label_box(tmp_img, tmp_anns, f_name)
    #     while label_thread.data_queue.qsize() >=10:
    #         time.sleep(0.01)
    #     label_thread.data_queue.put([tmp_img, tmp_anns, f_name])
    # while label_thread.data_queue.qsize()!=0:
    #     time.sleep(0.01)
    # label_thread.is_Done = True
    # for t in threads:
    #     t.join()
    for i,job in enumerate(jobs):
        result = job()
        print(i)


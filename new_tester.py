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

def NNM(all_box_list):
    all_box_list = sorted(all_box_list, key=lambda a:a[1][1])
    for i in range(len(all_box_list)):
        if all_box_list[i][2]:
            for j in range(i+1, len(all_box_list)):
                if all_box_list[j][2]:
                    if calc_jaccard(rec_centre_To_rec_corner_L(all_box_list[i][0]),
                                    rec_centre_To_rec_corner_L(all_box_list[j][0])) > 0:
                        all_box_list[j][2] = False
    res_box_list = []
    for i in range(len(all_box_list)):
        if all_box_list[i][2]:
            res_box_list.append(all_box_list[i])
    return res_box_list

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
sys.setrecursionlimit(1000000)
# hyper parameter
common_flags.define()
FLAGS.batch_size = 1
batch_size = FLAGS.batch_size
weight_decay = FLAGS.weight_decay
momentum = FLAGS.momentum
learning_rate = FLAGS.learning_rate
epoch = FLAGS.epoch
iteration = FLAGS.iteration
trust_rate = 0

# build model
train_imgs = tf.placeholder("float", [None, None, None, 3])  # batch_size, H, W, C
model = SSD_Net()
model.encoder(train_imgs,False)
model_pred = tf.nn.softmax(model.pred_labels)
# session and init
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# saver
saver = tf.train.Saver()
# saver.restore(sess, "models/e{0}_pixel_rate_background_0.5".format(98))#
saver.restore(sess, "models/e{0}_pixel_rate_loss_V3_with_angle".format(261))#


test_imgs = []
test_name = "mn470_13_2_"
test_img = cv2.imread("/home/hp/Data/train_data/slice_imgs_angle/{0}.png".format(test_name))
test_ann = np.load("/home/hp/Data/train_data/slice_box_anns_angle/{0}.npy".format(test_name))
test_class = np.load("/home/hp/Data/train_data/slice_class_anns_angle/{0}.npy".format(test_name))
test_train_ann =np.load("/home/hp/Data/train_data/train_box_anns_new_angle/{0}.npy".format(test_name))
test_train_class = np.load("/home/hp/Data/train_data/train_class_anns_new_angle/{0}.npy".format(test_name))
test_imgs.append(test_img)
res_preds,res_locs = sess.run([model_pred, model.pred_locs], feed_dict={train_imgs:test_imgs})

for i in range(len(test_imgs)):
    tmp_img = test_imgs[i]
    tmp_img_w = tmp_img.shape[1]
    tmp_img_h = tmp_img.shape[0]

    tmp_list_map = build_list_map(tmp_img)
    tmp_pred = res_preds[i]
    tmp_loc = res_locs[i]
    tmp_pred_class = np.argmax(tmp_pred, axis=-1)

    tmp_all_boxs = []
    for j in range(tmp_pred.shape[0]):
        if tmp_pred[j][tmp_pred_class[j]] > trust_rate and tmp_pred_class[j] != 7:
            tmp_default_box = tmp_list_map[j]
            tmp_default_c_x = tmp_default_box[0]
            tmp_default_c_y = tmp_default_box[1]
            tmp_default_w = tmp_default_box[2]
            tmp_default_h = tmp_default_box[3]
            tmp_default_map_w = tmp_default_box[4]
            tmp_default_map_h = tmp_default_box[5]
            tmp_real_c_x = tmp_default_c_x + tmp_loc[j][0] * tmp_default_w
            tmp_real_c_y = tmp_default_c_y + tmp_loc[j][1] * tmp_default_h
            tmp_real_w = tmp_default_w + tmp_loc[j][2] * tmp_default_w
            tmp_real_h = tmp_default_h + tmp_loc[j][3] * tmp_default_h
            tmp_real_angle = tmp_loc[j][4] * np.pi
            if tmp_real_w <0 or tmp_real_h < 0:
                continue
            tmp_all_boxs.append(
                [[tmp_real_c_x, tmp_real_c_y, tmp_real_w, tmp_real_h, tmp_real_angle],
                 [tmp_pred_class[j], tmp_pred[j][tmp_pred_class[j]]], True])
    tmp_res_boxs = tmp_all_boxs
    # tmp_res_boxs = tmp_all_boxs
    tmp_img1 = np.copy(tmp_img)
    for res_box in tmp_res_boxs:
        if res_box[1][0]!=7:
            tmp_color = class_color_map[res_box[1][0]]
            tmp_contour = mrec_centre_To_mrec_corners_L(res_box[0])
            tmp_contour = np.asarray(tmp_contour)
            # print(res_box[0])
            tst_contours = [tmp_contour]
            tmp_img1 = cv2.drawContours(tmp_img1, tst_contours, -1, tmp_color, 2)
    # cv2.imwrite("{0}pixel_rate_background_0.5.png".format(test_name), tmp_img1)
    cv2.imwrite("{0}pixel_rate_loss_V3.png".format(test_name), tmp_img1)
    cv2.imshow("0", tmp_img1)
    tmp_img2 = np.copy(tmp_img)
    for m, res_box in enumerate(test_ann):
        # print (res_box, test_class[m])
        if test_class[m]!=7:
            tmp_color = class_color_map[test_class[m]]
            res_box[4] = res_box[4] * np.pi
            tmp_contour = mrec_centre_To_mrec_corners_L(res_box)
            tmp_contour = np.asarray(tmp_contour)
            # print(res_box)
            tst_contours = [tmp_contour]
            tmp_img2 = cv2.drawContours(tmp_img2, tst_contours, -1, tmp_color, 2)
    cv2.imshow("1", tmp_img2)
    tmp_img3 = np.copy(tmp_img)
    for j in range(tmp_pred.shape[0]):
        if test_train_class[j] != 7:
            tmp_default_box = tmp_list_map[j]
            tmp_default_c_x = tmp_default_box[0]
            tmp_default_c_y = tmp_default_box[1]
            tmp_default_w = tmp_default_box[2]
            tmp_default_h = tmp_default_box[3]
            tmp_default_map_w = tmp_default_box[4]
            tmp_default_map_h = tmp_default_box[5]
            tmp_real_c_x = tmp_default_c_x + test_train_ann[j][0] * tmp_default_w
            tmp_real_c_y = tmp_default_c_y + test_train_ann[j][1] * tmp_default_h
            tmp_real_w = tmp_default_w + test_train_ann[j][2] * tmp_default_w
            tmp_real_h = tmp_default_h + test_train_ann[j][3] * tmp_default_h
            tmp_real_angle = test_train_ann[j][4] * np.pi
            tmp_color = class_color_map[test_train_class[j]]
            tmp_contour = mrec_centre_To_mrec_corners_L([tmp_real_c_x,tmp_real_c_y,tmp_real_w,tmp_real_h, tmp_real_angle])
            tmp_contour = np.asarray(tmp_contour)
            # print(res_box[0])
            tst_contours = [tmp_contour]
            tmp_img3 = cv2.drawContours(tmp_img3, tst_contours, -1, tmp_color, 2)
    cv2.imshow("2", tmp_img3)
    for j in range(tmp_pred.shape[0]):
        if test_train_class[j] != 7:
            print(test_train_ann[j], tmp_loc[j])
            print(test_train_class[j], tmp_pred_class[j], tmp_pred[j][tmp_pred_class[j]])
    cv2.waitKey()
    cv2.destroyAllWindows()







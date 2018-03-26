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
    s_min = 0.1
    s_max = 0.95
    m = 6.0

    s_k = s_min + (s_max - s_min)*(k - 1.0)/(m - 1.0)
    return s_k

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
                s_k = box_scale(deep_num + 1)
                s_k1 = box_scale(deep_num + 2)
                if deep_num == 0:
                    tmp_scale = 0.07
                else:
                    tmp_scale = s_k
                for b_idx, br in enumerate(box_ratios):
                    tmp_use_scale = tmp_scale
                    if deep_num != 0 and b_idx == 0:
                        tmp_use_scale = np.sqrt(s_k * s_k1)

                    default_w = tmp_use_scale * np.sqrt(br)
                    default_h = tmp_use_scale / np.sqrt(br)
                    # print(tmp_use_scale, np.sqrt(s_k * s_k1), default_w, default_h)
                    c_x = (tmp_x + 0.5) / float(tmp_w)
                    c_y = (tmp_y + 0.5) / float(tmp_h)
                    print([c_x, c_y, default_w, default_h, tmp_w, tmp_h], len(res_list_map))
                    res_list_map.append([c_x, c_y, default_w, default_h, tmp_w, tmp_h])
    return res_list_map

class_color_map = {0:  [255, 0, 0] # blue
                   ,1: [0, 255, 0] # green
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
trust_rate = 0.95

# build model
train_imgs = tf.placeholder("float", [None, None, None, 3])  # batch_size, H, W, C
model = SSD_Net()
model.encoder(train_imgs,)
model_pred = tf.nn.softmax(model.pred_labels)
# session and init
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# saver
saver = tf.train.Saver()
saver.restore(sess, "models/e{0}".format(56))

test_imgs = []
test_img = cv2.imread("/home/hp/Data/train_data/slice_imgs/mn3_3_0_.png")
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
    for j in range(tmp_pred.shape[0]):
        if tmp_pred[j][tmp_pred_class[j]] > trust_rate :
            tmp_color = class_color_map[tmp_pred_class[j]]
            tmp_default_box = tmp_list_map[j]
            tmp_default_c_x = tmp_default_box[0]
            tmp_default_c_y = tmp_default_box[1]
            tmp_default_w = tmp_default_box[2]
            tmp_default_h = tmp_default_box[3]
            tmp_default_map_w = tmp_default_box[4]
            tmp_default_map_h = tmp_default_box[5]
            print(j, "------------------------------------------")
            print(tmp_default_c_x, tmp_default_c_y, tmp_default_w, tmp_default_h, tmp_default_map_w, tmp_default_map_h)
            tmp_real_c_x = (tmp_default_c_x + tmp_loc[j][0])*tmp_img_w
            tmp_real_c_y = (tmp_default_c_y + tmp_loc[j][1]) * tmp_img_h
            tmp_real_w = (tmp_default_w + tmp_loc[j][2]) * tmp_img_w
            tmp_real_h = (tmp_default_h + tmp_loc[j][3])* tmp_img_h
            print(tmp_loc[j])
            print(tmp_real_c_x, tmp_real_c_y, tmp_real_w, tmp_real_h)
            tmp_contour = rec_centre_To_corners(tmp_real_c_x, tmp_real_c_y, tmp_real_w, tmp_real_h)
            print(tmp_contour)
            tmp_contour = np.asarray(tmp_contour)
            tst_contours = [tmp_contour]
            tmp_img = cv2.drawContours(tmp_img, tst_contours, -1, tmp_color, 2)
            print(j, "------------------------------------------")
    cv2.imshow("0", tmp_img)
    cv2.waitKey()
    cv2.destroyAllWindows()







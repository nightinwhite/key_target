import tensorflow as tf
from tensorflow.python.platform.flags import FLAGS
from SSD_Net import *
from build_loss import *
import common_flags
import variables_collection
from read_data import data_reader
import numpy as np
import sys
import time
import cv2

sys.setrecursionlimit(1000000)
# hyper parameter
common_flags.define()
FLAGS.batch_size = 16
batch_size = FLAGS.batch_size
weight_decay = FLAGS.weight_decay
momentum = FLAGS.momentum
learning_rate = FLAGS.learning_rate
epoch = FLAGS.epoch
iteration = FLAGS.iteration


# build model

# train_imgs = tf.placeholder("float", [32, 514, 257, 3])  # batch_size, H, W, C
# train_class_labels = tf.placeholder(tf.int32, [32, 8208])  # batch_size, logist_length
# train_loc_lables = tf.placeholder("float", [32, 8208, 4])  # batch_size, logist_length, box_corners
# train_mask = tf.placeholder("float", [32, 8208])  # batch_size, logist_length
# train_logist_length = tf.placeholder("float", [32])  # batch_size
train_imgs = tf.placeholder("float", [None, None, None, 3])  # batch_size, H, W, C
train_class_labels = tf.placeholder(tf.int32, [None, None])  # batch_size, logist_length
train_loc_lables = tf.placeholder("float", [None, None, 4])  # batch_size, logist_length, box_corners
train_mask = tf.placeholder("float", [None, None])  # batch_size, logist_length
train_logist_length = tf.placeholder("float", [None])  # batch_size
model = SSD_Net()
model.encoder(train_imgs,is_training=True)

# model loss
test_node= test_build_loss_v3(pred_labels=model.pred_labels, pred_locs=model.pred_locs,
                  anno_labels=train_class_labels, anno_locs=train_loc_lables,
                  anno_masks=train_mask, anno_logist_length=train_logist_length)

# class_acc = build_accuracy(pred_labels=model.pred_labels, anno_labels=train_class_labels)

# l2 Loss:
# for v in variables_collection.conv_weights:
#     total_loss += weight_decay*tf.nn.l2_loss(v)

# session and init
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
# saver
saver = tf.train.Saver()
saver.restore(sess, "models/e{0}_pixel_rate_loss_V3_background_0.5".format(67))

# data reader config
# test_name = "mn323_0_1_"
# test_img = cv2.imread("/home/hp/Data/train_data/slice_imgs/{0}.png".format(test_name))
# test_box_ann = np.load("/home/hp/Data/train_data/train_box_anns_new/{0}.npy".format(test_name))
# test_class_ann = np.load("/home/hp/Data/train_data/train_class_anns_new/{0}.npy".format(test_name))
# test_mask = np.load("/home/hp/Data/train_data/train_class_anns_new/{0}.npy".format(test_name))
# test_length = np.load("/home/hp/Data/train_data/train_logist_lengths_new/{0}.npy".format(test_name))
# test_imgs = [test_img]
# test_box_anns = [test_box_ann]
# test_class_anns = [test_class_ann]
# test_masks = [test_mask]
# test_lengths = [test_length]
# # --------------------------------------------------------------------------
imgs_path = "/home/hp/Data/train_data/slice_imgs"
class_path = "/home/hp/Data/train_data/train_class_anns_new"
boxs_path = "/home/hp/Data/train_data/train_box_anns_new"
masks_path = "/home/hp/Data/train_data/train_box_masks_new"
logist_length_path = "/home/hp/Data/train_data/train_logist_lengths_new"
data_reader = data_reader(imgs_path, class_path, boxs_path, masks_path, logist_length_path, batch_size)
# test_imgs1, test_class_anns1, test_box_anns1, test_masks1, test_lengths1 ,test_names= data_reader.test_read_data()
test_imgs, test_class_anns, test_box_anns, test_masks, test_lengths ,test_names= data_reader.test_read_data()


# test_names = ["mn390_16_0_","mn399_116_1_","mn391_108_1_", "mn689_33_1_",
#              "mn387_62_0_", "mn721_1_2_", "mn470_0_2_", "mn398_26_3_",
#              "mn1_74_2_", "mn380_29_1_", "mn1_35_0_", "mn250_1_1_",
#              "mn334_0_2_", "mn232_6_3_", "mn703_0_0_", "mn4_31_3_"]
# test_imgs = []
# test_box_anns = []
# test_class_anns = []
# test_masks = []
# test_lengths = []
# for test_name in test_names:
#     test_img = cv2.imread("/home/hp/Data/train_data/slice_imgs/{0}.png".format(test_name))
#     test_box_ann = np.load("/home/hp/Data/train_data/train_box_anns_new/{0}.npy".format(test_name))
#     test_class_ann = np.load("/home/hp/Data/train_data/train_class_anns_new/{0}.npy".format(test_name))
#     test_mask = np.load("/home/hp/Data/train_data/train_box_masks_new/{0}.npy".format(test_name))
#     test_length = np.load("/home/hp/Data/train_data/train_logist_lengths_new/{0}.npy".format(test_name))
#     test_imgs.append(test_img)
#     test_box_anns.append(test_box_ann)
#     test_class_anns.append(test_class_ann)
#     test_masks.append(test_mask)
#     test_lengths.append(test_length)

# # --------------------------------------------------------------------------

tmp_class_acc , tmp_all_class_acc= \
            sess.run(test_node,
                                feed_dict={train_imgs: test_imgs,
                                train_class_labels: test_class_anns,
                                train_loc_lables: test_box_anns,
                                train_mask: test_masks,
                                train_logist_length: test_lengths})
print (tmp_class_acc, tmp_all_class_acc)
# print((test_masks1 == test_mask).all())

# 69/169:class_l:0.26149699749315486, loc_l:0.09217584125752397, total_l:0.40807745386572447,acc:0.829909293002942, all_acc:0.7941104526028914

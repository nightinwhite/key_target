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

sys.setrecursionlimit(1000000)
# hyper parameter
common_flags.define()
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
train_loc_lables = tf.placeholder("float", [None, None, 5])  # batch_size, logist_length, box_corners
train_mask = tf.placeholder("float", [None, None])  # batch_size, logist_length
train_logist_length = tf.placeholder("float", [None])  # batch_size
model = SSD_Net()
model.encoder(train_imgs)

# model loss
class_loss, loc_loss, total_loss, class_acc,all_class_acc = build_loss_v3(pred_labels=model.pred_labels, pred_locs=model.pred_locs,
                  anno_labels=train_class_labels, anno_locs=train_loc_lables,
                  anno_masks=train_mask, anno_logist_length=train_logist_length)

# class_acc = build_accuracy(pred_labels=model.pred_labels, anno_labels=train_class_labels)

# l2 Loss:
for v in variables_collection.conv_weights:
    total_loss += weight_decay*tf.nn.l2_loss(v)
    tf.summary.histogram(v.name, v)

# summary :
for v in variables_collection.conv_bias:
    tf.summary.histogram(v.name, v)
tf.summary.scalar("class_acc", class_acc)
tf.summary.scalar("class_loss", class_loss)
tf.summary.scalar("loc_loss", loc_loss)
tf.summary.scalar("total_loss", total_loss)
all_summary = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter("Summary/")

# train_op:
train_op = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(total_loss)
# print("here!")
# session and init
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
# saver
saver = tf.train.Saver()
# saver.restore(sess, "models/e{0}_pixel_rate_loss_V3_with_angle".format(111))
# print("here")

# data reader config
imgs_path = "/home/hp/Data/train_data/slice_imgs"
class_path = "/home/hp/Data/train_data/train_class_anns_new_angle"
boxs_path = "/home/hp/Data/train_data/train_box_anns_new_angle"
masks_path = "/home/hp/Data/train_data/train_box_masks_new_angle"
logist_length_path = "/home/hp/Data/train_data/train_logist_lengths_new_angle"
data_reader = data_reader(imgs_path, class_path, boxs_path, masks_path, logist_length_path, batch_size)

# test node
# data_imgs, data_class_s, data_boxs, data_masks, data_logist_lengths = data_reader.read_data()
# test_node = test_build_loss_v3(pred_labels=model.pred_labels, pred_locs=model.pred_locs,
#                   anno_labels=train_class_labels, anno_locs=train_loc_lables,
#                   anno_masks=train_mask, anno_logist_length=train_logist_length)
# res_num, res_neg_mask, res_pos_mask ,res_act_mask, tmp_acc, tmp_all_acc, tmp_min_value= sess.run(test_node, feed_dict={train_imgs: data_imgs,
#                                 train_class_labels: data_class_s,
#                                 train_loc_lables: data_boxs,
#                                 train_mask: data_masks,
#                                 train_logist_length: data_logist_lengths})
# print(res_num)
# print(tmp_acc)
# print(tmp_all_acc)
# print(np.sum(res_pos_mask,1))
# print(np.sum(res_neg_mask,1))
# print(np.sum(res_act_mask,1))
# print(tmp_min_value)
# # res_pred = np.argmax(res, -1)
# # tst_index = 2
# # for i in range(res_pred.shape[1]):
# #     a = res_pred[tst_index, i]
# #     b = data_class_s[tst_index,i]
# #     print(a, b, a == b)
# print(res_mask.shape)
# tst_index = 0
# tmp_mask = res_mask[tst_index]
# for i in range(len(tmp_mask)):
#     if tmp_mask[i]!= 0:
#         print(tmp_mask[i])
#         print(res_label[tst_index][i])
#         print(data_class_s[tst_index][i])



# training
for e in range(epoch):
    e_class_loss = 0
    e_loc_loss = 0
    e_total_loss = 0
    e_class_acc = 0
    e_all_class_acc = 0
    for i in range(iteration):
        # start_time = time.time()
        data_imgs, data_class_s, data_boxs, data_masks, data_logist_lengths = data_reader.read_data()
        # print(time.time() - start_time)
        _, tmp_class_loss, tmp_loc_loss, tmp_total_loss, tmp_class_acc , tmp_all_class_acc= \
            sess.run([train_op, class_loss, loc_loss, total_loss, class_acc, all_class_acc],
                     feed_dict={train_imgs: data_imgs,
                                train_class_labels: data_class_s,
                                train_loc_lables: data_boxs,
                                train_mask: data_masks,
                                train_logist_length: data_logist_lengths})

        # if i % 10 == 0:
        #     summary_data = sess.run(all_summary, feed_dict={train_imgs: data_imgs,
        #                         train_class_labels: data_class_s,
        #                         train_loc_lables: data_boxs,
        #                         train_mask: data_masks,
        #                         train_logist_length: data_logist_lengths})
        #     summary_writer.add_summary(summary_data)

        e_class_loss += tmp_class_loss
        e_loc_loss += tmp_loc_loss
        e_total_loss += tmp_total_loss
        e_class_acc += tmp_class_acc
        e_all_class_acc += tmp_all_class_acc
        sys.stdout.write("                                                                                                      \r")
        sys.stdout.write("{4}/{5}:class_l:{0}, loc_l:{1}, total_l:{2},acc:{3}, all_acc:{6}".format(e_class_loss/(i+1), e_loc_loss/(i+1), e_total_loss/(i+1), e_class_acc/(i+1), e, i,e_all_class_acc/(i+1)))
        sys.stdout.flush()
    print("")
    saver.save(sess, "models/e{0}_pixel_rate_loss_V3_with_angle_model_0.75".format(e))


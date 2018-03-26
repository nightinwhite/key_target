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
from utils import *

sys.setrecursionlimit(1000000)
# hyper parameter
common_flags.define()
batch_size = FLAGS.batch_size
weight_decay = FLAGS.weight_decay
momentum = FLAGS.momentum
learning_rate = FLAGS.learning_rate
epoch = FLAGS.epoch
iteration = FLAGS.iteration
def box_scale(k):
    out_pixel_list = [32, 64, 128, 256, 1024, 2048, 2048]
    return out_pixel_list[k]

def NNM(all_box_list):
    all_box_list = sorted(all_box_list, key=lambda a:a[1][1])
    for i in range(len(all_box_list)):
        if all_box_list[i][2]:
            for j in range(i+1, len(all_box_list)):
                if all_box_list[j][2]:
                    if calc_jaccard(rec_centre_To_rec_corner_L(all_box_list[i][0]),
                                    rec_centre_To_rec_corner_L(all_box_list[j][0])) > 0.5:
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
model.encoder(train_imgs)
model_pred = tf.nn.softmax(model.pred_labels)
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
saver.restore(sess, "models/e{0}_pixel_rate_loss_V1".format(34))
# print("here")

# data reader config
imgs_path = "/home/hp/Data/train_data/slice_imgs"
class_path = "/home/hp/Data/train_data/train_class_anns_new"
boxs_path = "/home/hp/Data/train_data/train_box_anns_new"
masks_path = "/home/hp/Data/train_data/train_box_masks_new"
logist_length_path = "/home/hp/Data/train_data/train_logist_lengths_new"
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

test_names = ["mn390_16_0_","mn399_116_1_","mn391_108_1_", "mn689_33_1_",
             "mn387_62_0_", "mn721_1_2_", "mn470_0_2_", "mn398_26_3_",
             "mn1_74_2_", "mn380_29_1_", "mn1_35_0_", "mn250_1_1_",
             "mn334_0_2_", "mn232_6_3_", "mn703_0_0_", "mn4_31_3_"]
test_imgs = []
test_box_anns = []
test_class_anns = []
test_masks = []
test_lengths = []
for test_name in test_names:
    test_img = cv2.imread("/home/hp/Data/train_data/slice_imgs/{0}.png".format(test_name))
    test_box_ann = np.load("/home/hp/Data/train_data/train_box_anns_new/{0}.npy".format(test_name))
    test_class_ann = np.load("/home/hp/Data/train_data/train_class_anns_new/{0}.npy".format(test_name))
    test_mask = np.load("/home/hp/Data/train_data/train_box_masks_new/{0}.npy".format(test_name))
    test_length = np.load("/home/hp/Data/train_data/train_logist_lengths_new/{0}.npy".format(test_name))
    test_imgs.append(test_img)
    test_box_anns.append(test_box_ann)
    test_class_anns.append(test_class_ann)
    test_masks.append(test_mask)
    test_lengths.append(test_length)

# training
iteration = 10
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
    res_preds, res_locs = sess.run([model_pred, model.pred_locs], feed_dict={train_imgs: test_imgs})
    trust_rate = 0.9
    test_name = "loss_v1"
    print("test!")
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
                if tmp_real_w < 0 or tmp_real_h < 0:
                    continue
                tmp_all_boxs.append(
                    [[tmp_real_c_x, tmp_real_c_y, tmp_real_w, tmp_real_h],
                     [tmp_pred_class[j], tmp_pred[j][tmp_pred_class[j]]], True])
        tmp_res_boxs = tmp_all_boxs
        tmp_img1 = np.copy(tmp_img)
        for res_box in tmp_res_boxs:
            if res_box[1][0] != 7:
                tmp_color = class_color_map[res_box[1][0]]
                tmp_contour = rec_centre_To_corners_L(res_box[0])
                tmp_contour = np.asarray(tmp_contour)
                # print(res_box[0])
                tst_contours = [tmp_contour]
                tmp_img1 = cv2.drawContours(tmp_img1, tst_contours, -1, tmp_color, 2)
        # cv2.imwrite("{0}pixel_rate_background_0.5.png".format(test_name), tmp_img1)
        cv2.imwrite("{0}_{1}.png".format(test_name, i), tmp_img1)
        cv2.imshow("0", tmp_img1)
        tmp_res_boxs = NNM(tmp_all_boxs)
        tmp_img2 = np.copy(tmp_img)
        for res_box in tmp_res_boxs:
            if res_box[1][0] != 7:
                tmp_color = class_color_map[res_box[1][0]]
                tmp_contour = rec_centre_To_corners_L(res_box[0])
                tmp_contour = np.asarray(tmp_contour)
                # print(res_box[0])
                tst_contours = [tmp_contour]
                tmp_img2 = cv2.drawContours(tmp_img2, tst_contours, -1, tmp_color, 2)
        # cv2.imwrite("{0}pixel_rate_background_0.5.png".format(test_name), tmp_img1)
        cv2.imwrite("{0}_{1}_nnm.png".format(test_name, i), tmp_img2)
        cv2.imshow("1", tmp_img2)
        # tmp_img2 = np.copy(tmp_img)
        # for m, res_box in enumerate(test_ann):
        #     # print (res_box, test_class[m])
        #     if test_class[m] != 7:
        #         tmp_color = class_color_map[test_class[m]]
        #         tmp_contour = rec_centre_To_corners_L(res_box)
        #         tmp_contour = np.asarray(tmp_contour)
        #         # print(res_box)
        #         tst_contours = [tmp_contour]
        #         tmp_img2 = cv2.drawContours(tmp_img2, tst_contours, -1, tmp_color, 2)
        # cv2.imshow("1", tmp_img2)
        # tmp_img3 = np.copy(tmp_img)
        # for j in range(tmp_pred.shape[0]):
        #     if test_train_class[j] != 7:
        #         tmp_default_box = tmp_list_map[j]
        #         tmp_default_c_x = tmp_default_box[0]
        #         tmp_default_c_y = tmp_default_box[1]
        #         tmp_default_w = tmp_default_box[2]
        #         tmp_default_h = tmp_default_box[3]
        #         tmp_default_map_w = tmp_default_box[4]
        #         tmp_default_map_h = tmp_default_box[5]
        #         tmp_real_c_x = tmp_default_c_x + test_train_ann[j][0] * tmp_default_w
        #         tmp_real_c_y = tmp_default_c_y + test_train_ann[j][1] * tmp_default_h
        #         tmp_real_w = tmp_default_w + test_train_ann[j][2] * tmp_default_w
        #         tmp_real_h = tmp_default_h + test_train_ann[j][3] * tmp_default_h
        #         tmp_color = class_color_map[test_train_class[j]]
        #         tmp_contour = rec_centre_To_corners_L([tmp_real_c_x, tmp_real_c_y, tmp_real_w, tmp_real_h])
        #         tmp_contour = np.asarray(tmp_contour)
        #         # print(res_box[0])
        #         tst_contours = [tmp_contour]
        #         tmp_img3 = cv2.drawContours(tmp_img3, tst_contours, -1, tmp_color, 2)
        # cv2.imshow("2", tmp_img3)
        # for j in range(tmp_pred.shape[0]):
        #     if test_train_class[j] != 7:
        #         print(test_train_ann[j], tmp_loc[j])
        #         print(test_train_class[j], tmp_pred_class[j], tmp_pred[j][tmp_pred_class[j]])
        cv2.waitKey()
        cv2.destroyAllWindows()
    # saver.save(sess, "models/e{0}_pixel_rate_loss_V3_background_0.5".format(e))


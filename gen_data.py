# coding:utf-8
import cv2
import numpy as np
from parse_anns import *
from utils import *
import os

save_path = "/home/hp/Data/train_data"
if os.path.exists("{0}/slice_imgs/".format(save_path)) == False:
    os.mkdir("{0}/slice_imgs/".format(save_path))
if os.path.exists("{0}/slice_box_anns/".format(save_path)) == False:
    os.mkdir("{0}/slice_box_anns/".format(save_path))
if os.path.exists("{0}/slice_class_anns/".format(save_path)) == False:
    os.mkdir("{0}/slice_class_anns/".format(save_path))
base_img_w = 256
base_img_h = 256
splice_num = 4
wrong_anns_num = 0
boundary_anns_num = 0
anns_num = 0
def gen_single_data(img_path, anno_path):
    global anns_num
    global boundary_anns_num
    global wrong_anns_num
    anns = parse_xml(anno_path)
    img = cv2.imread(img_path)
    # cv2.imshow("0", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    img_h = img.shape[0]
    img_w = img.shape[1]
    img_name = img_path.split("/")[-1].split(".")[0]
    for ann_idx, ann in enumerate(anns):
        print(ann_idx)
        anns_num += 1
        box_ann = mrec_centre_To_rec_corner_L(ann[1])
        tmp_slice_w = base_img_w
        tmp_slice_h = base_img_h
        if box_ann[0] < 0 or box_ann[0] + box_ann[2] >= img_w or box_ann[1] < 0 or box_ann[1] + box_ann[3] >= img_h:
            boundary_anns_num += 1
            continue
        if ann[0] == -1:
            wrong_anns_num += 1
            continue
        while tmp_slice_w <= box_ann[2]:
            tmp_slice_w *= 2
        while tmp_slice_h <= box_ann[3]:
            tmp_slice_h *= 2
        if tmp_slice_w > 1024 or tmp_slice_h > 1024:
            continue
        w_change_len = tmp_slice_w - box_ann[2]
        h_change_len = tmp_slice_h - box_ann[3]
        x_bias = box_ann[0]
        y_bias = box_ann[1]
        w_change_len_b = np.max([(tmp_slice_w - box_ann[2]) - (img_w - x_bias - box_ann[2]), 0])
        w_change_len_e = np.min([w_change_len, x_bias])
        if w_change_len_b > w_change_len_e:
            continue
        h_change_len_b = np.max([(tmp_slice_h - box_ann[3]) - (img_h - y_bias - box_ann[3]), 0])
        h_change_len_e = np.min([h_change_len, y_bias])
        if h_change_len_b > h_change_len_e:
            continue
        for s in range(splice_num):
            # print(s)
            rand_x = x_bias - np.random.randint(w_change_len_b, w_change_len_e+1)
            rand_y = y_bias - np.random.randint(h_change_len_b, h_change_len_e+1)
            tmp_img = img[rand_y:rand_y + tmp_slice_h, rand_x:rand_x + tmp_slice_w, :]
            print(tmp_img.shape, tmp_slice_w, tmp_slice_h, w_change_len, h_change_len)
            tmp_class_anns = []
            tmp_box_anns = []
            tst_ann_box = [np.asarray([[rand_x, rand_y], [rand_x, rand_y + tmp_slice_h], [rand_x + tmp_slice_w, rand_y + tmp_slice_h], [rand_x + tmp_slice_w, rand_y], ])]
            for tmp_ann in anns:
                # tst_ann_box.append(np.asarray(mrec_centre_To_mrec_corners_L(tmp_ann[1])))
                tmp_box_ann = mrec_centre_To_rec_centre_L(tmp_ann[1])
                tmp_box_x = tmp_box_ann[0]
                tmp_box_y = tmp_box_ann[1]
                if rand_x<tmp_box_x and tmp_box_x < rand_x + tmp_slice_w and rand_y<tmp_box_y and tmp_box_y < rand_y + tmp_slice_h and tmp_ann[0] != -1:
                    tmp_class_anns.append(tmp_ann[0])
                    tmp_box_anns.append([tmp_box_ann[0] - rand_x, tmp_box_ann[1] - rand_y, tmp_box_ann[2], tmp_box_ann[3]])
            tmp_save_name = "{0}_{1}_{2}_".format(img_name, ann_idx, s)
            cv2.imwrite("{0}/slice_imgs/{1}.png".format(save_path, tmp_save_name), tmp_img)
            np.save("{0}/slice_box_anns/{1}.npy".format(save_path, tmp_save_name), np.asarray(tmp_box_anns))
            np.save("{0}/slice_class_anns/{1}.npy".format(save_path, tmp_save_name), np.asarray(tmp_class_anns))
            # print(tmp_box_anns)
            # tst_img = cv2.drawContours(img, tst_ann_box, -1, (255, 0, 0), 2)
            # cv2.imshow("1", tst_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
if __name__ == '__main__':
    all_anns_path = "/home/hp/Data/all_anns/"
    all_mn_path = "/home/hp/Data/key_target_data/micai/guonei/"
    all_anns_name = os.listdir(all_anns_path)
    for ann_name in all_anns_name:
        # print(ann_name)
        img_name = "mn"+ann_name.split("_")[1].split(".")[0]+".tif"
        # print(all_mn_path+img_name)
        gen_single_data(all_mn_path+img_name, all_anns_path+ann_name)
    print("------------------------------------------")
    print(anns_num)
    print(wrong_anns_num)
    print(boundary_anns_num)
    print("------------------------------------------")






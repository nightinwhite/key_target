from read_data import data_reader
import numpy as np

imgs_path = "/home/hp/Data/train_data/slice_imgs"
class_path = "/home/hp/Data/train_data/train_class_anns"
boxs_path = "/home/hp/Data/train_data/train_box_anns"
masks_path = "/home/hp/Data/train_data/train_box_masks"
logist_length_path = "/home/hp/Data/train_data/train_logist_lengths"

data_reader = data_reader(imgs_path, class_path, boxs_path, masks_path, logist_length_path, 32)

# while(True):
#     res = data_reader.read_data()
#     res_imgs = np.asarray(res[1])
#     print(res_imgs.shape)

tmp_imgs, tmp_classs, tmp_box, tmp_mask, tmp_lengths = data_reader.read_data()
tmp_index = 0
for i in range(tmp_classs[tmp_index].shape[0]):
    if tmp_classs[tmp_index][i]!=7:
        print("------------------------")
        print(tmp_classs[tmp_index][i], tmp_box[tmp_index][i], tmp_mask[tmp_index][i], tmp_lengths[tmp_index])
        print("------------------------")
    print(tmp_classs[tmp_index][i], tmp_box[tmp_index][i], tmp_mask[tmp_index][i], tmp_lengths[tmp_index])

import numpy as np
import cv2
import queue
import os
import threading
import _thread
import time
class data_reader():
    def __init__(self, img_path, class_path, box_path, mask_path, logist_length_path, batch_size):
        self.img_path = img_path
        self.class_path = class_path
        self.box_path = box_path
        self.mask_path = mask_path
        self.logist_length_path = logist_length_path
        self.batch_size = batch_size
        self.data_queue256_256 = queue.Queue(3 * batch_size)
        self.data_queue256_512 = queue.Queue(3 * batch_size)
        self.data_queue256_1024 = queue.Queue(3 * batch_size)
        self.data_queue512_256 = queue.Queue(3 * batch_size)
        self.data_queue512_512 = queue.Queue(3 * batch_size)
        self.data_queue512_1024 = queue.Queue(3 * batch_size)
        self.data_queue1024_256 = queue.Queue(3 * batch_size)
        self.data_queue1024_512 = queue.Queue(3 * batch_size)
        self.data_queue1024_1024 = queue.Queue(3 * batch_size)
        self.data_queue_list = [self.data_queue256_256, self.data_queue256_512, self.data_queue256_1024,
                                self.data_queue512_256, self.data_queue512_512, self.data_queue512_1024,
                                self.data_queue1024_256, self.data_queue1024_512, self.data_queue1024_1024,
                                ]
        key_img_names = os.listdir(img_path)
        self.key_names = []
        for tmp_name in key_img_names:
            self.key_names.append(tmp_name.split(".")[0])
        self.thread_lock = threading.Lock
        for t in range(4):
            _thread.start_new_thread(self.single_thread_fuc,())

    def read_single_data(self, key_name):
        tmp_img = cv2.imread(os.path.join(self.img_path, key_name+".png"))
        tmp_class = np.load(os.path.join(self.class_path, key_name+".npy"))
        tmp_box = np.load(os.path.join(self.box_path, key_name + ".npy"))
        tmp_mask = np.load(os.path.join(self.mask_path, key_name + ".npy"))
        tmp_logist_length = np.load(os.path.join(self.logist_length_path, key_name + ".npy"))
        return tmp_img, tmp_class, tmp_box, tmp_mask, tmp_logist_length,key_name

    def single_thread_fuc(self):
        while(True):
            tmp_key_index = np.random.random_integers(0, len(self.key_names) - 1)
            tmp_data = self.read_single_data(self.key_names[tmp_key_index])
            tmp_img = tmp_data[0]
            tmp_shape = tmp_img.shape
            if tmp_shape[0] == 256 and tmp_shape[1] == 256:
                if self.data_queue256_256.qsize() < 2*self.batch_size:
                    self.data_queue256_256.put(tmp_data)
            elif tmp_shape[0] == 256 and tmp_shape[1] == 512:
                if self.data_queue256_512.qsize() < 2*self.batch_size:
                    self.data_queue256_512.put(tmp_data)
            elif tmp_shape[0] == 256 and tmp_shape[1] == 1024:
                if self.data_queue256_1024.qsize() < 2*self.batch_size:
                    self.data_queue256_1024.put(tmp_data)
            elif tmp_shape[0] == 512 and tmp_shape[1] == 256:
                if self.data_queue512_256.qsize() < 2*self.batch_size:
                    self.data_queue512_256.put(tmp_data)
            elif tmp_shape[0] == 512 and tmp_shape[1] == 512:
                if self.data_queue512_512.qsize() < 2*self.batch_size:
                    self.data_queue512_512.put(tmp_data)
            elif tmp_shape[0] == 512 and tmp_shape[1] == 1024:
                if self.data_queue512_1024.qsize() < 2*self.batch_size:
                    self.data_queue512_1024.put(tmp_data)
            elif tmp_shape[0] == 1024 and tmp_shape[1] == 256:
                if self.data_queue1024_256.qsize() < 2*self.batch_size:
                    self.data_queue1024_256.put(tmp_data)
            elif tmp_shape[0] == 1024 and tmp_shape[1] == 512:
                if self.data_queue1024_512.qsize() < 2*self.batch_size:
                    self.data_queue1024_512.put(tmp_data)
            elif tmp_shape[0] == 1024 and tmp_shape[1] == 1024:
                if self.data_queue1024_1024.qsize() < 2*self.batch_size:
                    self.data_queue1024_1024.put(tmp_data)



    def read_data(self):
        select_queue = None
        select_index = None
        while(select_queue is None):
            can_use_queue = []
            for i, tmp_data_queue in enumerate(self.data_queue_list):
                if tmp_data_queue.qsize() > self.batch_size:
                    can_use_queue.append(i)
            if len(can_use_queue) == 0:
                time.sleep(0.01)
                continue
            else:
                # print(can_use_queue)
                select_index = np.random.random_integers(0, len(can_use_queue)-1)
                select_queue = self.data_queue_list[select_index]

        tmp_imgs = []
        tmp_class_s = []
        tmp_boxs = []
        tmp_masks = []
        tmp_logist_lengths = []

        for i in range(self.batch_size):
            tmp_data = select_queue.get()
            # print(" -------------------- ")
            # print(tmp_data[0].shape)
            # print(tmp_data[1].shape)
            # print(tmp_data[2].shape)
            # print(tmp_data[3].shape)
            # print(" -------------------- ")
            tmp_imgs.append(tmp_data[0])
            tmp_class_s.append(tmp_data[1])
            tmp_boxs.append(tmp_data[2])
            tmp_masks.append(tmp_data[3])
            tmp_logist_lengths.append(tmp_data[4])
            # print(tmp_data[5])
        tmp_imgs = np.asarray(tmp_imgs)
        tmp_class_s = np.asarray(tmp_class_s)
        tmp_boxs = np.asarray(tmp_boxs)
        tmp_masks = np.asarray(tmp_masks)
        tmp_logist_lengths = np.asarray(tmp_logist_lengths)
        return tmp_imgs, tmp_class_s, tmp_boxs, tmp_masks, tmp_logist_lengths

    def test_read_data(self):
        select_queue = None
        select_index = None
        while(select_queue is None):
            can_use_queue = []
            for i, tmp_data_queue in enumerate(self.data_queue_list):
                if tmp_data_queue.qsize() > self.batch_size:
                    can_use_queue.append(i)
            if len(can_use_queue) == 0:

                continue
            else:
                # print(can_use_queue)
                select_index = np.random.random_integers(0, len(can_use_queue)-1)
                select_queue = self.data_queue_list[select_index]

        tmp_imgs = []
        tmp_class_s = []
        tmp_boxs = []
        tmp_masks = []
        tmp_logist_lengths = []
        tmp_key_names = []
        for i in range(self.batch_size):
            tmp_data = select_queue.get()
            # print(" -------------------- ")
            # print(tmp_data[0].shape)
            # print(tmp_data[1].shape)
            # print(tmp_data[2].shape)
            # print(tmp_data[3].shape)
            # print(" -------------------- ")
            tmp_imgs.append(tmp_data[0])
            tmp_class_s.append(tmp_data[1])
            tmp_boxs.append(tmp_data[2])
            tmp_masks.append(tmp_data[3])
            tmp_logist_lengths.append(tmp_data[4])
            tmp_key_names.append(tmp_data[5])
        tmp_imgs = np.asarray(tmp_imgs)
        tmp_class_s = np.asarray(tmp_class_s)
        tmp_boxs = np.asarray(tmp_boxs)
        tmp_masks = np.asarray(tmp_masks)
        tmp_logist_lengths = np.asarray(tmp_logist_lengths)
        return tmp_imgs, tmp_class_s, tmp_boxs, tmp_masks, tmp_logist_lengths,tmp_key_names









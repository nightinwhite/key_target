import tensorflow as tf
from tensorflow.python.platform import flags
import numpy as np
from tensorflow.contrib.layers import xavier_initializer
import variables_collection
class Vgg16:
    def __init__(self,):
        self.flags = flags.FLAGS
        self.scope = "VGG16"
        self.encode_from_npy = self.flags.encode_from_npy
        self.vgg_weight = None
        if self.encode_from_npy:
            vgg_weight_path = self.flags.vgg_weight_npy_path#tmp path
            self.vgg_weight = np.load(vgg_weight_path, encoding='latin1').item()
        self.key_point = {}
        self.all_variables = []
        pass

    def input_norm(self, x):
        VGG_MEAN = [103.939, 116.779, 123.68]
        b, g, r = tf.split(x, 3, 3)
        x = tf.concat([b - VGG_MEAN[0],
                       g - VGG_MEAN[1],
                       r - VGG_MEAN[2]], 3)
        return x

    def _get_all_variables(self):
        all_variables = tf.global_variables()
        for v in all_variables:
            if self.scope in v.name:
                self.all_variables.append(v)

    def channel_scale(self, channels):
        return channels//4

    def encoder(self, x, is_training=False):
        with tf.variable_scope(self.scope, reuse=is_training):
            x = self.input_norm(x)
            self.key_point["x"] = x
            self.conv1_1 = self.conv_layer(x, [3, 3], 3, self.channel_scale(64),"conv1_1")
            self.conv1_2 = self.conv_layer(self.conv1_1, [3, 3], self.channel_scale(64), self.channel_scale(64), "conv1_2")
            self.pool1 = self.max_pool(self.conv1_2)
            self.key_point["pool1"] = self.pool1

            self.conv2_1 = self.conv_layer(self.pool1, [3, 3], self.channel_scale(64), self.channel_scale(128), "conv2_1")
            self.conv2_2 = self.conv_layer(self.conv2_1, [3, 3], self.channel_scale(128), self.channel_scale(128), "conv2_2")
            self.pool2 = self.max_pool(self.conv2_2)
            self.key_point["pool2"] = self.pool2

            self.conv3_1 = self.conv_layer(self.pool2, [3, 3], self.channel_scale(128), self.channel_scale(256), "conv3_1")
            self.conv3_2 = self.conv_layer(self.conv3_1, [3, 3], self.channel_scale(256), self.channel_scale(256), "conv3_2")
            self.conv3_3 = self.conv_layer(self.conv3_2, [3, 3], self.channel_scale(256), self.channel_scale(256), "conv3_3")
            self.pool3 = self.max_pool(self.conv3_3)
            self.key_point["pool3"] = self.pool3

            self.conv4_1 = self.conv_layer(self.pool3, [3, 3], self.channel_scale(256), self.channel_scale(512), "conv4_1")
            self.conv4_2 = self.conv_layer(self.conv4_1, [3, 3], self.channel_scale(512), self.channel_scale(512), "conv4_2")
            self.conv4_3 = self.conv_layer(self.conv4_2, [3, 3], self.channel_scale(512), self.channel_scale(512), "conv4_3")
            self.pool4 = self.max_pool(self.conv4_3)
            self.key_point["pool4"] = self.pool4

            self.conv5_1 = self.conv_layer(self.pool4, [3, 3], self.channel_scale(512), self.channel_scale(512), "conv5_1")
            self.conv5_2 = self.conv_layer(self.conv5_1, [3, 3], self.channel_scale(512), self.channel_scale(512), "conv5_2")
            self.conv5_3 = self.conv_layer(self.conv5_2, [3, 3], self.channel_scale(512), self.channel_scale(512), "conv5_3")
            # self.pool5 = self.max_pool(self.conv5_3)
            self.key_point["conv5_3"] = self.conv5_3

            self._get_all_variables()


    def conv_layer(self, input, kernel_size, channel_in, channel_out, scope):
        with tf.variable_scope(scope):
            if self.encode_from_npy:
                conv_weight = tf.get_variable("conv_weight", [kernel_size[0], kernel_size[1], channel_in, channel_out],
                                              initializer=tf.constant_initializer(value=self.get_conv_weight_from_npy(scope)))
                conv_bias = tf.get_variable("conv_bias", [channel_out], initializer=tf.constant_initializer(self.get_conv_bias_from_npy(scope)))

            else:
                conv_weight = tf.get_variable("conv_weight", [kernel_size[0], kernel_size[1], channel_in, channel_out],
                                              initializer=xavier_initializer())
                conv_bias = tf.get_variable("conv_bias", [channel_out], initializer=tf.constant_initializer(0))
            variables_collection.conv_weights.append(conv_weight)
            variables_collection.conv_bias.append(conv_bias)
            conv_res = tf.nn.conv2d(input, conv_weight, [1, 1, 1, 1], padding="SAME",)
            conv_res = tf.nn.bias_add(conv_res, conv_bias)
            conv_res = tf.nn.relu(conv_res)
            return conv_res

    def max_pool(self, input):
        return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def avg_pool(self, input):
        return tf.nn.avg_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


    def get_conv_weight_from_npy(self, name):
        # print(type(self.vgg_weight[name][0]))
        # print(self.vgg_weight[name][0])
        # print(self.vgg_weight[name][0].shape)
        tmp_data = self.vgg_weight[name][0]
        channels = tmp_data.shape[-1]
        channels = channels // 2
        return np.asarray(tmp_data[:, :, :, :channels])

    def get_conv_bias_from_npy(self, name):
        tmp_data = self.vgg_weight[name][1]
        channels = tmp_data.shape[-1]
        channels = channels // 2
        return np.asarray(tmp_data[:channels])


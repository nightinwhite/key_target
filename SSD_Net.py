# coding:utf-8
import tensorflow as tf
from tensorflow.python.platform import flags
from vgg16 import Vgg16
from tensorflow.contrib.layers import xavier_initializer
import variables_collection
class SSD_Net:
    def __init__(self):
        self.flags = flags.FLAGS
        self.scope = "SSG_Net"
        self.class_num = self.flags.class_num
        self.layer_boxs = [6, 6, 6, 6, 6, 6] # 源代码这部分第一个是3
        pass


    def channel_scale(self,channels):
        return channels//4

    def encoder(self, x, is_training=True, reuse=False):
        vgg_net = Vgg16()
        vgg_net.encoder(x)
        with tf.variable_scope(self.scope, reuse=False):
            self.conv6 = self.conv_layer(vgg_net.conv5_3, [3, 3], self.channel_scale(512), self.channel_scale(1024), is_training, scope="conv6")
            self.conv7 = self.conv_layer(self.conv6, [1, 1], self.channel_scale(1024), self.channel_scale(1024), is_training, scope="conv7")

            self.conv8_1 = self.conv_layer(self.conv7, [1, 1], self.channel_scale(1024), self.channel_scale(256), is_training, scope="conv8_1")
            self.conv8_2 = self.conv_layer(self.conv8_1, [3, 3], self.channel_scale(256), self.channel_scale(512), is_training, scope="conv8_2", stride=2)

            self.conv9_1 = self.conv_layer(self.conv8_2, [1, 1], self.channel_scale(512), self.channel_scale(128), is_training, scope="conv9_1")
            self.conv9_2 = self.conv_layer(self.conv9_1, [3, 3], self.channel_scale(128), self.channel_scale(256), is_training, scope="conv9_2",stride=2)

            self.conv10_1 = self.conv_layer(self.conv9_2, [1, 1], self.channel_scale(256), self.channel_scale(128), is_training, scope="conv10_1")
            self.conv10_2 = self.conv_layer(self.conv10_1, [3, 3], self.channel_scale(128), self.channel_scale(256), is_training, scope="conv10_2", stride=2)

            self.p11 = tf.nn.avg_pool(self.conv10_2, [1, 3, 3, 1], [1, 1, 1, 1], "SAME")

            all_class_num = self.class_num + 1

            self.out1 = self.conv_layer(vgg_net.conv4_3, [3, 3], self.channel_scale(512), self.layer_boxs[0]*(all_class_num + 4), is_training, scope="out1",
                                   activation_fn=None)
            self.out2 = self.conv_layer(self.conv7, [3, 3], self.channel_scale(1024), self.layer_boxs[1]* (all_class_num + 4), is_training, scope="out2",
                                   activation_fn=None)
            self.out3 = self.conv_layer(self.conv8_2, [3, 3], self.channel_scale(512), self.layer_boxs[2]* (all_class_num + 4), is_training, scope="out3",
                                   activation_fn=None)
            self.out4 = self.conv_layer(self.conv9_2, [3, 3], self.channel_scale(256), self.layer_boxs[3]* (all_class_num + 4), is_training, scope="out4",
                                   activation_fn=None)
            self.out5 = self.conv_layer(self.conv10_2, [3, 3], self.channel_scale(256), self.layer_boxs[4]* (all_class_num + 4), is_training, scope="out5",
                                   activation_fn=None)
            self.out6 = self.conv_layer(self.p11, [3, 3], self.channel_scale(256), self.layer_boxs[5]* (all_class_num + 4), is_training, scope="out6",
                                   activation_fn=None)

            self.outs = [self.out1, self.out2, self.out3, self.out4, self.out5, self.out6]
            self.outfs = []
            for i, out in enumerate(self.outs):
                tmp_outf = tf.reshape(out, [self.flags.batch_size, -1, all_class_num+4])
                self.outfs.append(tmp_outf)
            self.formatted_outs = tf.concat(self.outfs, 1)
            self.pred_labels = self.formatted_outs[:, :, :all_class_num]
            self.pred_locs = self.formatted_outs[:, :, all_class_num:]

    def batch_norm(self, x, is_training, decay=0.9, eps=1e-5):
        with tf.variable_scope("batch_norm"):
            shape = x.get_shape().as_list()
            assert len(shape) in [2, 3, 4]
            norm_num = shape[-1]
            beta = tf.get_variable(name="beta", shape=[norm_num],
                                   dtype=tf.float32, initializer=tf.constant_initializer(0))
            gamma = tf.get_variable(name="gamma", shape=[norm_num],
                                    dtype=tf.float32, initializer=tf.constant_initializer(1))
            if len(shape) == 2:
                batch_mean, batch_var = tf.nn.moments(x, [0])
            elif len(shape) == 3:
                batch_mean, batch_var = tf.nn.moments(x, [0, 1])
            else:
                batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
            ema = tf.train.ExponentialMovingAverage(decay=decay)

            def mean_var_update_when_training():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            def mean_var_update_when_not_training():
                return ema.average(batch_mean), ema.average(batch_var)
            if is_training:
                mean, var = mean_var_update_when_training()
            else:
                mean, var = mean_var_update_when_not_training()

        return tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)

    def conv_layer(self, input, kernel_size, channel_in, channel_out, is_training, scope, stride=1, activation_fn = tf.nn.relu):
        with tf.variable_scope(scope):
            conv_weight = tf.get_variable("conv_weight", [kernel_size[0], kernel_size[1], channel_in, channel_out],
                                          initializer=xavier_initializer())

            variables_collection.conv_weights.append(conv_weight)
            conv_res = tf.nn.conv2d(input, conv_weight, [1, stride, stride, 1], padding="SAME")

            conv_res = self.batch_norm(conv_res, is_training=is_training)
            if activation_fn is None:
                return conv_res
            else:
                return activation_fn(conv_res)

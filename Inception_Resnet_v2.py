import tensorflow as tf
import tensorflow.contrib.slim as slim
class Inception_resnet_v2:
    def __init__(self, num_classes=1001, is_training=True,
                 dropout_keep_prob=0.8, reuse=False,
                 scope='InceptionResnetV2'):
        self.num_classes = num_classes
        self.is_training = is_training
        self.dropout_keep_prob = dropout_keep_prob
        self.reuse = reuse
        self.scope = scope

    def encoder(self, x):
        with tf.variable_scope(name_or_scope=self.scope, reuse= self.reuse):
            slim.repeat()
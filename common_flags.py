#coding:utf-8
import tensorflow as tf
def define():
    tf.app.flags.DEFINE_bool("encode_from_npy", False, "是否从npy中恢复权值")
    tf.app.flags.DEFINE_string("vgg_weight_npy_path", "vgg16.npy", "vgg16的npy路径")
    tf.app.flags.DEFINE_integer("class_num", 7, "标签种类的数量（做物体检测内部自己会加负类")
    tf.app.flags.DEFINE_integer("negatives_scale", 3, "负类的倍数")
    tf.app.flags.DEFINE_integer("batch_size", 16, "批次数量")
    tf.app.flags.DEFINE_integer("epoch", 1000, "训练循环次数")
    tf.app.flags.DEFINE_integer("iteration", 1000, "每次循环迭代数")
    tf.app.flags.DEFINE_float("loss_alpha", 0.5, "平衡class与loc的loss值")
    tf.app.flags.DEFINE_float("learning_rate", 1e-2, "学习率")
    tf.app.flags.DEFINE_float("momentum", 0.9, "动量")
    tf.app.flags.DEFINE_float("weight_decay", 0.0001, "l2_loss")
    tf.app.flags.DEFINE_float("Conv_W_init_stddev", 0.1, "l2_loss")

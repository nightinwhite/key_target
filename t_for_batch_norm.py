import tensorflow as tf
img = tf.placeholder(tf.float32, [16, 32, 64, 3], name="img")
bn_img_mean, bn_img_variance = tf.nn.moments(img,axes=[0])
# with tf.variable_scope("batch_norm"):
#         shape = x.get_shape().as_list()
#         assert len(shape) in [2, 3, 4]
#         norm_num = shape[-1]
#         beta = tf.get_variable(name="beta", shape=[norm_num],
#                                dtype=tf.float32, initializer=tf.constant_initializer(0))
#         gamma = tf.get_variable(name="beta", shape=[norm_num],
#                                 dtype=tf.float32, initializer=tf.constant_initializer(1))
#         if len(shape) == 2:
#             batch_mean, batch_var = tf.nn.moments(x, [0])
#         elif len(shape) == 3:
#             batch_mean, batch_var = tf.nn.moments(x, [0, 1])
#         else:
#             batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
#         ema = tf.train.ExponentialMovingAverage(decay=decay)
#
#         def mean_var_update_when_training():
#             ema_apply_op = ema.apply([batch_mean, batch_var])
#             with tf.control_dependencies([ema_apply_op]):
#                 return tf.identity(batch_mean), tf.identity(batch_var)
#
#         def mean_var_update_when_not_training():
#             return ema.average(batch_mean), ema.average(batch_var)
#
#         mean, var = tf.cond(is_training, mean_var_update_when_training, mean_var_update_when_not_training)
#
#         return tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
def my_batch_norm(x):
    with tf.variable_scope("batch_norm"):
        input_shape = x.get_shape().as_list()
        if len(input_shape) == 2:
            batch_mean, batch_var = tf.nn.moments(x, [0])
        elif len(input_shape) == 3:
            batch_mean, batch_var = tf.nn.moments(x, [0, 1])
        else:
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])



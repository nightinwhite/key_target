import tensorflow as tf
a = tf.placeholder(tf.float32,[])
# bn_img_mean, bn_img_variance = tf.nn.moments(a,axes=[0])
ema = tf.train.ExponentialMovingAverage(decay= 0.9)
apply_op = ema.apply([a])
with tf.control_dependencies([apply_op]):
    res = tf.identity(a)
    # pass
avg_res = ema.average(a)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, "tst/test")
for i in range(10):
    print(sess.run([res,avg_res],feed_dict={a:i}))
    # saver.save(sess, "tst/test")
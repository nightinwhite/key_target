import tensorflow as tf

l = tf.placeholder(tf.float32, [0])
for i in range(l):
    res = l + 1

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print(sess.run([res], feed_dict={l: 3}))
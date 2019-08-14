import tensorflow as tf
x_ = tf.placeholder(tf.float32, shape=[4,2], name = 'x-input')

y_ = tf.placeholder(tf.float32, shape=[4,1], name = 'y-input')



Theta1 = tf.Variable(tf.random.uniform([2,2], -1, 1), name = "Theta1")

Theta2 = tf.Variable(tf.random.uniform([2,1], -1, 1), name = "Theta2")



Bias1 = tf.Variable(tf.zeros([2]), name = "Bias1")

Bias2 = tf.Variable(tf.zeros([1]), name = "Bias2")
with tf.name_scope("layer2") as scope:

	A2 = tf.sigmoid(tf.matmul(x_, Theta1) + Bias1)



with tf.name_scope("layer3") as scope:

	Hypothesis = tf.sigmoid(tf.matmul(A2, Theta2) + Bias2)



with tf.name_scope("cost") as scope:

	cost = tf.reduce_mean(( (y_ * tf.log(Hypothesis)) + 

		((1 - y_) * tf.log(1.0 - Hypothesis)) ) * -1)
XOR_X = [[0,0],[0,1],[1,0],[1,1]]

XOR_Y = [[0],[1],[1],[0]]


sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, "/tmp/model.ckpt")
print("Model restored.")
print('cost ', sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y}))

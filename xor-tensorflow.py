import tensorflow as tf
import numpy as np

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

X = tf.compat.v1.placeholder(tf.float32, shape=[4,2], name = 'X')
Y = tf.compat.v1.placeholder(tf.float32, shape=[4,1], name = 'Y')

W = tf.Variable(tf.random.uniform([2,2], -1, 1), name = "W")
w = tf.Variable(tf.random.uniform([2,1], -1, 1), name = "w")

c = tf.Variable(tf.zeros([4,2]), name = "c")
b = tf.Variable(tf.zeros([4,1]), name = "b")

with tf.name_scope("hidden_layer") as scope:
    h = tf.nn.relu(tf.add(tf.matmul(X, W),c))

with tf.name_scope("output") as scope:
    y_estimated = tf.sigmoid(tf.add(tf.matmul(h,w),b))

with tf.name_scope("loss") as scope:
    loss = tf.reduce_mean(tf.math.squared_difference(y_estimated, Y))

with tf.name_scope("train") as scope:
    train_step = tf.compat.v1.train.GradientDescentOptimizer(0.01).minimize(loss)

data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")

labels = np.array([[0],[1],[1],[0]], "float32")

init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    for epoch in range(10):
        for element in range(32):
            sess.run(train_step, {X: data, Y: labels})
        #if (epoch + 1) % 10 == 0:
        print('Epoch:', epoch + 1)
        print('loss:', sess.run(loss, {X: data, Y: labels}))
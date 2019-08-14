import tensorflow as tf
export_dir = "/tmp/model/0001"
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



with tf.name_scope("train") as scope:

	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)



XOR_X = [[0,0],[0,1],[1,0],[1,1]]

XOR_Y = [[0],[1],[1],[0]]
init = tf.global_variables_initializer()

with tf.Session() as sess:
  writer = tf.summary.FileWriter("./logs/xor_logs", sess.graph)
  #builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
  sess.run(init)
  for i in range(10001):
    
    sess.run(train_step, feed_dict={x_: XOR_X, y_: XOR_Y})

    if i % 10000 == 0:
        print('cost ', sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y}))
  #builder.add_meta_graph_and_variables(sess,
  #                                     [tag_constants.TRAINING],
  #                                     signature_def_map=foo_signatures,
  #                                     assets_collection=foo_assets,
  #                                     strip_default_attrs=True)
  simple_save(sess,
            export_dir,
            inputs={"x_": x, "y_": y},
            outputs={"z": z})
print("Model saved in path: %s" % export_dir)

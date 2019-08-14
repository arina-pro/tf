import numpy as np

import tensorflow as tf

train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(

      x={"x": train_data},

      y=train_labels,

      batch_size=100,

      num_epochs=20, #changed from None

      shuffle=True)

  mnist_classifier.train(

      input_fn=train_input_fn,

      steps=200, #changed from 20000

      hooks=[logging_hook])
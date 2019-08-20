import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

model = load_model('my_model.h5')

test = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
print('For', test, 'model predicted', model.predict(test, batch_size=1).round())
from tensorflow.compat.v1 import logging
from tensorflow.keras.models import load_model
from numpy import array

logging.set_verbosity(logging.ERROR)

model = load_model('my_model.h5')

test = array([[0,0],[0,1],[1,0],[1,1]], "float32")
print('For', test, 'model predicted', model.predict(test, batch_size=1).round())
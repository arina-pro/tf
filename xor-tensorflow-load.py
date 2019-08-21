from tensorflow.compat.v1 import logging, placeholder, get_variable, Session
from tensorflow import Graph, name_scope, nn, matmul, sigmoid, reduce_mean, Graph
from tensorflow.compat.v1.train import Saver
from numpy import array, square

logging.set_verbosity(logging.ERROR)

with Graph().as_default() as g:
    X = placeholder("float32", shape=[4,2], name = 'X')
    Y = placeholder("float32", shape=[4,1], name = 'Y')

    W = get_variable(shape=[2,2], name = 'W')
    w = get_variable(shape=[2,1], name = 'w')

    c = get_variable(shape=[4,2], name = 'c')
    b = get_variable(shape=[4,1], name = 'b')
    
    with name_scope("hidden_layer") as scope:
        h = nn.relu(matmul(X, W) + c)

    with name_scope("output") as scope:
        y_estimated = sigmoid(matmul(h,w) + b)

    with name_scope("loss") as scope:
        loss = reduce_mean(square(y_estimated - Y))
    
    with Session() as sess:
        saver = Saver()
        saver.restore(sess, "/tmp/model.ckpt")
        print("Model restored.")
        print('loss:', sess.run(loss, {X: array([[0,0],[0,1],[1,0],[1,1]], "float32"), Y: array([[0],[1],[1],[0]], "float32")}))
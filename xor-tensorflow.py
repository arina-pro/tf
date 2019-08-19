import tensorflow as tf #Импорт модуля ТФ
import numpy as np #Импорт модуля нампи

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #Не показываем лишние предупреждения

X = tf.compat.v1.placeholder(tf.float32, shape=[4,2], name = 'X') #Создаём объект для хранения входа
Y = tf.compat.v1.placeholder(tf.float32, shape=[4,1], name = 'Y') #Создаём объект для хранения выхода

W = tf.Variable(tf.random.uniform([2,2], -1, 1), name = "W") #Создаём переменную (наклон функции) с рандомным значением между -1 и 1 для входа
w = tf.Variable(tf.random.uniform([2,1], -1, 1), name = "w") #Создаём переменную (наклон функции) с рандомным значением между -1 и 1 для выхода

c = tf.Variable(tf.zeros([4,2]), name = "c") #Создаём переменную (смещение по оси координат) с нулями для входа
b = tf.Variable(tf.zeros([4,1]), name = "b") #Создаём переменную (смещение по оси координат) с нулями для выхода

with tf.name_scope("hidden_layer") as scope:
    h = tf.nn.relu(tf.add(tf.matmul(X, W),c)) #Слой применяет "выпрямитель" к функции ((входные данные * W) + c)

with tf.name_scope("output") as scope:
    y_estimated = tf.sigmoid(tf.add(tf.matmul(h,w),b)) #Слой применяет сигмоиду к функции ((предыдущий слой * w) + b)

with tf.name_scope("loss") as scope:
    loss = tf.reduce_mean(tf.math.squared_difference(y_estimated, Y)) #Слой считает среднее квадратичное ошибки между предыдущим слоем и выходными данными

with tf.name_scope("train") as scope:
    train_step = tf.compat.v1.train.GradientDescentOptimizer(0.01).minimize(loss) #Минимизируем ошибку при помощи метода градиентного спуска с шагом в 0.01 на каждом шаге

data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32") #Вход, массив нампи из 4 возможных конфигураций 0 и 1

labels = np.array([[0],[1],[1],[0]], "float32") #Выход, массив нампи из 4 соответствующих результатов операции xor

init = tf.compat.v1.global_variables_initializer() #Функция инициализации глобальных переменных

with tf.compat.v1.Session() as sess: #Открываем сессию
    sess.run(init) #Инициализируем переменные
    for epoch in range(10):
        for element in range(32):
            sess.run(train_step, {X: data, Y: labels}) #Обучаем модель на данных
        #if (epoch + 1) % 10 == 0:
        print('Epoch:', epoch + 1) #Эпоха обучения, каждые 32 шага
        print('loss:', sess.run(loss, {X: data, Y: labels})) #Выводим ошибку после каждой эпохи обучения
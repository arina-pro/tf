#Импорт модулей
from tensorflow.compat.v1 import logging, placeholder, global_variables_initializer, Session
from tensorflow import Variable, random, zeros, name_scope, nn, matmul, sigmoid, reduce_mean, Graph
from tensorflow.compat.v1.train import GradientDescentOptimizer, Saver
from numpy import array, square
from matplotlib import pyplot as plt

logging.set_verbosity(logging.ERROR) #Не показываем лишние предупреждения

with Graph().as_default(): #Открываем граф как главный

    X = placeholder("float32", shape=[4,2], name = 'X') #Создаём объект для хранения входа
    Y = placeholder("float32", shape=[4,1], name = 'Y') #Создаём объект для хранения выхода

    W = Variable(random.uniform([2,2], -1, 1), name = "W") #Создаём переменную (наклон функции) с рандомным значением между -1 и 1 для входа
    w = Variable(random.uniform([2,1], -1, 1), name = "w") #Создаём переменную (наклон функции) с рандомным значением между -1 и 1 для выхода

    c = Variable(zeros([4,2]), name = "c") #Создаём переменную (смещение по оси координат) с нулями для входа
    b = Variable(zeros([4,1]), name = "b") #Создаём переменную (смещение по оси координат) с нулями для выхода

    with name_scope("hidden_layer") as scope:
        h = nn.relu(matmul(X, W) + c) #Слой применяет "выпрямитель" к функции ((входные данные * W) + c)

    with name_scope("output") as scope:
        y_estimated = sigmoid(matmul(h,w) + b) #Слой применяет сигмоиду к функции ((предыдущий слой * w) + b)

    with name_scope("loss") as scope:
        loss = reduce_mean(square(y_estimated - Y)) #Слой считает среднее квадратичное ошибки между предыдущим слоем и выходными данными

    with name_scope("train") as scope:
        train_step = GradientDescentOptimizer(0.01).minimize(loss) #Минимизируем ошибку при помощи метода градиентного спуска с шагом в 0.01 на каждом шаге

    data = array([[0,0],[0,1],[1,0],[1,1]], "float32") #Вход, массив нампи из 4 возможных конфигураций 0 и 1

    labels = array([[0],[1],[1],[0]], "float32") #Выход, массив нампи из 4 соответствующих результатов операции xor

    init = global_variables_initializer() #Функция инициализации глобальных переменных

    saver = Saver() #Метод сохранения переменных

    with Session() as sess: #Открываем сессию
        sess.run(init) #Инициализируем переменные
        losses = []
        for epoch in range(100):
            for element in range(32):
                sess.run(train_step, {X: data, Y: labels}) #Обучаем модель на данных
            #if (epoch + 1) % 10 == 0:
            print('Epoch:', epoch + 1) #Эпоха обучения, каждые 32 шага
            cur_loss = sess.run(loss, {X: data, Y: labels})
            losses.append(cur_loss)
            print('loss:', cur_loss) #Выводим ошибку после каждой эпохи обучения
            save_path = saver.save(sess, "/tmp/model.ckpt") #Сохраняем чекпоинт каждую эпоху
        print("Model saved in path: %s" % save_path) #Где найти сохранённую "модель"
        plt.plot(range(1, 101), losses)
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        #plt.savefig('losses_plot.png')
        plt.show()
import tensorflow as tf #Подключаем модуль Тензорфлоу
from tensorflow.keras.models import load_model
import numpy as np #Импортируем нампи для создания массивов
import matplotlib.pyplot as plt #Импортируем модуль для визуализации

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #Убираем лишние предупреждения

model = tf.keras.Sequential([ #Создаём модель последовательных слоёв для обучения
#Функция активации этого слоя для одномерных данных - выпрямитель
tf.keras.layers.Dense(64, activation='relu', input_dim=2), #Второй слой принимает двухмерного инпут (вход+выход) и передаёт результаты 64 нейронам (спрятанный слой)
#Функция активации этого слоя для одномерных данных - сигмоида
tf.keras.layers.Dense(1, activation='sigmoid')]) #Третий слой принимает инфу из 64 нейронов спрятанного слоя и выдаёт результат на 1 нейрон

#Добавляем конфигурацию к модели
model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(0.01), #Используем в нашей модели оптимизатор Адама с шагом в 0.01
              loss='mse',       #Функция потери - среднее квадратичное ошибки
              metrics=['binary_accuracy'])  #Собираем метрику - бинарная точность, вероятность, что модель выберет "правильный" результат

data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32") #4 разных входных состояний в виде двухмерного массива

labels = np.array([[0],[1],[1],[0]], "float32") #4 выхода, соответствующим входным данным

hist = model.fit(data, labels, epochs=10, verbose=2) #Обучение модели, 10 эпох, размер набора по дефолту 32, печатаем 1 линию для каждой эпохи

fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8)) #Создаём 2 графика
fig.suptitle('Training Metrics') #Подзаголовок

axes[0].set_ylabel("Loss", fontsize=14) #Подписываем ось ординат 1 графика "Ошибка"
axes[0].plot(hist.history['loss']) #Вносим наши значения ошибки в график
axes[1].set_ylabel("Accuracy", fontsize=14) #Подписываем ось ординат 2 графика "Точность"
axes[1].set_xlabel("Epoch", fontsize=14) #Подписываем ось абсцисс "Эпохи"
axes[1].plot(hist.history['binary_accuracy']) #Вносим наши значения бинарной точности в график
plt.show() #Визуализируем

model.save('my_model.h5') #Сохраняем обученную модель

model.evaluate(data, labels, batch_size=1) #Оценка модели
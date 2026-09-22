#перевод из цельсия в фарангейты

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow import keras
from tensorflow.keras.layers import Dense



c = np.array([-40, -10, 0, 8, 15, 22, 38])
f = np.array([-40, 14, 32, 46, 59, 72, 100])

model = keras.Sequential() # какая модель 
model.add(Dense(units=1, input_shape=(1,), activation='linear')) #на вход один икс, не считая биас, сколько нейроонов, функция активайии- линейная
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.1)) #критерий ошибки как высчисляем ее, насколько сильно можно ошибиться(типо эпсилон)
history = model.fit(c, f, epochs=500, verbose=0) #начинаем обучение, выбираем сколько эпох будет прогоняться, 0 означает, что в консоль каждую эпоху не выводим
print("Обучение завершено")
print(model.predict(np.array([100])))#просим предсказать сколько будет 100 градусов
print(model.get_weights())#выводим всю инфу, веса, биас


plt.plot(history.history['loss'])#график уменьшения ошибок
plt.grid(True)
plt.show()
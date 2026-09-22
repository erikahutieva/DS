import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist         # библиотека базы выборок Mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten




(x_train, y_train), (x_test, y_test) = mnist.load_data() #из базы достаем тестовую и обучающую выборки


x_train = x_train / 255 #ускоряем обучение, чтобы веса нормально менялись
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)# на выходе получаем 10-мерный вектор, потому что у нас могут быть от 0 до 9 цифры
y_test_cat = keras.utils.to_categorical(y_test, 10)


model = keras.Sequential([Flatten(input_shape=(28, 28, 1)), Dense(128, activation='relu'), Dense(10, activation='softmax')]) #релу вносит нелинейность, чтобы нейронка лучше обучалась(если <0, то 0, если >0, то само число),softmax вероятность показывает

print(model.summary()) #выводит всю инфу, 0 потому что нет обучаемых параметров, 28*28*128+128,129*10
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])#подготавливаем к обучению, адам обновляет веса модели, чтобы минимизировать loss, процент процент правильных объектов
history=model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)#через каждые 32 раза обновляем веса, 5 эпох смешиваемся,20% для проверки обучилась ли нейросеть 
model.evaluate(x_test, y_test_cat)#проверяем на данных, которые нейросеть не видела



# отображение первых 25 изображений из обучающей выборки
plt.figure(figsize=(10,5))
for i in range(25):    
    plt.subplot(5,5,i+1)
    plt.xticks([])    
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)
plt.show()

n = 1
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)
print( res )
print( np.argmax(res) )
plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()

# Распознавание всей тестовой выборки
pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)
print(pred.shape)
print(pred[:20])
print(y_test[:20])


# Выделение неверных вариантов
mask = pred == y_test
print(mask[:10])
x_false = x_test[~mask]
y_false = x_test[~mask]
print(x_false.shape)
# Вывод первых 25 неверных результатов
plt.figure(figsize=(10,5))
for i in range(25):
    plt.subplot(5,5,i+1)    
    plt.xticks([])
    plt.yticks([])    
    plt.imshow(x_false[i], cmap=plt.cm.binary)
plt.show()

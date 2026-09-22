import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
import kagglehub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

#path = kagglehub.dataset_download("dansbecker/hot-dog-not-hot-dog") скачиваем хотдоги и не хотдоги

test_datagen = ImageDataGenerator(rescale=1./255)  # чтобы достать нормализованные изображения
train_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
    os.path.join('C:\Users\79897\Desktop\pic', 'test'), 
    target_size=(200, 200),  
    batch_size=32, 
    class_mode='binary',  # Бинарная классификация: 0 (Not Hot Dog) и 1 (Hot Dog)
    shuffle=True  # перемешиваем, чтобы не сохранить порядок
)

train_set = train_datagen.flow_from_directory(
    os.path.join('C:\Users\79897\Desktop\pic', 'train'), 
    target_size=(200, 200),  
    batch_size=32, 
    class_mode='binary',  
    shuffle=True  
)

x_test = test_set[0][0]
y_test = test_set[0][1]


#print(f"x_test shape: {x_test.shape}") #возвращает формы данных (32, 150*150,3 тоесть цветное изображение)
#print(f"y_test shape: {y_test.shape}")

x_train = train_set[0][0] #ускоряем обучение, чтобы веса нормально менялись
y_train = train_set[0][1]

#print("тест х",x_test) проверяем нормально ли заполнились массивы
#print("тест у",y_test)
#print("трейн х",x_train)
#print("трейн у",y_train)




y_train_cat = keras.utils.to_categorical(y_train, 2)# на выходе получаем 2-мерный вектор, потому что у нас могут быть от 0 или 1
y_test_cat = keras.utils.to_categorical(y_test, 2)


model = keras.Sequential([Flatten(input_shape=(200, 200, 3)), Dense(500, activation='relu'), Dense(2, activation='softmax')]) #релу вносит нелинейность, чтобы нейронка лучше обучалась(если <0, то 0, если >0, то само число),softmax вероятность показывает

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
for i in range(10):
    plt.subplot(5,5,i+1)    
    plt.xticks([])
    plt.yticks([])    
    plt.imshow(x_false[i], cmap=plt.cm.binary)
plt.show()

plt.plot(history.history['loss'])#график уменьшения ошибок
plt.grid(True)
plt.show()
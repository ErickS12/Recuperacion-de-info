# -*- coding: utf-8 -*-
"""CNN Fashion-MNIST: todas las gráficas se muestran juntas en ventanas distintas."""

# Si en tu computadora no se abren las ventanas, descomenta una de estas dos líneas:
# import matplotlib; matplotlib.use('TkAgg')
# import matplotlib; matplotlib.use('Qt5Agg')

from keras.datasets import fashion_mnist
(train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()

import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt

print('Training data shape : ', train_X.shape, train_Y.shape)
print('Testing data shape : ', test_X.shape, test_Y.shape)

# Find the unique numbers from the train labels
classes = np.unique(train_Y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

# Figura 1: una imagen de entrenamiento y una de prueba para observar
# el formato de los datos y su etiqueta real.
plt.figure(1, figsize=[5, 5])

plt.subplot(121)
plt.imshow(train_X[0, :, :], cmap='gray')
plt.title("Ground Truth : {}".format(train_Y[0]))

plt.subplot(122)
plt.imshow(test_X[0, :, :], cmap='gray')
plt.title("Ground Truth : {}".format(test_Y[0]))
plt.tight_layout()

# Agregamos un canal para que cada imagen tenga el formato que espera Conv2D:
# cantidad de imágenes, alto, ancho y canales.
train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)
print(train_X.shape, test_X.shape)

# Convertimos los píxeles a float32 y los normalizamos entre 0 y 1.
# Esto facilita y estabiliza el entrenamiento de la red neuronal.
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.

# Convertimos cada etiqueta numérica a codificación one-hot.
# Por ejemplo, la clase 5 se representa como [0,0,0,0,0,1,0,0,0,0].
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

# Comprobamos cómo cambió la primera etiqueta.
print('Original label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])

# Separamos el 20% de los datos de entrenamiento para validar el modelo
# mientras aprende, sin utilizar todavía los datos finales de prueba.
from sklearn.model_selection import train_test_split
train_X, valid_X, train_label, valid_label = train_test_split(
    train_X, train_Y_one_hot, test_size=0.2, random_state=13)

print(train_X.shape, valid_X.shape, train_label.shape, valid_label.shape)

import keras
from keras.models import Sequential, Model
from keras import Input
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU

batch_size = 64
epochs = 5  # 20
num_classes = 10

# Construimos una red neuronal convolucional para clasificar las 10 categorías
# de Fashion-MNIST.
fashion_model = Sequential()

# Detectamos bordes y características de la imagen, con 32 filtros de 3x3,
# activación lineal y padding same para mantener el tamaño de la imagen
fashion_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear',
                         input_shape=(28, 28, 1), padding='same'))
# Agregamos la unidad de activación LeakyReLU con un alpha de 0.1 para permitir
# que los valores negativos pasen a través de la red
fashion_model.add(LeakyReLU(alpha=0.1))
# MaxPooling2D de 2x2 con padding same para reducir la dimensionalidad de la imagen
fashion_model.add(MaxPooling2D((2, 2), padding='same'))

# Volvemos a repetir el proceso
fashion_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
fashion_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

# Flatten convierte la salida de la última capa convolucional en un vector unidimensional
fashion_model.add(Flatten())
# Capa densa con 128 unidades y activación lineal
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))

# Softmax en la capa de salida: clasificación multiclase con 10 clases,
# donde cada clase representa un tipo de ropa diferente
fashion_model.add(Dense(num_classes, activation='softmax'))

# Configuramos la función de pérdida, el optimizador y la métrica de precisión.
fashion_model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

fashion_model.summary()

# Entrenamos la red usando los datos de entrenamiento y medimos su desempeño
# con los datos de validación después de cada época.
fashion_train = fashion_model.fit(train_X, train_label,
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  verbose=1,
                                  validation_data=(valid_X, valid_label))

# Evaluamos el modelo con los datos de prueba para obtener la pérdida y la precisión
test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=0)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

accuracy = fashion_train.history['accuracy']
val_accuracy = fashion_train.history['val_accuracy']
loss = fashion_train.history['loss']
val_loss = fashion_train.history['val_loss']
epochs_range = range(len(accuracy))

# Figura 2: precisión (qué proporción de imágenes se clasificó bien).
plt.figure(2, figsize=[6, 5])
plt.plot(epochs_range, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs_range, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.tight_layout()

# Figura 3: pérdida (valores menores indican un mejor ajuste del modelo).
plt.figure(3, figsize=[6, 5])
plt.plot(epochs_range, loss, 'bo', label='Training loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.tight_layout()

# Generamos las probabilidades de pertenencia a cada una de las 10 clases
# para todas las imágenes de prueba.
predicted_classes = fashion_model.predict(test_X)

# Seleccionamos la clase con la probabilidad más alta para cada imagen.
predicted_classes = np.argmax(predicted_classes, axis=1)

print(predicted_classes.shape, test_Y.shape)

# Figura 4: primeras nueve imágenes clasificadas correctamente.
correct = np.where(predicted_classes == test_Y)[0]
print('Found %d correct labels' % len(correct))
plt.figure(4, figsize=(8, 8))
for i, idx in enumerate(correct[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_X[idx].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[idx], test_Y[idx]))
plt.tight_layout()

# Figura 5: primeras nueve imágenes clasificadas incorrectamente.
incorrect = np.where(predicted_classes != test_Y)[0]
print('Found %d incorrect labels' % len(incorrect))
plt.figure(5, figsize=(8, 8))
for i, idx in enumerate(incorrect[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_X[idx].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[idx], test_Y[idx]))
plt.tight_layout()

# Mostramos todas las gráficas juntas en ventanas distintas.
plt.show()
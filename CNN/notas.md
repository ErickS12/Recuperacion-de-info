# Notas: CNN para clasificar ropa (Fashion-MNIST)

## Qué hace el programa

Entrena una red neuronal convolucional (CNN) que mira imágenes de prendas de 28x28 píxeles en escala de grises y decide a cuál de 10 categorías pertenecen. Una CNN funciona como una serie de filtros que se deslizan sobre la imagen: primero detectan bordes, luego formas más complejas, y al final una capa de decisión elige la categoría.

## Las 10 categorías

| Clase | Prenda |
|---|---|
| 0 | Camiseta/top |
| 1 | Pantalón |
| 2 | Suéter |
| 3 | Vestido |
| 4 | Abrigo |
| 5 | Sandalia |
| 6 | Camisa |
| 7 | Tenis |
| 8 | Bolsa |
| 9 | Botín |

## Pasos del código

1. Carga el dataset: 60,000 imágenes de entrenamiento y 10,000 de prueba.
2. Muestra la forma de los datos y las clases.
3. Agrega una dimensión de canal para que `Conv2D` acepte las imágenes: de `(28, 28)` a `(28, 28, 1)`.
4. Normaliza los píxeles de 0-255 a 0-1, lo que estabiliza el entrenamiento.
5. Convierte las etiquetas a one-hot. Por ejemplo, la clase 9 queda como `[0,0,0,0,0,0,0,0,0,1]`.
6. Separa el 20% del entrenamiento para validación. El modelo practica con el 80% y se autoevalúa con ese 20% sin ver los datos de prueba.
7. Construye la red con 3 bloques `Conv2D` + `LeakyReLU` + `MaxPooling`, luego `Flatten`, una capa densa de 128 y una capa de salida `softmax` de 10 neuronas.
8. Entrena 5 épocas con lotes de 64 imágenes.
9. Evalúa con los datos de prueba, genera predicciones y dibuja las gráficas.

## Salida en la terminal (en orden)

### 1. Formas de los datos

```
Training data shape :  (60000, 28, 28) (60000,)
Testing data shape :  (10000, 28, 28) (10000,)
Total number of outputs :  10
Output classes :  [0 1 2 3 4 5 6 7 8 9]
```

60,000 imágenes de 28x28 para entrenar, 10,000 para probar, y 10 clases numeradas de 0 a 9.

### 2. Forma tras agregar el canal

```
(60000, 28, 28, 1) (10000, 28, 28, 1)
```

### 3. Comprobación del one-hot

La primera imagen es un botín (clase 9).

```
Original label: 9
After conversion to one-hot: [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
```

### 4. División entrenamiento/validación

```
(48000, 28, 28, 1) (12000, 28, 28, 1) (48000, 10) (12000, 10)
```

48,000 imágenes para entrenar y 12,000 para validar.

### 5. Resumen del modelo (`fashion_model.summary()`)

| Capa | Forma de salida | Parámetros |
|---|---|---|
| Conv2D (32 filtros) | (None, 28, 28, 32) | 320 |
| LeakyReLU | (None, 28, 28, 32) | 0 |
| MaxPooling2D | (None, 14, 14, 32) | 0 |
| Conv2D (64 filtros) | (None, 14, 14, 64) | 18,496 |
| LeakyReLU | (None, 14, 14, 64) | 0 |
| MaxPooling2D | (None, 7, 7, 64) | 0 |
| Conv2D (128 filtros) | (None, 7, 7, 128) | 73,856 |
| LeakyReLU | (None, 7, 7, 128) | 0 |
| MaxPooling2D | (None, 4, 4, 128) | 0 |
| Flatten | (None, 2048) | 0 |
| Dense (128) | (None, 128) | 262,272 |
| LeakyReLU | (None, 128) | 0 |
| Dense (10) | (None, 10) | 1,290 |

**Total de parámetros: 356,234**

La imagen se va achicando (28 → 14 → 7 → 4) mientras aumentan los filtros (32 → 64 → 128). `Flatten` la convierte en un vector de 2,048 valores (4x4x128). El formato exacto de la tabla puede cambiar según tu versión de Keras, pero las formas y el total de parámetros deben coincidir.

### 6. Entrenamiento

Una línea por época:

```
Epoch 1/5
750/750 ━━━━━━━━━━━━━━━━━━━━ ... accuracy: 0.xx - loss: 0.xx - val_accuracy: 0.xx - val_loss: 0.xx
...
Epoch 5/5
```

Hay 750 pasos por época porque 48,000 imágenes / 64 por lote = 750. Lo normal es que `accuracy` suba y `loss` baje de una época a otra. Con 5 épocas, la precisión de validación suele quedar cerca del 90%, aunque los números exactos cambian en cada corrida.

### 7. Evaluación final con datos de prueba

```
Test loss: 0.2x
Test accuracy: 0.9x
```

Este es el resultado que cuenta, porque son imágenes que el modelo nunca vio. Lo esperable es una precisión alrededor de 0.90 a 0.92, con valores distintos en cada ejecución.

### 8. Predicciones, aciertos y errores

```
(10000,) (10000,)
Found 9xxx correct labels
Found xxx incorrect labels
```

Los dos números suman 10,000. Si la precisión es de 0.91, aparecen unas 9,100 correctas y unas 900 incorrectas.

## Salida en imágenes (5 ventanas abiertas a la vez)

### Figura 1: imágenes de ejemplo

Dos imágenes en gris, una de entrenamiento y una de prueba. Ambas son un botín y ambas dicen `Ground Truth : 9`. Sirve para ver cómo luce un dato y su etiqueta real.

### Figura 2: precisión (Training and validation accuracy)

Los puntos azules son la precisión en entrenamiento y la línea azul es la de validación, una pareja por época. Debe subir con las épocas. Si los puntos siguen subiendo pero la línea se estanca, el modelo empieza a memorizar (sobreajuste).

### Figura 3: pérdida (Training and validation loss)

Misma idea, pero con el error. Debe bajar con las épocas. Si la pérdida de validación empieza a subir mientras la de entrenamiento baja, es otra señal de sobreajuste.

### Figura 4: aciertos

Cuadrícula de 3x3 con las primeras 9 imágenes bien clasificadas. Cada título dice `Predicted X, Class Y`, y en todas X es igual a Y.

### Figura 5: errores

Cuadrícula de 3x3 con las primeras 9 imágenes mal clasificadas. En todas X es distinto de Y. Normalmente son prendas parecidas entre sí, como camisa y camiseta, o suéter y abrigo. Es la figura más útil para entender dónde se equivoca el modelo.

## Notas rápidas

- Si ves advertencias en la terminal sobre `input_shape` o `alpha` en `LeakyReLU`, son avisos de versiones nuevas de Keras. No afectan el resultado.
- Las 5 ventanas aparecen hasta el final, cuando se ejecuta el único `plt.show()`. Mientras tanto solo verás texto en la terminal. Con el programa en pausa en ese punto, puedes revisar todas con calma y cerrarlas cuando termines.
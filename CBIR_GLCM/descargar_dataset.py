"""Descarga y selección reproducible de las 100 imágenes de la práctica.

Fuente: dataset WANG (Corel-1k), 10 categorías x 100 imágenes.
El repo indicado en la práctica (harshitaggarwal2010/Improving-the-CBIR-Process)
no incluye imágenes y sus enlaces de descarga están caídos, así que se usa
una copia en GitHub fijada a un commit concreto.

Criterio de selección:
  - 5 categorías: Beach, Bus, Dinosaurs, Flowers, Mountains
      * Beach y Mountains comparten colores (cielo azul) -> ponen a prueba al color
      * Bus y Flowers se distinguen sobre todo por el color
      * Dinosaurs tiene fondo uniforme (clase "fácil")
  - 20 imágenes por clase elegidas al azar con random.Random(SEMILLA).sample
    sobre los números 1..100 de cada categoría.

Uso:  python descargar_dataset.py
Genera: imagenes/<categoria>_NN.jpg  y  etiquetas.csv (archivo, categoria, origen)
"""
import csv
import random
import urllib.parse
import urllib.request
from pathlib import Path

SEMILLA = 42
POR_CLASE = 20
CATEGORIAS = ['Beach', 'Bus', 'Dinosaurs', 'Flowers', 'Mountains']
REPO = 'BuenoZ/Wang'
COMMIT = 'a4598992c254a342508a382595d99eda955be742'

BASE = Path(__file__).resolve().parent
DESTINO = BASE / 'imagenes'


def main():
    DESTINO.mkdir(exist_ok=True)
    rng = random.Random(SEMILLA)
    filas = []
    for cat in CATEGORIAS:
        numeros = sorted(rng.sample(range(1, 101), POR_CLASE))
        for k, n in enumerate(numeros, start=1):
            origen = f'{cat} ({n}).jpg'
            archivo = f'{cat.lower()}_{k:02d}.jpg'
            ruta = DESTINO / archivo
            if not ruta.exists():
                url = (f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/img/'
                       + urllib.parse.quote(origen))
                urllib.request.urlretrieve(url, ruta)
            filas.append((archivo, cat.lower(), origen))
        print(f'{cat:10} -> {numeros}')

    with open(BASE / 'etiquetas.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['archivo', 'categoria', 'origen'])
        w.writerows(filas)
    print(f'{len(filas)} imágenes en {DESTINO}')


if __name__ == '__main__':
    main()

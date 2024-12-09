# pip install --upgrade opencv-python opencv-contrib-python

import cv2 as cv
import os
import numpy as np
from time import time

dataset = 'C:/Users/Cristian/Pictures/CIA/reconocimiento1/data'
listaData = os.listdir(dataset)

ids = []
rostrosData = []

id = 0
timepoInicial = time()

for fila in listaData:
    
    rutaCompleta = f'{dataset}/{fila}'
    print('Iniciando lectura...')
    
    for archivo in os.listdir(rutaCompleta):
        
        print(f'Imagenes: {fila}/{archivo}')
        ids.append(id)
        rostrosData.append(cv.imread(f'{rutaCompleta}/{archivo}', 0))

    id = id + 1
    tiempoFinal = time()
    timepoLectura = tiempoFinal - timepoInicial
    print(f'Tiempo total: {timepoLectura}')

# Método correcto para versiones recientes de OpenCV
entrenamientoModelo = cv.face.LBPHFaceRecognizer.create()
#entrenamientoModelo1 = cv.face.EigenFaceRecognizer.create()
print('Iniciando el entrenamiento...')
entrenamientoModelo.train(rostrosData, np.array(ids))
print('Finalizando el entrenamiento...')
tiempoFinalEntrenamiento = time()
print(f'Tiempo de entrenamiento total: {tiempoFinalEntrenamiento - timepoInicial}')

entrenamientoModelo.write('Entrenamiento_createEigenFaceRecognizer.xml')
print('Entrenamiento guardado...')
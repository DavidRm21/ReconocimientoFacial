import cv2 as cv
import os 
import imutils

modelo = 'FotosCris'
ruta1 = 'C:/Users/Cristian/Pictures/CIA/reconocimiento1'
rutaCompleta = f'{ruta1}/{modelo}'
if not os.path.exists(rutaCompleta):
    os.makedirs(rutaCompleta)

camara = cv.VideoCapture(1)
ruidos = cv.CascadeClassifier('C:/Users/Cristian/Pictures/CIA/reconocimiento1/haarcascade_frontalface_default.xml')
id = 0

while True:
    respuesta,captura = camara.read()
    if respuesta == False: 
        break
    captura=imutils.resize(captura, width=640)

    grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    idCaptura = captura.copy()

    rostro = ruidos.detectMultiScale(grises, 1.4, 4)

    for(x, y, e1, e2) in rostro:
        cv.rectangle(captura, (x,y), (x+e1, y+e2), (255,0,0), 2)
        rostroCapturado = idCaptura[y:y+e1, x:x+e2]
        rostroCapturado=cv.resize(rostroCapturado, (160, 160), interpolation=cv.INTER_CUBIC)
        cv.imwrite(f'{rutaCompleta}/imagen_{id}.jpg', rostroCapturado)
        id=id+1

    cv.imshow('Resultado del rostro', captura)

    if id >= 351:
        break

camara.release()
cv.destroyAllWindows()
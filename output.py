import cv2 as cv
import os

dataset = 'C:/Users/Cristian/Pictures/CIA/reconocimiento1/data'
listaData = os.listdir(dataset)

entrenamientoModelo = cv.face.LBPHFaceRecognizer.create()
entrenamientoModelo.read('Entrenamiento LBPHFaceRecognizer.xml')
ruidos = cv.CascadeClassifier('C:/Users/Cristian/Pictures/CIA/reconocimiento1/haarcascade_frontalface_default.xml')

camara=cv.VideoCapture(1)

while True:
    _,captura = camara.read()
    grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    idCaptura = grises.copy()
    rostro = ruidos.detectMultiScale(grises, 1.3, 5)

    for(x, y, e1, e2) in rostro:
        rostroCapturado = idCaptura[y:y+e1, x:x+e2]
        rostroCapturado=cv.resize(rostroCapturado, (160, 160), interpolation=cv.INTER_CUBIC)
        resultado = entrenamientoModelo.predict(rostroCapturado)
        cv.putText(captura, f'{resultado}', (x,y-20), 1, 1.3, (0,255,0), 1, cv.LINE_AA)
        if resultado[1]<9000:
            cv.putText(captura, f'Desconocido', (x,y-5), 2, 1.3, (0,255,0), 1, cv.LINE_AA)
            cv.rectangle(captura, (x,y), (x+e1, y+e2), (255,0,0), 2)
        else:
            cv.putText(captura, f'{listaData[resultado[0]]}', (x,y-5), 2, 0.8, (0,255,0), 1, cv.LINE_AA)
            cv.rectangle(captura, (x,y), (x+e1, y+e2), (255,0,0), 2)
        

    cv.imshow("Resultados", captura)
    
    if cv.waitKey(1) == ord('s'):
        break

camara.release()
cv.destroyAllWindows()

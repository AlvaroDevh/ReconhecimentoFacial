import cv2

classificador = cv2.CascadeClassifier("cascade/haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(0)
amostra = 1
numamostras = 5
id = input ("Digite seu id ")
largura, altura = 220 , 220
print ("Capturando as faces")

while (True):
    conectado, frame = video.read()

    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = classificador.detectMultiScale(cinza,scaleFactor=1.5, minSize=(60,60))
    for(x, y, l ,a ) in face:
        cv2.rectangle(frame, (x,y) , (x + l, y + a) , (0,0,2569695), 2)
        if cv2.waitKey(1) & 0xFF== ord('q'):
            imagemFace = cv2.resize(cinza [y:y + a, x:x + l] , (largura, altura))
            cv2.imwrite ("fotos/pessoa." + str(id) + "." + str(amostra) + ".jpg" , imagemFace)
            print ("[foto" + str (amostra) + "capturada]")
            amostra +=1

  
    cv2.imshow("Face" , frame)
    cv2.waitKey(1)
    if(amostra >= numamostras + 1):
        break

print("Faces capturadas com sucesso")
video.release()
cv2.destroyAllWindows()
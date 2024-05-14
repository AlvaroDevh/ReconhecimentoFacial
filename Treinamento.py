import cv2
import os
import numpy as np

eigenface_dir = 'eigenface'
fisherface_dir = 'fisherface'
lbph_dir = 'lbph'

os.makedirs(eigenface_dir, exist_ok=True)
os.makedirs(fisherface_dir, exist_ok=True)
os.makedirs(lbph_dir, exist_ok=True)

# Criando os reconhecedores
eigenface = cv2.face.EigenFaceRecognizer_create(threshold=2)
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()

def getImagemComId():
    caminho = [os.path.join('Fotos', f) for f in os.listdir('Fotos')]
    faces = []
    ids = []

    for caminhoImagem in caminho:
        imagemFace = cv2.imread(caminhoImagem)
        # Convertendo a imagem para escala de cinza
        imagemFace = cv2.cvtColor(imagemFace, cv2.COLOR_BGR2GRAY)

        id = int(os.path.split(caminhoImagem)[-1].split('.')[1])
        ids.append(id)
        faces.append(imagemFace)

    return  np.array(ids), faces

ids, faces = getImagemComId()

print("Treinando Eigenface")
eigenface.setNumComponents(50)  # Definindo o número de componentes
eigenface.train(faces, ids)
eigenface.write(os.path.join(eigenface_dir, 'classificadorEigen.yml'))

print("Treinando Fisherface")
fisherface.train(faces, ids)
fisherface.write(os.path.join(fisherface_dir, 'classificadorFisherface.yml'))

print("Treinando LBPH")
lbph.train(faces, ids)
lbph.write(os.path.join(lbph_dir, 'classificadorLBPH.yml'))

print("Treinamento concluído")

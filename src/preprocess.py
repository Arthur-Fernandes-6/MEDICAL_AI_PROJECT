import cv2 as cv
import numpy as np #Representará a imagem como uma matriz.
import os #Servirá para trabalhar com caminhos de arquivos.

def carregar_imagem(caminho):
    imagem = cv.imread(caminho)

    if imagem is None:
        print("Imagem não encontrada!")
        return None

    return imagem

def redimensionar(imagem):
    pass


def normalizar(imagem):
    pass



imagem = carregar_imagem('datasets/yes/Y1.jpg')
print(imagem)





#img = cv.imread('datasets/Son_Gohan.jpeg')  vai ler o caminho da imagem 

#cv.imshow('Gohan', img)
#cv.waitKey(0)

#print(img.shape)
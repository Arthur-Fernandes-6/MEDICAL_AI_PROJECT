import cv2 as cv
import numpy as np #Representará a imagem como uma matriz.
import os #Servirá para trabalhar com caminhos de arquivos.

def carregar_imagem(caminho):
    pass

def redimensionar(imagem):
    pass

def normalizar(imagem):
    pass

img = cv.imread('datasets/Son_Gohan.jpeg')
cv.imshow('Gohan', img)
cv.waitKey(0)

print(img.shape)

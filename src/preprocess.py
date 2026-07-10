import cv2 as cv
import numpy as np #Representará a imagem como uma matriz.
import os #Servirá para trabalhar com caminhos de arquivos.

def carregar_imagem(caminho):
    imagem = cv.imread(caminho)
#recebe o caminho da imagem 
    if imagem is None:
        print("Imagem não encontrada!")
        return None

    return imagem


def redimensionar(imagem):
    if imagem is None:
        print("Imagem não encontrada")
        return None
    #receber a imagem
    altura, largura, _ = imagem.shape

    if altura > 224 or largura > 224:
        interpolacao = cv.INTER_AREA
#se a imagem for maior que 224 x 224 usar INTER_AREA
    elif altura < 224  or largura < 224:
        interpolacao = cv.INTER_LINEAR
    #se for menor usar INTER_LINEAR
    else:
        interpolacao = cv.INTER_LINEAR
#se for menor usar INTER_LINEAR
    imagem_redimensionada = cv.resize(imagem, (224, 224),interpolation = interpolacao)
    return imagem_redimensionada
#redimensionar para 224x224
#retornar a imagem redimensionada



def normalizar(imagem):
    if imagem is None:
        print("Imagem não encontrada")
        return None
    #normalizar a imagem para o intervalo [0, 1]
    imagem = imagem.astype(np.float32)
    #converter a imagem para float32 
    #tensorflow trabalha melhor com float32
    imagem_normalizada = imagem / 255.0
    return imagem_normalizada



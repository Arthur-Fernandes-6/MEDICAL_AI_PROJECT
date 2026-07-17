import os
import numpy as np
from preprocess import preprocessar_imagem

def carregar_dataset_v2(caminho_dataset, conjunto):
    x = []
    y = []

    mapeamento = {
        "glioma" : 1,
        "meningioma" : 1,
        "pituitary" : 1,
        "notumor" : 0
    }

    for nome_classe, rotulos in mapeamento.items():
        print(nome_classe)
        print(rotulos)

        caminhoClasse = os.path.join(caminho_dataset, conjunto, nome_classe)
        arquivos = os.listdir(caminhoClasse)

        for imagem in arquivos:
            caminho_classe = os.path.join(caminho_dataset, conjunto, )
            os.listdir(caminho_classe)
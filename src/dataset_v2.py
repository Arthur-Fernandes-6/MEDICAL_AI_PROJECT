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

    for nome_classe, rotulo in mapeamento.items():
        
        caminhoClasse = os.path.join(caminho_dataset, conjunto, nome_classe)
        arquivos = os.listdir(caminhoClasse)

        for nome_arquivo in arquivos:
            caminho_completo = os.path.join(caminhoClasse, nome_arquivo)

            imagem = preprocessar_imagem(caminho_completo)

            if imagem is None:
                continue
            
            x.append(imagem)
            y.append(rotulo)

    
    x_array = np.array(x)
    y_array = np.array(y)

    print(f"total das imagens: {len(x_array)}")

    return x_array, y_array
    
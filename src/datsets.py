import os
from preprocess import *

# função responsavel por carregar todo o dataset
def carregar_dataset(dataset_path):
    x = []
    y = []
    
    pasta_yes = os.path.join(dataset_path, 'yes')
    imagens_yes = os.listdir(pasta_yes)
    # processa as imagens com tumor
    for nome_arquivo in imagens_yes:
        caminho_completo = os.path.join(pasta_yes, nome_arquivo)
        #p athjoin junta a pasta mais os arquivos
        imagem = carregar_imagem(caminho_completo)

        if imagem is None:
            continue
        # pule a imagem que esta vazia e va para a proxima repetição do for

        imagem = redimensionar(imagem)
        imagem = normalizar(imagem)

        x.append(imagem)
        # lista com as imagens com tumor
        y.append(1)
        # rotulo para se tiver tumor

    pasta_no = os.path.join(dataset_path, 'no')
    imagens_no = os.listdir(pasta_no)    
# processa as imagens sem tumor
    for nome_arquivo in imagens_no:
        caminho_completo = os.path.join(pasta_no, nome_arquivo)
        imagem = carregar_imagem(caminho_completo)

        if imagem is None:
            continue

        imagem = redimensionar(imagem)
        imagem = normalizar(imagem)

        x.append(imagem)
        y.append(0)
        # rotulo para imagens que não tem tumor
    return x, y
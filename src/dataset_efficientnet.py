import os
import numpy as np

from preprocess_efficientnet import (
    preprocessar_imagem_efficientnet
)


EXTENSOES_VALIDAS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


def carregar_dataset_efficientnet(
    caminho_dataset,
    conjunto
):
    imagens = []
    rotulos = []

    mapeamento = {
        "glioma": 1,
        "meningioma": 1,
        "pituitary": 1,
        "notumor": 0
    }

    for nome_classe, rotulo in mapeamento.items():

        caminho_classe = os.path.join(
            caminho_dataset,
            conjunto,
            nome_classe
        )

        if not os.path.exists(caminho_classe):
            print(
                f"Pasta não encontrada: {caminho_classe}"
            )
            continue

        arquivos = os.listdir(caminho_classe)

        for nome_arquivo in arquivos:

            if not nome_arquivo.lower().endswith(
                EXTENSOES_VALIDAS
            ):
                continue

            caminho_completo = os.path.join(
                caminho_classe,
                nome_arquivo
            )

            imagem = preprocessar_imagem_efficientnet(
                caminho_completo
            )

            if imagem is None:
                continue

            imagens.append(imagem)
            rotulos.append(rotulo)

    x = np.array(
        imagens,
        dtype=np.float32
    )

    y = np.array(
        rotulos,
        dtype=np.int32
    )

    print(f"Total de imagens: {len(x)}")
    print(f"Formato das imagens: {x.shape}")
    print(f"Formato dos rótulos: {y.shape}")

    return x, y
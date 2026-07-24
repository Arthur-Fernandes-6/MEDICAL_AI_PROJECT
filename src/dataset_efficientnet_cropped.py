import os
import numpy as np
from preprocess_efficientnet import preprocessar_imagem_efficientnet


def carregar_dataset_efficientnet_cropped(
    caminho_dataset,
    conjunto
):
    x = []
    y = []

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

        if not os.path.isdir(caminho_classe):
            raise FileNotFoundError(
                f"Pasta não encontrada: {caminho_classe}"
            )

        arquivos = os.listdir(caminho_classe)

        print(
            f"Carregando {conjunto}/{nome_classe}: "
            f"{len(arquivos)} arquivos"
        )

        for nome_arquivo in arquivos:
            caminho_completo = os.path.join(
                caminho_classe,
                nome_arquivo
            )

            imagem = preprocessar_imagem_efficientnet(
                caminho_completo
            )

            if imagem is None:
                continue

            x.append(imagem)
            y.append(rotulo)

    x_array = np.array(
        x,
        dtype=np.float32
    )

    y_array = np.array(
        y,
        dtype=np.int32
    )

    print("\nDataset carregado:")
    print("X:", x_array.shape)
    print("y:", y_array.shape)

    return x_array, y_array
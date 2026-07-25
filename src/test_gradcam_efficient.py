import os
import tensorflow as tf
from api.gradcam_explainer import (
    gerar_explicacao_gradcam
)


CAMINHO_MODELO = (
    "models/brain_tumor_efficientnet_v1.keras"
)

CAMINHO_IMAGEM = (
    "externalTest/yes/glioma/image(23).jpg"
)

CAMINHO_SAIDA = (
    "resultados_gradcam/gradcam_teste.jpg"
)


def testar_gradcam():

    print("Carregando modelo...")

    if not os.path.exists(CAMINHO_MODELO):
        raise FileNotFoundError(
            f"Modelo não encontrado: {CAMINHO_MODELO}"
        )

    if not os.path.exists(CAMINHO_IMAGEM):
        raise FileNotFoundError(
            f"Imagem não encontrada: {CAMINHO_IMAGEM}"
        )

    modelo = tf.keras.models.load_model(
        CAMINHO_MODELO
    )

    print("Modelo carregado.")

    print("\nGerando Grad-CAM...")

    resultado = gerar_explicacao_gradcam(
        modelo=modelo,
        caminho_imagem=CAMINHO_IMAGEM,
        caminho_saida=CAMINHO_SAIDA,
        intensidade=0.40,
        corte_minimo=0.15,
        threshold_classificacao=0.5
    )

    print("\nGrad-CAM gerado com sucesso!")

    print(
        f"Classe: {resultado['classe_texto']}"
    )

    print(
        "Probabilidade de tumor: "
        f"{resultado['probabilidade_tumor']:.4f}"
    )

    print(
        "Confiança da previsão: "
        f"{resultado['confianca']:.2%}"
    )

    print(
        "Formato dos mapas: "
        f"{resultado['shape_mapas']}"
    )

    print(
        "Formato dos gradientes: "
        f"{resultado['shape_gradientes']}"
    )

    print(
        "Imagem salva em: "
        f"{resultado['caminho_gradcam']}"
    )


if __name__ == "__main__":
    testar_gradcam()
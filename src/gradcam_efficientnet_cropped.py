import os
import argparse

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.preprocess_brain_crop import preprocessar_brain_crop


# ============================================================
# CONFIGURAÇÕES
# ============================================================

CAMINHO_MODELO_PADRAO = (
    "models/brain_tumor_efficientnet_cropped_best.keras"
)

PASTA_RESULTADOS = "results_gradcam_cropped"

TAMANHO_IMAGEM = (224, 224)


# ============================================================
# LOCALIZAR A EFFICIENTNET DENTRO DO MODELO
# ============================================================

def encontrar_base_model(modelo):
    """
    Procura o modelo EfficientNet aninhado dentro do modelo completo.
    """

    for camada in modelo.layers:
        if isinstance(camada, tf.keras.Model):
            nome = camada.name.lower()

            if "efficientnet" in nome:
                return camada

    raise ValueError(
        "Não foi possível encontrar a EfficientNet dentro do modelo."
    )


# ============================================================
# PREPARAR IMAGEM
# ============================================================

def preparar_imagem(caminho_imagem):
    """
    Aplica o mesmo Brain Crop usado para gerar o dataset cropped.

    Retorna:
    - imagem RGB para exibição;
    - tensor float32 com batch para o modelo.
    """

    imagem_cropped = preprocessar_brain_crop(
        caminho_imagem,
        normalizar=False
    )

    if imagem_cropped is None:
        raise ValueError(
            f"Não foi possível processar a imagem: {caminho_imagem}"
        )

    imagem_cropped = cv2.resize(
        imagem_cropped,
        TAMANHO_IMAGEM
    )

    # O OpenCV usa BGR, mas o modelo foi treinado com RGB.
    imagem_rgb = cv2.cvtColor(
        imagem_cropped,
        cv2.COLOR_BGR2RGB
    )

    imagem_modelo = imagem_rgb.astype(
        np.float32
    )

    imagem_modelo = np.expand_dims(
        imagem_modelo,
        axis=0
    )

    return imagem_rgb, imagem_modelo


# ============================================================
# GERAR GRAD-CAM
# ============================================================

def gerar_gradcam(
    modelo,
    imagem_modelo,
    nome_camada_convolucional="top_conv"
):
    """
    Gera Grad-CAM acessando a última camada convolucional
    da EfficientNet e aplicando manualmente o cabeçalho
    de classificação.

    Essa abordagem evita o erro de grafo desconectado
    em modelos com EfficientNet aninhada.
    """

    base_model = encontrar_base_model(
        modelo
    )

    try:
        camada_convolucional = base_model.get_layer(
            nome_camada_convolucional
        )
    except ValueError as erro:
        raise ValueError(
            f"A camada '{nome_camada_convolucional}' "
            "não foi encontrada na EfficientNet."
        ) from erro

    extrator_caracteristicas = tf.keras.Model(
        inputs=base_model.input,
        outputs=[
            camada_convolucional.output,
            base_model.output
        ]
    )

    indice_base_model = modelo.layers.index(
        base_model
    )

    camadas_cabecalho = modelo.layers[
        indice_base_model + 1:
    ]

    entrada = tf.convert_to_tensor(
        imagem_modelo,
        dtype=tf.float32
    )

    with tf.GradientTape() as fita:
        saida_conv, x = extrator_caracteristicas(
            entrada,
            training=False
        )

        fita.watch(
            saida_conv
        )

        # Aplica manualmente:
        # GlobalAveragePooling2D
        # BatchNormalization
        # Dense
        # Dropout
        # Dense sigmoid
        for camada in camadas_cabecalho:
            x = camada(
                x,
                training=False
            )

        probabilidade_tumor = x[:, 0]

    gradientes = fita.gradient(
        probabilidade_tumor,
        saida_conv
    )

    if gradientes is None:
        raise RuntimeError(
            "Os gradientes retornaram None."
        )

    pesos = tf.reduce_mean(
        gradientes,
        axis=(1, 2)
    )

    mapa = tf.reduce_sum(
        saida_conv * pesos[:, None, None, :],
        axis=-1
    )

    mapa = tf.nn.relu(
        mapa
    )

    mapa = mapa[0]

    valor_maximo = tf.reduce_max(
        mapa
    )

    if valor_maximo > 0:
        mapa = mapa / valor_maximo

    mapa = mapa.numpy()

    probabilidade = float(
        probabilidade_tumor.numpy()[0]
    )

    return mapa, probabilidade


# ============================================================
# CRIAR SOBREPOSIÇÃO
# ============================================================

def criar_sobreposicao(
    imagem_rgb,
    mapa_calor,
    intensidade=0.40
):
    mapa_redimensionado = cv2.resize(
        mapa_calor,
        (
            imagem_rgb.shape[1],
            imagem_rgb.shape[0]
        )
    )

    mapa_uint8 = np.uint8(
        255 * mapa_redimensionado
    )

    mapa_colorido_bgr = cv2.applyColorMap(
        mapa_uint8,
        cv2.COLORMAP_JET
    )

    mapa_colorido_rgb = cv2.cvtColor(
        mapa_colorido_bgr,
        cv2.COLOR_BGR2RGB
    )

    sobreposicao = cv2.addWeighted(
        imagem_rgb,
        1.0 - intensidade,
        mapa_colorido_rgb,
        intensidade,
        0
    )

    return mapa_colorido_rgb, sobreposicao


# ============================================================
# SALVAR RESULTADO
# ============================================================

def salvar_resultado(
    imagem_original,
    mapa_colorido,
    sobreposicao,
    probabilidade,
    caminho_imagem
):
    os.makedirs(
        PASTA_RESULTADOS,
        exist_ok=True
    )

    nome_arquivo = os.path.splitext(
        os.path.basename(caminho_imagem)
    )[0]

    classe_predita = (
        "Tumor"
        if probabilidade >= 0.5
        else "Sem tumor"
    )

    figura = plt.figure(
        figsize=(15, 5)
    )

    eixo1 = figura.add_subplot(
        1,
        3,
        1
    )

    eixo1.imshow(
        imagem_original
    )

    eixo1.set_title(
        "Imagem após Brain Crop"
    )

    eixo1.axis(
        "off"
    )

    eixo2 = figura.add_subplot(
        1,
        3,
        2
    )

    eixo2.imshow(
        mapa_colorido
    )

    eixo2.set_title(
        "Mapa Grad-CAM"
    )

    eixo2.axis(
        "off"
    )

    eixo3 = figura.add_subplot(
        1,
        3,
        3
    )

    eixo3.imshow(
        sobreposicao
    )

    eixo3.set_title(
        f"Grad-CAM sobreposto\n"
        f"Predição: {classe_predita} | "
        f"P(tumor): {probabilidade:.4f}"
    )

    eixo3.axis(
        "off"
    )

    plt.tight_layout()

    caminho_resultado = os.path.join(
        PASTA_RESULTADOS,
        f"{nome_arquivo}_gradcam_cropped.png"
    )

    plt.savefig(
        caminho_resultado,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        figura
    )

    print(
        f"\nResultado salvo em: {caminho_resultado}"
    )

    return caminho_resultado


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Gerar Grad-CAM para a EfficientNet "
            "treinada com Brain Crop."
        )
    )

    parser.add_argument(
        "--imagem",
        required=True,
        help="Caminho da imagem de ressonância."
    )

    parser.add_argument(
        "--modelo",
        default=CAMINHO_MODELO_PADRAO,
        help="Caminho do modelo .keras."
    )

    argumentos = parser.parse_args()

    caminho_imagem = argumentos.imagem
    caminho_modelo = argumentos.modelo

    if not os.path.isfile(
        caminho_imagem
    ):
        raise FileNotFoundError(
            f"Imagem não encontrada: {caminho_imagem}"
        )

    if not os.path.isfile(
        caminho_modelo
    ):
        raise FileNotFoundError(
            f"Modelo não encontrado: {caminho_modelo}"
        )

    print("\n" + "=" * 60)
    print("CARREGANDO MODELO CROPPED")
    print("=" * 60)

    modelo = tf.keras.models.load_model(
        caminho_modelo
    )

    print(
        "Modelo carregado com sucesso!"
    )

    print("\n" + "=" * 60)
    print("PROCESSANDO IMAGEM")
    print("=" * 60)

    imagem_rgb, imagem_modelo = preparar_imagem(
        caminho_imagem
    )

    print(
        "Formato enviado ao modelo:",
        imagem_modelo.shape
    )

    print(
        "Valor mínimo:",
        imagem_modelo.min()
    )

    print(
        "Valor máximo:",
        imagem_modelo.max()
    )

    print("\n" + "=" * 60)
    print("GERANDO GRAD-CAM")
    print("=" * 60)

    mapa_calor, probabilidade = gerar_gradcam(
        modelo,
        imagem_modelo
    )

    classe_predita = (
        "Tumor"
        if probabilidade >= 0.5
        else "Sem tumor"
    )

    print(
        f"Probabilidade de tumor: "
        f"{probabilidade:.6f}"
    )

    print(
        f"Classe predita: "
        f"{classe_predita}"
    )

    mapa_colorido, sobreposicao = criar_sobreposicao(
        imagem_rgb,
        mapa_calor
    )

    salvar_resultado(
        imagem_rgb,
        mapa_colorido,
        sobreposicao,
        probabilidade,
        caminho_imagem
    )

    print("\n" + "=" * 60)
    print("GRAD-CAM FINALIZADO")
    print("=" * 60)


if __name__ == "__main__":
    main()
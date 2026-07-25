import base64

import cv2
import numpy as np
import tensorflow as tf


def encontrar_efficientnet(
    modelo: tf.keras.Model
) -> tf.keras.Model:
    """
    Localiza a EfficientNet aninhada dentro do modelo completo.
    """

    for camada in modelo.layers:
        if isinstance(camada, tf.keras.Model):
            if "efficientnet" in camada.name.lower():
                return camada

    raise ValueError(
        "A EfficientNet não foi encontrada dentro do modelo."
    )


def gerar_mapa_gradcam(
    modelo: tf.keras.Model,
    imagem_modelo: np.ndarray,
    nome_camada: str = "top_conv"
) -> tuple[np.ndarray, float]:
    """
    Gera o mapa Grad-CAM para uma imagem já processada.

    imagem_modelo:
        shape (1, 224, 224, 3)
        float32
        escala 0-255
    """

    base_model = encontrar_efficientnet(
        modelo
    )

    camada_convolucional = base_model.get_layer(
        nome_camada
    )

    extrator = tf.keras.Model(
        inputs=base_model.input,
        outputs=[
            camada_convolucional.output,
            base_model.output
        ]
    )

    indice_base = modelo.layers.index(
        base_model
    )

    camadas_cabecalho = modelo.layers[
        indice_base + 1:
    ]

    entrada = tf.convert_to_tensor(
        imagem_modelo,
        dtype=tf.float32
    )

    with tf.GradientTape() as fita:
        saida_conv, x = extrator(
            entrada,
            training=False
        )

        fita.watch(
            saida_conv
        )

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
            "Não foi possível calcular os gradientes."
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

    if float(valor_maximo) > 0:
        mapa = mapa / valor_maximo

    probabilidade = float(
        probabilidade_tumor.numpy()[0]
    )

    return mapa.numpy(), probabilidade


def criar_sobreposicao_gradcam(
    imagem_rgb: np.ndarray,
    mapa_gradcam: np.ndarray,
    intensidade: float = 0.40
) -> np.ndarray:
    """
    Sobrepõe o mapa Grad-CAM na imagem RGB.
    """

    mapa_redimensionado = cv2.resize(
        mapa_gradcam,
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

    return sobreposicao


def converter_imagem_para_base64(
    imagem_rgb: np.ndarray
) -> str:
    """
    Converte uma imagem RGB para PNG em Base64.
    """

    imagem_bgr = cv2.cvtColor(
        imagem_rgb,
        cv2.COLOR_RGB2BGR
    )

    sucesso, buffer = cv2.imencode(
        ".png",
        imagem_bgr
    )

    if not sucesso:
        raise RuntimeError(
            "Não foi possível converter o Grad-CAM para PNG."
        )

    return base64.b64encode(
        buffer.tobytes()
    ).decode("utf-8")


def gerar_explicacao_gradcam(
    modelo: tf.keras.Model,
    imagem_rgb: np.ndarray,
    imagem_modelo: np.ndarray
) -> dict:
    """
    Gera a sobreposição Grad-CAM e devolve a imagem em Base64.
    """

    mapa, probabilidade_gradcam = gerar_mapa_gradcam(
        modelo=modelo,
        imagem_modelo=imagem_modelo
    )

    sobreposicao = criar_sobreposicao_gradcam(
        imagem_rgb=imagem_rgb,
        mapa_gradcam=mapa,
        intensidade=0.40
    )

    imagem_base64 = converter_imagem_para_base64(
        sobreposicao
    )

    return {
        "imagem_base64": imagem_base64,
        "probabilidade_tumor": probabilidade_gradcam
    }
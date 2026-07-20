import base64
import io
import matplotlib.pyplot as plt
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries


explainer = lime_image.LimeImageExplainer()


def gerar_explicacao_lime(modelo, imagem_original):
    """
    Gera uma explicação visual com LIME.

    Parâmetros
    ----------
    modelo:
        Modelo Keras já carregado.

    imagem_original:
        Imagem no formato NumPy, com dimensões
        (altura, largura, canais), contendo valores de 0 a 255.

    Retorno
    -------
    str:
        Imagem explicativa codificada em Base64.
    """

    imagem_original = np.array(imagem_original)

    if imagem_original.dtype != np.uint8:
        imagem_original = np.clip(imagem_original, 0, 255).astype(np.uint8)

    def funcao_predicao(imagens):
        imagens = np.array(imagens).astype("float32") / 255.0

        previsoes = modelo.predict(
            imagens,
            verbose=0
        )

        previsoes = np.array(previsoes)

        # Caso o modelo binário retorne apenas uma probabilidade:
        # [[0.92], [0.15], ...]
        if previsoes.ndim == 2 and previsoes.shape[1] == 1:
            probabilidade_tumor = previsoes[:, 0]
            probabilidade_sem_tumor = 1 - probabilidade_tumor

            previsoes = np.column_stack(
                (
                    probabilidade_sem_tumor,
                    probabilidade_tumor
                )
            )

        return previsoes

    explicacao = explainer.explain_instance(
        imagem_original.astype("double"),
        funcao_predicao,
        top_labels=1,
        hide_color=0,
        num_samples=200
    )

    rotulo_explicado = explicacao.top_labels[0]

    imagem_lime, mascara = explicacao.get_image_and_mask(
        rotulo_explicado,
        positive_only=True,
        num_features=8,
        hide_rest=False
    )

    imagem_normalizada = imagem_lime.astype("float32")

    if imagem_normalizada.max() > 1.0:
        imagem_normalizada = imagem_normalizada / 255.0

    imagem_com_contornos = mark_boundaries(
    imagem_normalizada,
    mascara,
    color=(1, 1, 0),
    mode="thick"
)

    buffer = io.BytesIO()

    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    ax.imshow(imagem_com_contornos)
    ax.set_title(
        "",
        fontsize=13,
        pad=12
    )
    ax.axis("off")

    fig.tight_layout()

    fig.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        pad_inches=0.15,
        dpi=105
    )

    plt.close(fig)
    buffer.seek(0)

    imagem_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return imagem_base64
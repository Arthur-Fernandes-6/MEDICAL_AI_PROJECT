import cv2
import numpy as np


TAMANHO_FINAL = (224, 224)


def carregar_imagem(caminho):
    imagem = cv2.imread(caminho)

    if imagem is None:
        raise ValueError(
            f"Não foi possível abrir a imagem: {caminho}"
        )

    return imagem


def encontrar_regiao_principal(imagem):
    """
    Encontra aproximadamente a região ocupada pela cabeça.

    Retorna:
        recorte da região principal;
        máscara correspondente.
    """
    cinza = cv2.cvtColor(
        imagem,
        cv2.COLOR_BGR2GRAY
    )

    cinza = cv2.GaussianBlur(
        cinza,
        (5, 5),
        0
    )

    # Separa automaticamente o fundo escuro
    # da região anatômica.
    _, mascara = cv2.threshold(
        cinza,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones(
        (7, 7),
        dtype=np.uint8
    )

    mascara = cv2.morphologyEx(
        mascara,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contornos:
        return imagem, np.ones(
            imagem.shape[:2],
            dtype=np.uint8
        ) * 255

    maior_contorno = max(
        contornos,
        key=cv2.contourArea
    )

    x, y, largura, altura = cv2.boundingRect(
        maior_contorno
    )

    # Pequena margem ao redor da região detectada
    margem_x = int(largura * 0.03)
    margem_y = int(altura * 0.03)

    x1 = max(0, x - margem_x)
    y1 = max(0, y - margem_y)

    x2 = min(
        imagem.shape[1],
        x + largura + margem_x
    )

    y2 = min(
        imagem.shape[0],
        y + altura + margem_y
    )

    recorte = imagem[y1:y2, x1:x2]
    mascara_recortada = mascara[y1:y2, x1:x2]

    return recorte, mascara_recortada


def criar_mascara_eliptica(imagem):
    """
    Cria uma máscara elíptica central para reduzir
    a influência de regiões periféricas.
    """
    altura, largura = imagem.shape[:2]

    mascara = np.zeros(
        (altura, largura),
        dtype=np.uint8
    )

    centro = (
        largura // 2,
        altura // 2
    )

    # A elipse é um pouco menor que o recorte total
    # para reduzir bordas, crânio e fundo.
    eixo_horizontal = int(largura * 0.44)
    eixo_vertical = int(altura * 0.46)

    cv2.ellipse(
        mascara,
        center=centro,
        axes=(
            eixo_horizontal,
            eixo_vertical
        ),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=-1
    )

    # Suaviza a transição da máscara.
    mascara = cv2.GaussianBlur(
        mascara,
        (21, 21),
        0
    )

    return mascara


def aplicar_mascara(imagem, mascara):
    mascara_float = (
        mascara.astype(np.float32) / 255.0
    )

    mascara_float = np.expand_dims(
        mascara_float,
        axis=-1
    )

    imagem_mascarada = (
        imagem.astype(np.float32)
        * mascara_float
    )

    return imagem_mascarada.astype(
        np.uint8
    )


def preprocessar_brain_crop(
    caminho,
    normalizar=True
):
    """
    Pipeline:

    1. abre a imagem;
    2. encontra a região principal;
    3. aplica uma máscara elíptica;
    4. redimensiona para 224x224;
    5. normaliza para [0, 1].
    """
    imagem = carregar_imagem(caminho)

    recorte, _ = encontrar_regiao_principal(
        imagem
    )

    if recorte.size == 0:
        raise RuntimeError(
            "O recorte produzido está vazio."
        )

    mascara_eliptica = criar_mascara_eliptica(
        recorte
    )

    imagem_mascarada = aplicar_mascara(
        recorte,
        mascara_eliptica
    )

    imagem_redimensionada = cv2.resize(
        imagem_mascarada,
        TAMANHO_FINAL,
        interpolation=cv2.INTER_AREA
    )

    if normalizar:
        imagem_redimensionada = (
            imagem_redimensionada.astype(
                np.float32
            )
            / 255.0
        )

    return imagem_redimensionada
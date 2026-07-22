import cv2
import numpy as np


TAMANHO_IMAGEM = 224


def redimensionar_com_padding(imagem, tamanho=224):
    """
    Redimensiona a imagem preservando a proporção.

    O espaço restante é preenchido com pixels pretos,
    evitando deformar a anatomia da ressonância.
    """

    altura, largura = imagem.shape[:2] # Obtém a altura e largura da imagem

    escala = min(
        tamanho / largura,
        tamanho / altura
    )

    nova_largura = int(largura * escala)
    nova_altura = int(altura * escala)

    imagem_redimensionada = cv2.resize(
        imagem,
        (nova_largura, nova_altura),
        interpolation=cv2.INTER_AREA
    )

    imagem_final = np.zeros(# Cria uma imagem preta do tamanho desejado
        (tamanho, tamanho, 3),
        dtype=np.uint8
    )

    inicio_x = (tamanho - nova_largura) // 2
    inicio_y = (tamanho - nova_altura) // 2

    imagem_final[
        inicio_y:inicio_y + nova_altura,
        inicio_x:inicio_x + nova_largura
    ] = imagem_redimensionada

    return imagem_final


def preprocessar_imagem_efficientnet(caminho_imagem):
    """
    Carrega uma imagem para utilização com EfficientNetB0.

    A EfficientNet espera:
    - imagem RGB;
    - três canais;
    - tamanho 224x224;
    - pixels entre 0 e 255.
    """

    imagem = cv2.imread(caminho_imagem)

    if imagem is None:
        print(f"Não foi possível carregar: {caminho_imagem}")
        return None

    # OpenCV carrega em BGR, portanto convertemos para RGB
    imagem = cv2.cvtColor(
        imagem,
        cv2.COLOR_BGR2RGB
    )

    imagem = redimensionar_com_padding(
        imagem,
        TAMANHO_IMAGEM
    )

    
    imagem = imagem.astype(np.float32)

    return imagem
import os
import cv2
import numpy as np
from preprocess_brain_crop import (
    preprocessar_brain_crop
)


CAMINHO_IMAGEM = (
    "externalTest/yes/pituitary/image(22).jpg"
)

CAMINHO_SAIDA = (
    "resultados_brain_crop/"
    "brain_crop_teste.jpg"
)


imagem_processada = preprocessar_brain_crop(
    CAMINHO_IMAGEM,
    normalizar=True
)

# Volta de [0,1] para [0,255]
# apenas para salvar e visualizar.
imagem_salvar = np.uint8(
    np.clip(imagem_processada, 0, 1) * 255
)

os.makedirs(
    os.path.dirname(CAMINHO_SAIDA),
    exist_ok=True
)

salvou = cv2.imwrite(
    CAMINHO_SAIDA,
    imagem_salvar
)

if not salvou:
    raise RuntimeError(
        "Não foi possível salvar a imagem."
    )

print("Imagem processada com sucesso!")
print(f"Formato: {imagem_processada.shape}")
print(f"Mínimo: {imagem_processada.min():.4f}")
print(f"Máximo: {imagem_processada.max():.4f}")
print(f"Imagem salva em: {CAMINHO_SAIDA}")
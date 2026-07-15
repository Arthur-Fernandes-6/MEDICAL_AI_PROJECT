import numpy as np
from tensorflow.keras.models import load_model
from preprocess import preprocessar_imagem


modelo = load_model("models/brain_tumor_cnn.keras")


def predict(caminho):
    imagem = preprocessar_imagem(caminho)

    if imagem is None:
        return None
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import preprocessar_imagem


modelo = load_model("models/brain_tumor_cnn.keras")


def predict(caminho):
    imagem = preprocessar_imagem(caminho)

    if imagem is None:
        return None

    imagem = np.expand_dims(imagem, axis=0)# cria a nova dimensão no início para adicionar uma imagem.

    resultado = modelo.predict(imagem)

    probabilidade = float(resultado[0][0])

    if probabilidade >= 0.5:
        classificacao = "Tumor"
    
    else:
        classificacao =  "Sem tumor"

    return classificacao, probabilidade


classificacao, probabilidade = predict("datasets/yes/Y1.jpg")
print("Resultado: ", classificacao)
print(f"Probabilidade de tumor: {probabilidade * 100:.2f}%")
import numpy as np
from tensorflow.keras.models import load_model
from dataset_v2 import carregar_dataset_v2
from preprocess import preprocessar_imagem
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# testando o modelo treinado em um conjunto de teste separado, que não foi usado durante o treinamento ou validação, para avaliar a capacidade do modelo de generalizar para novos dados
def avaliar_modelo():
    print("Carregando o modelo...")

    modelo = load_model("models/brain_tumor_cnn_v2.keras")

    print("Carregando o conjunto de teste...")

    x_test, y_test = carregar_dataset_v2( # x e y recebem as imagens e rótulos do conjunto de teste
        "brain_tumor_mri_dataset",
        "Testing"
    )

    print("\nFormato de x_test:", x_test.shape)
    print("Formato de y_test:", y_test.shape)

    print("\nRealizando previsões...")

    probabilidades = modelo.predict(x_test) # gera as probabilidades de cada imagem do conjunto de teste pertencer à classe "Tumor" (1) ou "Sem tumor" (0)

    previsoes = (probabilidades >= 0.5).astype(int).flatten() # converte as probabilidades em rótulos binários (0 ou 1) com base em um limiar de 0,5 e achata o array para uma dimensão

    y_test = y_test.astype(int)

    acuracia = accuracy_score(y_test, previsoes)

    matriz = confusion_matrix(y_test, previsoes)

    relatorio = classification_report( # gera um relatório detalhado de métricas de classificação, incluindo precisão, recall e F1-score para cada classe
        y_test,
        previsoes,
        target_names=["Sem tumor", "Tumor"]
    )

    print("\n=== ACURÁCIA FINAL ===")
    print(f"Acurácia no conjunto Testing: {acuracia:.4f}")
    print(f"Acurácia em porcentagem: {acuracia * 100:.2f}%")

    print("\n=== MATRIZ DE CONFUSÃO ===")
    print(matriz)

    print("\n=== RELATÓRIO DE CLASSIFICAÇÃO ===")
    print(relatorio)


if __name__ == "__main__":
    avaliar_modelo()
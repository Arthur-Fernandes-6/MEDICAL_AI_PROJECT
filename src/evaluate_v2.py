import numpy as np
from tensorflow.keras.models import load_model
from dataset_v2 import carregar_dataset_v2
from preprocess import preprocessar_imagem
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


def avaliar_modelo():
    print("Carregando o modelo...")

    modelo = load_model("models/brain_tumor_cnn_v2.keras")

    print("Carregando o conjunto de teste...")

    x_test, y_test = carregar_dataset_v2(
        "brain_tumor_mri_dataset",
        "Testing"
    )

    print("\nFormato de x_test:", x_test.shape)
    print("Formato de y_test:", y_test.shape)

    print("\nRealizando previsões...")

    probabilidades = modelo.predict(x_test)

    previsoes = (probabilidades >= 0.5).astype(int).flatten()

    y_test = y_test.astype(int)

    acuracia = accuracy_score(y_test, previsoes)

    matriz = confusion_matrix(y_test, previsoes)

    relatorio = classification_report(
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
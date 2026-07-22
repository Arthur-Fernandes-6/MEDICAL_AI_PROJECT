import os
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
from preditcv1 import prever_imagem


def validar_externamente(caminho_dataset):
    y_real = []
    y_pred = []

    pasta_yes = os.path.join(caminho_dataset, "yes")
    pasta_no = os.path.join(caminho_dataset, "no")

    # Processa as imagens com tumor
    for nome_arquivo in os.listdir(pasta_yes):
        caminho_completo = os.path.join(pasta_yes, nome_arquivo)

        resultado = prever_imagem(caminho_completo)

        if resultado is None:
            continue

        classificacao, probabilidade = resultado

        y_real.append(1)

        if classificacao == "Tumor":
            y_pred.append(1)
        else:
            y_pred.append(0)

    # Processa as imagens sem tumor
    for nome_arquivo in os.listdir(pasta_no):
        caminho_completo = os.path.join(pasta_no, nome_arquivo)

        resultado = prever_imagem(caminho_completo)

        if resultado is None:
            continue

        classificacao, probabilidade = resultado

        y_real.append(0)

        if classificacao == "Tumor":
            y_pred.append(1)
        else:
            y_pred.append(0)

    return y_real, y_pred


y_real, y_pred = validar_externamente("externalTest")

accuracy = accuracy_score(y_real, y_pred)

print(f"Acurácia externa: {accuracy * 100:.2f}%")

matriz = confusion_matrix(y_real, y_pred)

print("Matriz de confusão externa:")
print(matriz)

visualizacao = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=["Sem tumor", "Tumor"]
)

visualizacao.plot()
plt.title("Matriz de Confusão — Validação Externa")
plt.show()

print(
    classification_report(
        y_real,
        y_pred,
        target_names=["Sem tumor", "Tumor"]
    )
)

if __name__ == "__main__":
    classificacao, probabilidade = prever_imagem("externalTest\yes\Te-aug-me_1.jpg")

    print(classificacao)
    print(probabilidade)
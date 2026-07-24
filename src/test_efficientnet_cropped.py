import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from dataset_efficientnet_cropped import (
    carregar_dataset_efficientnet_cropped
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

CAMINHO_DATASET = "brain_tumor_mri_dataset_cropped"

CAMINHO_MODELO = (
    "models/brain_tumor_efficientnet_cropped_best.keras"
)

PASTA_RESULTADOS = "results_efficientnet_cropped"

BATCH_SIZE = 32

LIMIAR = 0.5


# ============================================================
# CRIAR PASTA DOS RESULTADOS
# ============================================================

os.makedirs(
    PASTA_RESULTADOS,
    exist_ok=True
)


# ============================================================
# CARREGAR MODELO
# ============================================================

print("\n" + "=" * 60)
print("CARREGANDO MODELO EFFICIENTNET CROPPED")
print("=" * 60)

if not os.path.exists(CAMINHO_MODELO):
    raise FileNotFoundError(
        f"Modelo não encontrado: {CAMINHO_MODELO}"
    )

modelo = tf.keras.models.load_model(
    CAMINHO_MODELO
)

print("Modelo carregado com sucesso!")


# ============================================================
# CARREGAR CONJUNTO TESTING
# ============================================================

print("\n" + "=" * 60)
print("CARREGANDO CONJUNTO TESTING")
print("=" * 60)

x_test, y_test = carregar_dataset_efficientnet_cropped(
    CAMINHO_DATASET,
    "Testing"
)

print("\nConjunto de teste carregado!")

print("Formato de X:", x_test.shape)
print("Formato de y:", y_test.shape)
print("Tipo de X:", x_test.dtype)
print("Tipo de y:", y_test.dtype)
print("Valor mínimo de X:", x_test.min())
print("Valor máximo de X:", x_test.max())


# ============================================================
# GERAR PROBABILIDADES
# ============================================================

print("\n" + "=" * 60)
print("REALIZANDO PREDIÇÕES")
print("=" * 60)

probabilidades = modelo.predict(
    x_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

probabilidades = probabilidades.reshape(-1)

predicoes = (
    probabilidades >= LIMIAR
).astype(np.int32)

print("\nPredições finalizadas!")

print("Quantidade de probabilidades:", len(probabilidades))
print("Quantidade de predições:", len(predicoes))


# ============================================================
# CALCULAR MÉTRICAS
# ============================================================

acuracia = accuracy_score(
    y_test,
    predicoes
)

precisao = precision_score(
    y_test,
    predicoes,
    zero_division=0
)

recall = recall_score(
    y_test,
    predicoes,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predicoes,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    probabilidades
)

matriz = confusion_matrix(
    y_test,
    predicoes
)

verdadeiros_negativos = int(matriz[0, 0])
falsos_positivos = int(matriz[0, 1])
falsos_negativos = int(matriz[1, 0])
verdadeiros_positivos = int(matriz[1, 1])

especificidade = (
    verdadeiros_negativos
    / (
        verdadeiros_negativos
        + falsos_positivos
    )
)

print("\n" + "=" * 60)
print("RESULTADOS NO CONJUNTO TESTING")
print("=" * 60)

print(f"Accuracy:      {acuracia:.4f}")
print(f"Precision:     {precisao:.4f}")
print(f"Recall:        {recall:.4f}")
print(f"Specificity:   {especificidade:.4f}")
print(f"F1-score:      {f1:.4f}")
print(f"AUC:           {auc:.4f}")

print("\nResultados em porcentagem:")

print(f"Accuracy:      {acuracia * 100:.2f}%")
print(f"Precision:     {precisao * 100:.2f}%")
print(f"Recall:        {recall * 100:.2f}%")
print(f"Specificity:   {especificidade * 100:.2f}%")
print(f"F1-score:      {f1 * 100:.2f}%")
print(f"AUC:           {auc * 100:.2f}%")

print("\nMatriz de confusão:")

print(matriz)

print("\nDetalhamento:")

print(f"Verdadeiros negativos: {verdadeiros_negativos}")
print(f"Falsos positivos:      {falsos_positivos}")
print(f"Falsos negativos:      {falsos_negativos}")
print(f"Verdadeiros positivos: {verdadeiros_positivos}")


# ============================================================
# RELATÓRIO DE CLASSIFICAÇÃO
# ============================================================

relatorio_texto = classification_report(
    y_test,
    predicoes,
    target_names=[
        "Sem tumor",
        "Tumor"
    ],
    digits=4,
    zero_division=0
)

print("\n" + "=" * 60)
print("RELATÓRIO DE CLASSIFICAÇÃO")
print("=" * 60)

print(relatorio_texto)


# ============================================================
# SALVAR MÉTRICAS EM JSON
# ============================================================

metricas = {
    "modelo": CAMINHO_MODELO,
    "dataset": CAMINHO_DATASET,
    "conjunto": "Testing",
    "quantidade_imagens": int(len(y_test)),
    "limiar": float(LIMIAR),

    "accuracy": float(acuracia),
    "precision": float(precisao),
    "recall": float(recall),
    "specificity": float(especificidade),
    "f1_score": float(f1),
    "auc": float(auc),

    "verdadeiros_negativos": verdadeiros_negativos,
    "falsos_positivos": falsos_positivos,
    "falsos_negativos": falsos_negativos,
    "verdadeiros_positivos": verdadeiros_positivos
}

caminho_json = os.path.join(
    PASTA_RESULTADOS,
    "metricas_testing_cropped.json"
)

with open(
    caminho_json,
    "w",
    encoding="utf-8"
) as arquivo_json:
    json.dump(
        metricas,
        arquivo_json,
        indent=4,
        ensure_ascii=False
    )

print(
    "\nMétricas salvas em:",
    caminho_json
)


# ============================================================
# SALVAR RELATÓRIO EM TXT
# ============================================================

caminho_relatorio = os.path.join(
    PASTA_RESULTADOS,
    "classification_report_testing_cropped.txt"
)

with open(
    caminho_relatorio,
    "w",
    encoding="utf-8"
) as arquivo_relatorio:

    arquivo_relatorio.write(
        "AVALIAÇÃO EFFICIENTNET CROPPED\n"
    )

    arquivo_relatorio.write(
        "=" * 60 + "\n\n"
    )

    arquivo_relatorio.write(
        f"Modelo: {CAMINHO_MODELO}\n"
    )

    arquivo_relatorio.write(
        f"Dataset: {CAMINHO_DATASET}\n"
    )

    arquivo_relatorio.write(
        f"Quantidade de imagens: {len(y_test)}\n"
    )

    arquivo_relatorio.write(
        f"Limiar: {LIMIAR}\n\n"
    )

    arquivo_relatorio.write(
        f"Accuracy: {acuracia:.4f}\n"
    )

    arquivo_relatorio.write(
        f"Precision: {precisao:.4f}\n"
    )

    arquivo_relatorio.write(
        f"Recall: {recall:.4f}\n"
    )

    arquivo_relatorio.write(
        f"Specificity: {especificidade:.4f}\n"
    )

    arquivo_relatorio.write(
        f"F1-score: {f1:.4f}\n"
    )

    arquivo_relatorio.write(
        f"AUC: {auc:.4f}\n\n"
    )

    arquivo_relatorio.write(
        "Matriz de confusão:\n"
    )

    arquivo_relatorio.write(
        str(matriz)
    )

    arquivo_relatorio.write(
        "\n\nRelatório de classificação:\n\n"
    )

    arquivo_relatorio.write(
        relatorio_texto
    )

print(
    "Relatório salvo em:",
    caminho_relatorio
)


# ============================================================
# SALVAR MATRIZ DE CONFUSÃO
# ============================================================

display = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=[
        "Sem tumor",
        "Tumor"
    ]
)

display.plot(
    values_format="d"
)

plt.title(
    "Matriz de Confusão - EfficientNet Cropped"
)

plt.tight_layout()

caminho_matriz = os.path.join(
    PASTA_RESULTADOS,
    "matriz_confusao_testing_cropped.png"
)

plt.savefig(
    caminho_matriz,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Matriz de confusão salva em:",
    caminho_matriz
)


# ============================================================
# SALVAR PREDIÇÕES
# ============================================================

caminho_predicoes = os.path.join(
    PASTA_RESULTADOS,
    "predicoes_testing_cropped.csv"
)

dados_predicoes = np.column_stack(
    (
        y_test,
        probabilidades,
        predicoes
    )
)

np.savetxt(
    caminho_predicoes,
    dados_predicoes,
    delimiter=",",
    header=(
        "rotulo_real,"
        "probabilidade_tumor,"
        "rotulo_predito"
    ),
    comments="",
    fmt=[
        "%d",
        "%.8f",
        "%d"
    ]
)

print(
    "Predições salvas em:",
    caminho_predicoes
)


print("\n" + "=" * 60)
print("AVALIAÇÃO FINALIZADA")
print("=" * 60)
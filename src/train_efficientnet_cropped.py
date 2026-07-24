import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from dataset_efficientnet_cropped import (
    carregar_dataset_efficientnet_cropped
)
from model_efficientnet import criar_modelo_efficientnet


# ============================================================
# CONFIGURAÇÕES
# ============================================================

CAMINHO_DATASET = "brain_tumor_mri_dataset_cropped"

CAMINHO_MODELO_FINAL = (
    "models/brain_tumor_efficientnet_cropped.keras"
)

CAMINHO_MELHOR_MODELO = (
    "models/brain_tumor_efficientnet_cropped_best.keras"
)

SEMENTE = 42
TAMANHO_VALIDACAO = 0.20
EPOCAS = 30
BATCH_SIZE = 32


# ============================================================
# REPRODUTIBILIDADE
# ============================================================

np.random.seed(SEMENTE)
tf.random.set_seed(SEMENTE)


# ============================================================
# CRIAR PASTA DOS MODELOS
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)


# ============================================================
# CARREGAR DATASET RECORTADO
# ============================================================

print("\n" + "=" * 60)
print("CARREGANDO DATASET COM BRAIN CROP")
print("=" * 60)

x, y = carregar_dataset_efficientnet_cropped(
    CAMINHO_DATASET,
    "Training"
)

print("\nDataset carregado com sucesso!")

print("Formato de X:", x.shape)
print("Formato de y:", y.shape)
print("Tipo de X:", x.dtype)
print("Tipo de y:", y.dtype)
print("Valor mínimo de X:", x.min())
print("Valor máximo de X:", x.max())

print("\nDistribuição das classes:")

valores, quantidades = np.unique(
    y,
    return_counts=True
)

for valor, quantidade in zip(
    valores,
    quantidades
):
    nome_classe = (
        "Sem tumor"
        if valor == 0
        else "Tumor"
    )

    print(
        f"{nome_classe}: "
        f"{quantidade} imagens"
    )


# ============================================================
# DIVISÃO TREINO E VALIDAÇÃO
# ============================================================

x_train, x_val, y_train, y_val = train_test_split(
    x,
    y,
    test_size=TAMANHO_VALIDACAO,
    random_state=SEMENTE,
    stratify=y
)

print("\n" + "=" * 60)
print("DIVISÃO DO DATASET")
print("=" * 60)

print("Treinamento:")
print("X:", x_train.shape)
print("y:", y_train.shape)

print("\nValidação:")
print("X:", x_val.shape)
print("y:", y_val.shape)


# ============================================================
# PESOS DAS CLASSES
# ============================================================

classes = np.unique(y_train)

pesos = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

pesos_classes = {
    int(classe): float(peso)
    for classe, peso in zip(
        classes,
        pesos
    )
}

print("\nPesos das classes:")

for classe, peso in pesos_classes.items():
    nome_classe = (
        "Sem tumor"
        if classe == 0
        else "Tumor"
    )

    print(
        f"{nome_classe}: {peso:.4f}"
    )


# ============================================================
# CRIAR MODELO
# ============================================================

print("\n" + "=" * 60)
print("CRIANDO EFFICIENTNETB0")
print("=" * 60)

modelo, base_model = criar_modelo_efficientnet()
modelo.summary()


# ============================================================
# CALLBACKS
# ============================================================

parada_antecipada = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduzir_taxa_aprendizado = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

salvar_melhor_modelo = ModelCheckpoint(
    filepath=CAMINHO_MELHOR_MODELO,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)


# ============================================================
# TREINAMENTO
# ============================================================

print("\n" + "=" * 60)
print("INICIANDO TREINAMENTO")
print("=" * 60)

historico = modelo.fit(
    x_train,
    y_train,

    validation_data=(
        x_val,
        y_val
    ),

    epochs=EPOCAS,
    batch_size=BATCH_SIZE,

    class_weight=pesos_classes,

    callbacks=[
        parada_antecipada,
        reduzir_taxa_aprendizado,
        salvar_melhor_modelo
    ],

    shuffle=True,
    verbose=1
)


# ============================================================
# AVALIAÇÃO NA VALIDAÇÃO
# ============================================================

print("\n" + "=" * 60)
print("AVALIAÇÃO NA VALIDAÇÃO")
print("=" * 60)

loss, accuracy, auc, precision, recall = modelo.evaluate(
    x_val,
    y_val,
    batch_size=BATCH_SIZE,
    verbose=1
)

print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")


print(
    f"Perda na validação: "
    f"{perda_validacao:.4f}"
)

print(
    f"Acurácia na validação: "
    f"{acuracia_validacao * 100:.2f}%"
)


# ============================================================
# SALVAR MODELO FINAL
# ============================================================

modelo.save(
    CAMINHO_MODELO_FINAL
)

print("\n" + "=" * 60)
print("TREINAMENTO FINALIZADO")
print("=" * 60)

print(
    "Modelo final salvo em:\n"
    f"{CAMINHO_MODELO_FINAL}"
)

print(
    "\nMelhor checkpoint salvo em:\n"
    f"{CAMINHO_MELHOR_MODELO}"
)
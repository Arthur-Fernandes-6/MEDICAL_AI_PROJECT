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

from dataset_efficientnet import (
    carregar_dataset_efficientnet
)

from model_efficientnet import (
    criar_modelo_efficientnet
)


CAMINHO_DATASET = "brain_tumor_mri_dataset"
PASTA_TREINO = "Training"

CAMINHO_MODELO = (
    "models/brain_tumor_efficientnet_v1.keras"
)


def treinar_modelo():
    os.makedirs(
        "models",
        exist_ok=True
    )

    print("=" * 60)
    print("TREINAMENTO COM EFFICIENTNETB0")
    print("=" * 60)

    x, y = carregar_dataset_efficientnet(
        CAMINHO_DATASET,
        PASTA_TREINO
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nFormato dos dados:")
    print("x_train:", x_train.shape)
    print("x_val:", x_val.shape)
    print("y_train:", y_train.shape)
    print("y_val:", y_val.shape)

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
    print(pesos_classes)

    modelo, base_model = criar_modelo_efficientnet()

    modelo.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    reduzir_learning_rate = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    salvar_melhor_modelo = ModelCheckpoint(
        filepath=CAMINHO_MODELO,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    print("\n" + "=" * 60)
    print("ETAPA 1 — BASE CONGELADA")
    print("=" * 60)

    historico_inicial = modelo.fit(
        x_train,
        y_train,
        validation_data=(
            x_val,
            y_val
        ),
        epochs=20,
        batch_size=16,
        class_weight=pesos_classes,
        callbacks=[
            early_stopping,
            reduzir_learning_rate,
            salvar_melhor_modelo
        ]
    )

    print("\nAvaliação após a primeira etapa:")

    resultados = modelo.evaluate(
        x_val,
        y_val,
        verbose=1
    )

    for nome, valor in zip(
        modelo.metrics_names,
        resultados
    ):
        print(f"{nome}: {valor:.4f}")

    print(
        f"\nModelo salvo em: {CAMINHO_MODELO}"
    )

    return (
        modelo,
        base_model,
        x_train,
        x_val,
        y_train,
        y_val,
        pesos_classes,
        historico_inicial
    )


if __name__ == "__main__":
    treinar_modelo()
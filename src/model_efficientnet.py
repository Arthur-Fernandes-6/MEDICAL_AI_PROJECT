import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    RandomRotation,
    RandomZoom,
    RandomTranslation,
    RandomContrast,
    GlobalAveragePooling2D,
    BatchNormalization,
    Dense,
    Dropout
)
from tensorflow.keras.applications import (
    EfficientNetB0
)


def criar_modelo_efficientnet():
    entrada = Input(
        shape=(224, 224, 3),
        name="imagem_entrada"
    )

    # Aumento de dados moderado
    x = RandomRotation(
        factor=0.05,
        name="random_rotation" 
    )(entrada)

    x = RandomZoom(
        height_factor=0.10,
        width_factor=0.10,
        name="random_zoom"
    )(x)

    x = RandomTranslation(
        height_factor=0.05,
        width_factor=0.05,
        name="random_translation"
    )(x)

    x = RandomContrast(
        factor=0.10,
        name="random_contrast"
    )(x)

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )

    # Primeira etapa do transfer learning:
    # congela os pesos pré-treinados
    base_model.trainable = False

    # training=False mantém as camadas de BatchNormalization
    # da EfficientNet em modo de inferência
    x = base_model(
        x,
        training=False
    )

    x = GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = BatchNormalization(
        name="batch_normalization"
    )(x)

    x = Dense(
        256,
        activation="relu",
        name="dense_256"
    )(x)

    x = Dropout(
        0.40,
        name="dropout"
    )(x)

    saida = Dense(
        1,
        activation="sigmoid",
        name="classificacao"
    )(x)

    modelo = Model(
        inputs=entrada,
        outputs=saida,
        name="NeuroScan_EfficientNetB0"
    )

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(
                name="precision"
            ),
            tf.keras.metrics.Recall(
                name="recall"
            ),
            tf.keras.metrics.AUC(
                name="auc"
            )
        ]
    )

    return modelo, base_model
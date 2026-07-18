import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from dataset_v2 import carregar_dataset_v2
from model import criar_modelo


x, y = carregar_dataset_v2("brain_tumor_mri_dataset", "Training")
# x recebe todas as 5600 imagens do training 
# y  recebe todos os rotulos das 5600 imagens 

x_train, x_val, y_train, y_val = train_test_split(
    x,
    y,
    test_size= 0.2,
    random_state=42,
    stratify= y
)

classes = np.unique(y_train)

# Calcula os pesos automaticamente
pesos = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

# Converte para um dicionário
pesos_classes = dict(zip(classes, pesos))

print("Pesos das classes:")
print(pesos_classes)

modelo = criar_modelo()
modelo.summary()

parada_antecipada = EarlyStopping(# para o treinamento em caso de overfitting 
    monitor= "val_loss",
    patience= 3,
    restore_best_weights= True
)

historico = modelo.fit(
    x_train,
    y_train,
    epochs = 30,
    batch_size = 32,
    validation_data = (x_val, y_val),
    callbacks=[parada_antecipada],
    class_weight = pesos_classes
)

perda, acuracia = modelo.evaluate(x_val, y_val)
print(f"Perda na validação: {perda:.4f}")
print(f"Acurácia na validação: {acuracia:.4f}")

modelo.save("models/brain_tumor_cnn_v2.keras")
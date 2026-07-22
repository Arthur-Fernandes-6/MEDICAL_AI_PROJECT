from datasetv1 import carregar_dataset
from baseline_model import criar_modelo_baseline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

x, y = carregar_dataset("datasets")

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size= 0.2,
    random_state=42,
    stratify= y
)

modelo_baseline = criar_modelo_baseline()
modelo_baseline.summary()

parada_antecipada = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True # recupera os pesos da melhor época (menor val_loss) ao final do treinamento
)

historico_baseline = modelo_baseline.fit(
                    x_train,
                    y_train,
                    epochs = 30,
                    batch_size = 32,
                    validation_data = (x_test, y_test),
                    callbacks=[parada_antecipada]
                    )

loss_baseline, accuracy_baseline = modelo_baseline.evaluate(
    x_test,
    y_test
)

print(f"Loss baseline: {loss_baseline:.4f}")
print(f"Accuracy baseline: {accuracy_baseline * 100:.2f}%")

probabilidade_baseline = modelo_baseline.predict(x_test)

y_pred_baseline = (probabilidade_baseline >= 0.5).astype(int)
y_pred_baseline = y_pred_baseline.ravel()


matriz_baseline = confusion_matrix(
    y_test,
    y_pred_baseline
)

print("Matriz de confusão do baseline:")
print(matriz_baseline)

visualizacao = ConfusionMatrixDisplay(
    confusion_matrix=matriz_baseline,
    display_labels=["Sem tumor", "Tumor"]
)

visualizacao.plot()
plt.title("Matriz de Confusão — Baseline")
plt.show()


print(
    classification_report(
        y_test,
        y_pred_baseline,
        target_names=["Sem tumor", "Tumor"]
    )
)

modelo_baseline.save("models/brain_tumor_baseline.keras")

plt.plot(
    historico_baseline.history["accuracy"],
    label="Treino"
)

plt.plot(
    historico_baseline.history["val_accuracy"],
    label="Validação"
)

plt.xlabel("Épocas")
plt.ylabel("Acurácia")
plt.title("Acurácia — Modelo Baseline")
plt.legend()
plt.show()

plt.plot(
    historico_baseline.history["loss"],
    label="Treino"
)

plt.plot(
    historico_baseline.history["val_loss"],
    label="Validação"
)

plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.title("Loss — Modelo Baseline")
plt.legend()
plt.show()


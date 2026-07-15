from dataset import carregar_dataset
from sklearn.model_selection import train_test_split
from model import criar_modelo
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
import os
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
#carregar o dataset
x,y = carregar_dataset("datasets")

#separar treino e teste
x_train, x_test, y_train, y_test = train_test_split(
    x,
    # variavel que armazena a lista com imagens
    y,
    # variavel que armazena a lista com os rotulos
    test_size= 0.2,
    # 20% de imagens separados para teste
    random_state= 42, 
    stratify=y
)

#criar o modelo
modelo = criar_modelo()
modelo.summary()# mostra cada camada, formato de saída, número de parâmetros

# configurar o EarlyStopping
# monitora a perda de validação e parar o treinamento se não houver melhora por 3 épocas consecutivas
parada_antecipada = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

#treinar
historico = modelo.fit(
                x_train,
                y_train,
                epochs = 30,#quantas vezes o modelo vai ver todas as imagens
                batch_size = 32,#grupo de imagens que serão processadas de uma vez
                validation_data = (x_test, y_test),#avaliar o modelo com o conjunto de teste
                callbacks=[parada_antecipada]#para parar o treinamento se não houver melhora na validação
                )

#avaliar
loss, accuracy = modelo.evaluate(x_test,y_test)

print(f"Loss final: {loss:.4f}")
print(f"Acuraccy : {accuracy*100:.2f}%")
print(os.getcwd()) 

print(f"Loss final: {loss:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

# Fazer previsões no conjunto de teste
probabilidades = modelo.predict(x_test)

# Converter probabilidades em classes
y_pred = (probabilidades >= 0.5).astype(int)
y_pred = y_pred.ravel()

# Criar matriz de confusão
matriz = confusion_matrix(y_test, y_pred)

print("Matriz de confusão:")
print(matriz)

# Exibir matriz
visualizacao = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=["Sem tumor", "Tumor"]
)

visualizacao.plot()
plt.title("Matriz de Confusão")
plt.show()

# Relatório de classificação
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Sem tumor", "Tumor"]
    )
)

#salvar o modelo
modelo.save('models/brain_tumor_cnn.keras')


print(type(plt))
print(plt)
plt.plot(historico.history["accuracy"], label="Treino")
plt.plot(historico.history["val_accuracy"], label="Validação")
plt.xlabel("Épocas")
plt.ylabel("Acurácia")
plt.legend()
plt.show()

plt.plot(historico.history["loss"], label="Treino")
plt.plot(historico.history["val_loss"], label="Validação")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.legend()
plt.show()
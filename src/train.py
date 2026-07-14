from dataset import carregar_dataset
from sklearn.model_selection import train_test_split
from model import criar_modelo

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


#treinar
historico = modelo.fit(
                x_train,
                y_train,
                epochs = 15,
                batch_size = 32,
                validation_data = (x_test, y_test)
                )

#avaliar
loss, accuracy = modelo.evaluate(x_test,y_test)
print(f"Loss final: {loss:.4f}")
print(f"Acuraccy : {accuracy*100:.2f}%")

#salvar o modelo
modelo.save('models/brain_tumor_cnn.keras')

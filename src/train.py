from dataset import carregar_dataset
from sklearn.model_selection import train_test_split


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
#treinar
#avaliar
#salvar o modelo
print("Formato de X treino:", x_train.shape)
print("Formato de X teste:", x_test.shape)

print("Formato de y treino:", y_train.shape)
print("Formato de y teste:", y_test.shape)

print("Quantidade total:", len(x))
print("Quantidade de treino:", len(x_train))
print("Quantidade de teste:", len(x_test))
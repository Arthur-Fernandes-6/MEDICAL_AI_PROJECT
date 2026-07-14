from tensorflow import keras
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

def criar_modelo():
    modelo = keras.Sequential([
        # encontra os padrões
        Conv2D(
            filters =32, kernel_size = (3, 3), 
            activation = 'relu', 
            input_shape = (224, 224, 3)
            ),
        MaxPooling2D(pool_size = (2,2)),
        #mantem os padrões mais importantes

        # Aprende padrões mais complexos
        Conv2D(
            filters=64, 
            kernel_size=(3, 3), 
            activation='relu'), 

        MaxPooling2D(pool_size=(2, 2)),
        
        # transforma  a matriz da imagem em um array
        Flatten(),
        

        # Analisa as características encontradas
        Dense(128, activation= 'relu'),
        Dense(1, activation='sigmoid')
    ])

    return modelo
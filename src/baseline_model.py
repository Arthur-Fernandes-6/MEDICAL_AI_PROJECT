from tensorflow import keras
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

def criar_modelo_baseline():
    modelo = keras.Sequential([

        # Primeira camada convolucional
        Conv2D(
            filters= 16,
            kernel_size=(3,3),
            activation='relu',
            input_shape = (224, 224, 3)
        ),

        # Reduz a dimensionalidade da imagem
        MaxPooling2D(pool_size = (2,2)),

        Conv2D(
            filters= 32,
            kernel_size=(3,3),
            activation='relu'
            ),
        
        MaxPooling2D(pool_size = (2,2)),

        # Converte os mapas de características em um vetor
        Flatten(),

        # Camada totalmente conectada
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

# Ensina a IA a aprender 
# Mede o erro da IA
# acompanha o desempenho 
    modelo.compile(
        optimizer = 'adam',# funciona muito bem em CNN
        loss = 'binary_crossentropy',
        metrics = ['accuracy']
    )

    return modelo
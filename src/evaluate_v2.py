import numpy as np
from tensorflow.keras.models import load_model
from dataset_v2 import carregar_dataset_v2
from preprocess import preprocessar_imagem

modelo = load_model("models/brain_tumor_cnn.keras")



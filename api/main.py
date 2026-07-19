import os
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

#cria a rota para a API
app = FastAPI(
    title="Brain Tumor Detection API",
    description=(
        "API educacional para classificação de imagens de "
        "ressonância magnética utilizando uma CNN."
    ),
    version="1.0.0"
)


# Permite que o frontend acesse a API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Caminho absoluto da raiz do projeto
CAMINHO_PROJETO = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)


CAMINHO_MODELO = os.path.join(
    CAMINHO_PROJETO,
    "models",
    "brain_tumor_cnn_v2.keras"
)


# Carrega o modelo apenas uma vez
try:
    modelo = load_model(CAMINHO_MODELO)
    modelo_carregado = True
    erro_modelo = None


except Exception as erro:
    modelo = None
    modelo_carregado = False
    erro_modelo = str(erro)


@app.get("/")
def home():
    return {
        "mensagem": "Brain Tumor Detection API funcionando!"
    }

# Rota para verificar o status da API
@app.get("/api/status")
def status():
    return {
        "status": "online",
        "modelo": "Brain Tumor CNN V2",
        "modelo_carregado": modelo_carregado,
        "versao": "1.0.0"
    }

# Função para pré-processar a imagem
def preprocessar_imagem(conteudo: bytes) -> np.ndarray:
    try:
        imagem = Image.open(
            io.BytesIO(conteudo)
        ).convert("RGB")# a imagem é convertida para RGB para garantir que tenha 3 canais de cor, mesmo que a imagem original seja em escala de cinza ou tenha um canal alfa (transparência)

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="O arquivo enviado não é uma imagem válida."
        )

    imagem = imagem.resize((224, 224)) # a imagem é redimensionada para 224x224 pixels, que é o tamanho esperado pelo modelo CNN treinado

    array_imagem = np.array(
        imagem,
        dtype=np.float32
    )

    array_imagem = array_imagem / 255.0 # a imagem é normalizada para que os valores dos pixels fiquem entre 0 e 1, o que ajuda na convergência do modelo durante a previsão

    array_imagem = np.expand_dims(
        array_imagem,
        axis=0 
    )

    return array_imagem

# Rota para previsão de tumor cerebral
@app.post("/api/predict")
async def predict(
    arquivo: UploadFile = File(...)# arquivo de imagem enviado pelo usuário
):
    if not modelo_carregado:
        raise HTTPException(
            status_code=503,
            detail=(
                "O modelo não foi carregado corretamente. "
                f"Erro: {erro_modelo}"
            )
        )

    tipos_permitidos = [
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/bmp",
        "image/webp"
    ]

    if arquivo.content_type not in tipos_permitidos:
        raise HTTPException(
            status_code=400,
            detail=(
                "Formato de arquivo não permitido. "
                "Envie uma imagem JPG, JPEG, PNG, BMP ou WEBP."
            )
        )

    conteudo = await arquivo.read()

    if not conteudo:# verifica se o arquivo enviado pelo usuário está vazio
        raise HTTPException(
            status_code=400,
            detail="O arquivo enviado está vazio."
        )

    imagem_processada = preprocessar_imagem(
        conteudo
    )

    previsao = modelo.predict( # faz a previsão da imagem enviada pelo usuário
        imagem_processada,
        verbose=0 
    )

    probabilidade_tumor = float(
        previsao[0][0] # pega a probabilidade de tumor da previsão
    )

    if probabilidade_tumor >= 0.5:
        classificacao = "Tumor"
        confianca = probabilidade_tumor

    else:
        classificacao = "Sem tumor"
        confianca = 1 - probabilidade_tumor

    return {
        "classificacao": classificacao,
        "probabilidade_tumor": round(
            probabilidade_tumor * 100,
            2
        ),
        "confianca": round(
            confianca * 100,
            2
        ),
        "limiar": 0.5,
        "arquivo": arquivo.filename,
        "aviso": (
            "Resultado computacional para fins educacionais. "
            "Não representa diagnóstico médico."
        )
    }
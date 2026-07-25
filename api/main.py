import os
import cv2
import tempfile
import numpy as np
from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException
)
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api.gradcam_explainer import (
    gerar_explicacao_gradcam
)
from src.preprocess_brain_crop import (
    preprocessar_brain_crop
)


app = FastAPI(
    title="Brain Tumor Detection API",
    description=(
        "API educacional para classificação de imagens de "
        "ressonância magnética utilizando EfficientNetB0 "
        "com Brain Crop."
    ),
    version="2.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


CAMINHO_PROJETO = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)


CAMINHO_MODELO = os.path.join(
    CAMINHO_PROJETO,
    "models",
    "brain_tumor_efficientnet_cropped_best.keras"
)


CAMINHO_AMOSTRAS = os.path.join(
    CAMINHO_PROJETO,
    "datasets"
)


CAMINHO_FRONTEND = os.path.join(
    CAMINHO_PROJETO,
    "frontend"
)


app.mount(
    "/samples",
    StaticFiles(
        directory=CAMINHO_AMOSTRAS
    ),
    name="samples"
)


app.mount(
    "/frontend",
    StaticFiles(
        directory=CAMINHO_FRONTEND
    ),
    name="frontend"
)


print("=" * 60)
print("CAMINHO DO MODELO:")
print(CAMINHO_MODELO)
print("EXISTE?")
print(os.path.exists(CAMINHO_MODELO))

if os.path.exists(CAMINHO_MODELO):
    tamanho_mb = (
        os.path.getsize(CAMINHO_MODELO)
        / 1024
        / 1024
    )

    print(
        f"Tamanho: {tamanho_mb:.2f} MB"
    )

print("=" * 60)


try:
    modelo = load_model(
        CAMINHO_MODELO
    )

    modelo_carregado = True
    erro_modelo = None

    print(
        "EfficientNet Cropped carregada com sucesso!"
    )

except Exception as erro:
    print(
        "ERRO AO CARREGAR O MODELO:"
    )

    print(
        erro
    )

    modelo = None
    modelo_carregado = False
    erro_modelo = str(
        erro
    )


@app.get("/")
def home():
    caminho_index = os.path.join(
        CAMINHO_FRONTEND,
        "index.html"
    )

    return FileResponse(
        caminho_index
    )


@app.get("/api/samples")
def listar_amostras():
    amostras = []

    pastas = {
        "yes": "yes",
        "no": "no"
    }

    extensoes_permitidas = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    )

    for nome_pasta, label in pastas.items():
        caminho_pasta = os.path.join(
            CAMINHO_AMOSTRAS,
            nome_pasta
        )

        if not os.path.exists(
            caminho_pasta
        ):
            continue

        arquivos = os.listdir(
            caminho_pasta
        )

        arquivos_imagem = [
            arquivo
            for arquivo in arquivos
            if arquivo.lower().endswith(
                extensoes_permitidas
            )
        ]

        arquivos_imagem.sort()

        for arquivo in arquivos_imagem[:4]:
            amostras.append(
                {
                    "filename": arquivo,
                    "label": label,
                    "url": (
                        f"/samples/"
                        f"{nome_pasta}/"
                        f"{arquivo}"
                    )
                }
            )

    return amostras


@app.get("/api/status")
def status():
    return {
        "status": "online",
        "modelo": "EfficientNetB0 com Brain Crop",
        "arquivo_modelo": (
            "brain_tumor_efficientnet_cropped_best.keras"
        ),
        "modelo_carregado": modelo_carregado,
        "versao": "2.0.0"
    }


def preprocessar_imagem(
    conteudo: bytes,
    extensao: str = ".jpg"
) -> tuple[np.ndarray, np.ndarray]:

    caminho_temporario = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extensao,
            delete=False
        ) as arquivo_temporario:

            arquivo_temporario.write(
                conteudo
            )

            caminho_temporario = (
                arquivo_temporario.name
            )

        imagem_cropped = preprocessar_brain_crop(
            caminho_temporario,
            normalizar=False
        )

        if imagem_cropped is None:
            raise ValueError(
                "O Brain Crop não conseguiu processar a imagem."
            )

        imagem_cropped = cv2.resize(
            imagem_cropped,
            (224, 224)
        )

        imagem_rgb = cv2.cvtColor(
            imagem_cropped,
            cv2.COLOR_BGR2RGB
        )

        imagem_modelo = imagem_rgb.astype(
            np.float32
        )

        imagem_modelo = np.expand_dims(
            imagem_modelo,
            axis=0
        )

        return imagem_rgb, imagem_modelo

    except Exception as erro:
        raise HTTPException(
            status_code=400,
            detail=(
                "Não foi possível processar a imagem. "
                f"Detalhes: {erro}"
            )
        )

    finally:
        if (
            caminho_temporario is not None
            and os.path.exists(
                caminho_temporario
            )
        ):
            os.remove(
                caminho_temporario
            )


@app.post("/api/predict")
async def predict(
    arquivo: UploadFile = File(...)
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
                "Envie JPG, JPEG, PNG, BMP ou WEBP."
            )
        )

    conteudo = await arquivo.read()

    if not conteudo:
        raise HTTPException(
            status_code=400,
            detail="O arquivo enviado está vazio."
        )

    extensao = os.path.splitext(
        arquivo.filename or ""
    )[1].lower()

    if extensao not in (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    ):
        extensao = ".jpg"

    imagem_cropped, imagem_processada = (
        preprocessar_imagem(
            conteudo,
            extensao
        )
    )

    print("\nShape:", imagem_processada.shape)
    print("Tipo:", imagem_processada.dtype)
    print("Mínimo:", imagem_processada.min())
    print("Máximo:", imagem_processada.max())

    previsao = modelo.predict(
        imagem_processada,
        verbose=0
    )

    probabilidade_tumor = float(
        np.asarray(
            previsao
        ).reshape(-1)[0]
    )

    try:
        explicacao_gradcam = gerar_explicacao_gradcam(
        modelo=modelo,
        imagem_rgb=imagem_cropped,
        imagem_modelo=imagem_processada
    )

        gradcam_imagem = explicacao_gradcam[
        "imagem_base64"
        ]

        print("Grad-CAM gerado com sucesso!")

    except Exception as erro_gradcam:
        print("ERRO AO GERAR GRAD-CAM:")
        print(erro_gradcam)

        gradcam_imagem = None

    limiar = 0.5

    if probabilidade_tumor >= limiar:
        classificacao = "Tumor"
        confianca = probabilidade_tumor

    else:
        classificacao = "Sem tumor"
        confianca = 1.0 - probabilidade_tumor

    return {
        "classificacao": classificacao,
        "probabilidade_tumor": probabilidade_tumor,
        "confianca": confianca,
        "confianca_percentual": min(
            round(confianca * 100, 4),
            99.9999
        ),
        "limiar": limiar,
        "modelo": "EfficientNetB0 com Brain Crop",
        "gradcam_imagem": gradcam_imagem,
        "arquivo": arquivo.filename,
        "simulado": False,
        "aviso": (
            "Resultado computacional para fins educacionais. "
            "Não representa diagnóstico médico."
        )
    }
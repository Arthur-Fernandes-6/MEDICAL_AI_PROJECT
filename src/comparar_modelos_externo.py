import os
import csv
import json
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from preprocess_brain_crop import preprocessar_brain_crop


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_TESTE_EXTERNO = "externalTest"

PASTA_SEM_TUMOR = os.path.join(
    PASTA_TESTE_EXTERNO,
    "noTumor"
)

PASTA_TUMOR = os.path.join(
    PASTA_TESTE_EXTERNO,
    "yes"
)

PASTA_RESULTADOS = "resultados_comparacao_modelos"

LIMIAR = 0.5

TAMANHO_IMAGEM = (224, 224)

EXTENSOES_VALIDAS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
)


# ============================================================
# MODELOS
# ============================================================

MODELOS = [
    {
        "nome": "CNN V2",
        "caminho": "models/brain_tumor_cnn_v2.keras",
        "tipo_preprocessamento": "cnn"
    },
    {
        "nome": "EfficientNet V1",
        "caminho": "models/brain_tumor_efficientnet_v1.keras",
        "tipo_preprocessamento": "efficientnet"
    },
    {
        "nome": "EfficientNet Cropped",
        "caminho": (
            "models/"
            "brain_tumor_efficientnet_cropped_best.keras"
        ),
        "tipo_preprocessamento": "efficientnet_cropped"
    }
]


# ============================================================
# LISTAR IMAGENS
# ============================================================

def listar_imagens_recursivamente(pasta):
    imagens = []

    if not os.path.isdir(pasta):
        return imagens

    for raiz, _, arquivos in os.walk(pasta):
        for nome_arquivo in arquivos:
            if nome_arquivo.lower().endswith(
                EXTENSOES_VALIDAS
            ):
                imagens.append(
                    os.path.join(
                        raiz,
                        nome_arquivo
                    )
                )

    imagens.sort()

    return imagens


def criar_lista_dataset():
    dados = []

    imagens_sem_tumor = listar_imagens_recursivamente(
        PASTA_SEM_TUMOR
    )

    imagens_tumor = listar_imagens_recursivamente(
        PASTA_TUMOR
    )

    for caminho in imagens_sem_tumor:
        dados.append(
            {
                "caminho": caminho,
                "rotulo": 0,
                "classe": "Sem tumor"
            }
        )

    for caminho in imagens_tumor:
        dados.append(
            {
                "caminho": caminho,
                "rotulo": 1,
                "classe": "Tumor"
            }
        )

    return dados


# ============================================================
# PRÉ-PROCESSAMENTO
# ============================================================

def carregar_imagem_rgb(caminho):
    imagem = cv2.imread(
        caminho
    )

    if imagem is None:
        raise ValueError(
            f"Não foi possível abrir: {caminho}"
        )

    imagem = cv2.resize(
        imagem,
        TAMANHO_IMAGEM
    )

    imagem = cv2.cvtColor(
        imagem,
        cv2.COLOR_BGR2RGB
    )

    return imagem


def preprocessar_cnn(caminho):
    """
    Ajuste esta função somente se o seu CNN V2 tiver usado
    outro pré-processamento durante o treinamento.
    """

    imagem = carregar_imagem_rgb(
        caminho
    )

    imagem = imagem.astype(
        np.float32
    ) / 255.0

    return imagem


def preprocessar_efficientnet(caminho):
    """
    Mantém os pixels em 0-255, igual ao pipeline utilizado
    pela EfficientNet do projeto.
    """

    imagem = carregar_imagem_rgb(
        caminho
    )

    imagem = imagem.astype(
        np.float32
    )

    return imagem


def preprocessar_efficientnet_cropped(caminho):
    imagem = preprocessar_brain_crop(
        caminho,
        normalizar=False
    )

    if imagem is None:
        raise ValueError(
            f"Brain Crop falhou em: {caminho}"
        )

    imagem = cv2.resize(
        imagem,
        TAMANHO_IMAGEM
    )

    imagem = cv2.cvtColor(
        imagem,
        cv2.COLOR_BGR2RGB
    )

    imagem = imagem.astype(
        np.float32
    )

    return imagem


def preprocessar_imagem(
    caminho,
    tipo_preprocessamento
):
    if tipo_preprocessamento == "cnn":
        return preprocessar_cnn(
            caminho
        )

    if tipo_preprocessamento == "efficientnet":
        return preprocessar_efficientnet(
            caminho
        )

    if tipo_preprocessamento == "efficientnet_cropped":
        return preprocessar_efficientnet_cropped(
            caminho
        )

    raise ValueError(
        "Tipo de pré-processamento desconhecido: "
        f"{tipo_preprocessamento}"
    )


# ============================================================
# PREPARAR DATASET PARA UM MODELO
# ============================================================

def preparar_dataset(
    dados,
    tipo_preprocessamento
):
    imagens = []
    rotulos = []
    caminhos_validos = []
    erros = []

    total = len(dados)

    for indice, item in enumerate(
        dados,
        start=1
    ):
        caminho = item["caminho"]
        rotulo = item["rotulo"]

        try:
            imagem = preprocessar_imagem(
                caminho,
                tipo_preprocessamento
            )

            imagens.append(
                imagem
            )

            rotulos.append(
                rotulo
            )

            caminhos_validos.append(
                caminho
            )

        except Exception as erro:
            erros.append(
                {
                    "imagem": caminho,
                    "erro": str(erro)
                }
            )

            print(
                f"Erro ao processar {caminho}: {erro}"
            )

        if indice % 20 == 0 or indice == total:
            print(
                f"{indice}/{total} imagens preparadas"
            )

    x = np.array(
        imagens,
        dtype=np.float32
    )

    y = np.array(
        rotulos,
        dtype=np.int32
    )

    return x, y, caminhos_validos, erros


# ============================================================
# NORMALIZAR SAÍDA DO MODELO
# ============================================================

def extrair_probabilidade_tumor(saidas):
    saidas = np.asarray(
        saidas
    )

    # Modelo binário com sigmoid:
    # shape = (N, 1)
    if saidas.ndim == 2 and saidas.shape[1] == 1:
        return saidas[:, 0]

    # Modelo binário com saída direta:
    # shape = (N,)
    if saidas.ndim == 1:
        return saidas

    # Modelo com duas saídas softmax:
    # índice 0 = sem tumor
    # índice 1 = tumor
    if saidas.ndim == 2 and saidas.shape[1] == 2:
        return saidas[:, 1]

    raise ValueError(
        "Formato de saída não reconhecido: "
        f"{saidas.shape}"
    )


# ============================================================
# CALCULAR MÉTRICAS
# ============================================================

def calcular_metricas(
    y_real,
    probabilidades,
    predicoes
):
    matriz = confusion_matrix(
        y_real,
        predicoes,
        labels=[0, 1]
    )

    vn = int(matriz[0, 0])
    fp = int(matriz[0, 1])
    fn = int(matriz[1, 0])
    vp = int(matriz[1, 1])

    especificidade = (
        vn / (vn + fp)
        if (vn + fp) > 0
        else 0.0
    )

    try:
        auc = roc_auc_score(
            y_real,
            probabilidades
        )
    except ValueError:
        auc = 0.0

    return {
        "accuracy": float(
            accuracy_score(
                y_real,
                predicoes
            )
        ),
        "precision": float(
            precision_score(
                y_real,
                predicoes,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_real,
                predicoes,
                zero_division=0
            )
        ),
        "specificity": float(
            especificidade
        ),
        "f1_score": float(
            f1_score(
                y_real,
                predicoes,
                zero_division=0
            )
        ),
        "auc": float(
            auc
        ),
        "verdadeiros_negativos": vn,
        "falsos_positivos": fp,
        "falsos_negativos": fn,
        "verdadeiros_positivos": vp
    }


# ============================================================
# SALVAR PREDIÇÕES
# ============================================================

def salvar_predicoes(
    nome_modelo,
    caminhos,
    y_real,
    probabilidades,
    predicoes
):
    nome_seguro = (
        nome_modelo
        .lower()
        .replace(" ", "_")
    )

    caminho_csv = os.path.join(
        PASTA_RESULTADOS,
        f"predicoes_{nome_seguro}.csv"
    )

    with open(
        caminho_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as arquivo:
        escritor = csv.writer(
            arquivo
        )

        escritor.writerow(
            [
                "imagem",
                "rotulo_real",
                "probabilidade_tumor",
                "rotulo_predito",
                "acertou"
            ]
        )

        for (
            caminho,
            real,
            probabilidade,
            predito
        ) in zip(
            caminhos,
            y_real,
            probabilidades,
            predicoes
        ):
            escritor.writerow(
                [
                    caminho,
                    int(real),
                    float(probabilidade),
                    int(predito),
                    int(real == predito)
                ]
            )

    return caminho_csv


# ============================================================
# AVALIAR UM MODELO
# ============================================================

def avaliar_modelo(
    configuracao,
    dados
):
    nome = configuracao["nome"]
    caminho_modelo = configuracao["caminho"]
    tipo_preprocessamento = configuracao[
        "tipo_preprocessamento"
    ]

    print("\n" + "=" * 70)
    print(f"AVALIANDO: {nome}")
    print("=" * 70)

    if not os.path.isfile(
        caminho_modelo
    ):
        raise FileNotFoundError(
            f"Modelo não encontrado: {caminho_modelo}"
        )

    print(
        f"Modelo: {caminho_modelo}"
    )

    print(
        f"Pré-processamento: {tipo_preprocessamento}"
    )

    modelo = tf.keras.models.load_model(
        caminho_modelo
    )

    x, y, caminhos, erros = preparar_dataset(
        dados,
        tipo_preprocessamento
    )

    print(
        "\nFormato de X:",
        x.shape
    )

    print(
        "Formato de y:",
        y.shape
    )

    print(
        "Mínimo:",
        x.min()
    )

    print(
        "Máximo:",
        x.max()
    )

    saidas = modelo.predict(
        x,
        batch_size=32,
        verbose=1
    )

    probabilidades = extrair_probabilidade_tumor(
        saidas
    )

    predicoes = (
        probabilidades >= LIMIAR
    ).astype(
        np.int32
    )

    metricas = calcular_metricas(
        y,
        probabilidades,
        predicoes
    )

    relatorio = classification_report(
        y,
        predicoes,
        labels=[0, 1],
        target_names=[
            "Sem tumor",
            "Tumor"
        ],
        digits=4,
        zero_division=0
    )

    caminho_predicoes = salvar_predicoes(
        nome_modelo=nome,
        caminhos=caminhos,
        y_real=y,
        probabilidades=probabilidades,
        predicoes=predicoes
    )

    print("\nResultados:")

    print(
        f"Accuracy:    "
        f"{metricas['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision:   "
        f"{metricas['precision'] * 100:.2f}%"
    )

    print(
        f"Recall:      "
        f"{metricas['recall'] * 100:.2f}%"
    )

    print(
        f"Specificity: "
        f"{metricas['specificity'] * 100:.2f}%"
    )

    print(
        f"F1-score:    "
        f"{metricas['f1_score'] * 100:.2f}%"
    )

    print(
        f"AUC:         "
        f"{metricas['auc'] * 100:.2f}%"
    )

    print("\nMatriz de confusão:")

    print(
        [
            [
                metricas["verdadeiros_negativos"],
                metricas["falsos_positivos"]
            ],
            [
                metricas["falsos_negativos"],
                metricas["verdadeiros_positivos"]
            ]
        ]
    )

    print("\nRelatório:")

    print(
        relatorio
    )

    return {
        "nome": nome,
        "caminho_modelo": caminho_modelo,
        "tipo_preprocessamento": tipo_preprocessamento,
        "quantidade_imagens": int(
            len(y)
        ),
        "quantidade_erros_preprocessamento": int(
            len(erros)
        ),
        "metricas": metricas,
        "arquivo_predicoes": caminho_predicoes
    }


# ============================================================
# SALVAR COMPARAÇÃO
# ============================================================

def salvar_comparacao(resultados):
    caminho_json = os.path.join(
        PASTA_RESULTADOS,
        "comparacao_modelos_externo.json"
    )

    with open(
        caminho_json,
        "w",
        encoding="utf-8"
    ) as arquivo:
        json.dump(
            resultados,
            arquivo,
            indent=4,
            ensure_ascii=False
        )

    caminho_csv = os.path.join(
        PASTA_RESULTADOS,
        "comparacao_modelos_externo.csv"
    )

    with open(
        caminho_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as arquivo:
        escritor = csv.writer(
            arquivo
        )

        escritor.writerow(
            [
                "modelo",
                "accuracy",
                "precision",
                "recall",
                "specificity",
                "f1_score",
                "auc",
                "vn",
                "fp",
                "fn",
                "vp"
            ]
        )

        for resultado in resultados:
            metricas = resultado["metricas"]

            escritor.writerow(
                [
                    resultado["nome"],
                    metricas["accuracy"],
                    metricas["precision"],
                    metricas["recall"],
                    metricas["specificity"],
                    metricas["f1_score"],
                    metricas["auc"],
                    metricas["verdadeiros_negativos"],
                    metricas["falsos_positivos"],
                    metricas["falsos_negativos"],
                    metricas["verdadeiros_positivos"]
                ]
            )

    return caminho_json, caminho_csv


# ============================================================
# MOSTRAR RANKING
# ============================================================

def mostrar_ranking(resultados):
    ranking = sorted(
        resultados,
        key=lambda item: (
            item["metricas"]["f1_score"],
            item["metricas"]["recall"],
            item["metricas"]["accuracy"],
            item["metricas"]["auc"]
        ),
        reverse=True
    )

    print("\n" + "=" * 70)
    print("RANKING DOS MODELOS")
    print("=" * 70)

    for posicao, resultado in enumerate(
        ranking,
        start=1
    ):
        metricas = resultado["metricas"]

        print(
            f"\n{posicao}º - {resultado['nome']}"
        )

        print(
            f"F1:       "
            f"{metricas['f1_score'] * 100:.2f}%"
        )

        print(
            f"Recall:   "
            f"{metricas['recall'] * 100:.2f}%"
        )

        print(
            f"Accuracy: "
            f"{metricas['accuracy'] * 100:.2f}%"
        )

        print(
            f"AUC:      "
            f"{metricas['auc'] * 100:.2f}%"
        )

    melhor = ranking[0]

    print("\n" + "-" * 70)

    print(
        "Melhor resultado pelo critério atual:"
    )

    print(
        melhor["nome"]
    )

    print(
        "\nCritério de desempate:"
        " F1, Recall, Accuracy e AUC."
    )


# ============================================================
# EXECUÇÃO
# ============================================================

def main():
    os.makedirs(
        PASTA_RESULTADOS,
        exist_ok=True
    )

    dados = criar_lista_dataset()

    if len(dados) == 0:
        raise RuntimeError(
            "Nenhuma imagem foi encontrada em "
            f"{PASTA_TESTE_EXTERNO}"
        )

    quantidade_sem_tumor = sum(
        item["rotulo"] == 0
        for item in dados
    )

    quantidade_tumor = sum(
        item["rotulo"] == 1
        for item in dados
    )

    print("\n" + "=" * 70)
    print("CONJUNTO EXTERNO")
    print("=" * 70)

    print(
        f"Sem tumor: {quantidade_sem_tumor}"
    )

    print(
        f"Tumor:     {quantidade_tumor}"
    )

    print(
        f"Total:     {len(dados)}"
    )

    resultados = []

    for configuracao in MODELOS:
        resultado = avaliar_modelo(
            configuracao,
            dados
        )

        resultados.append(
            resultado
        )

    caminho_json, caminho_csv = salvar_comparacao(
        resultados
    )

    mostrar_ranking(
        resultados
    )

    print("\nArquivos salvos:")

    print(
        caminho_json
    )

    print(
        caminho_csv
    )


if __name__ == "__main__":
    main()
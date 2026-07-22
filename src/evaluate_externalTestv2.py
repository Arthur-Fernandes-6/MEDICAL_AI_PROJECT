import os
import csv
import shutil
import numpy as np

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from preprocessv1 import preprocessar_imagem


# Caminho do modelo treinado
CAMINHO_MODELO = "models/brain_tumor_cnn_v2.keras"

# Caminho da pasta com as imagens externas
PASTA_EXTERNAL = "externalTest"

# Pasta onde serão salvas as imagens que o modelo errou
PASTA_ERROS = "false_predictions"

# Extensões permitidas
EXTENSOES_VALIDAS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


def obter_rotulo_pela_pasta(caminho_imagem):
    """
    Descobre o rótulo verdadeiro da imagem com base
    no nome das pastas.

    Tumor = 1
    Sem tumor = 0
    """

    caminho_minusculo = caminho_imagem.lower()

    if "notumor" in caminho_minusculo:
        return 0

    if "no_tumor" in caminho_minusculo:
        return 0

    if "sem_tumor" in caminho_minusculo:
        return 0

    if "tumor" in caminho_minusculo:
        return 1

    if "glioma" in caminho_minusculo:
        return 1

    if "meningioma" in caminho_minusculo:
        return 1

    if "pituitary" in caminho_minusculo:
        return 1

    return None


def obter_tipo_imagem(caminho_imagem):
    """
    Descobre a categoria específica da imagem.
    """

    caminho_minusculo = caminho_imagem.lower()

    if "glioma" in caminho_minusculo:
        return "Glioma"

    if "meningioma" in caminho_minusculo:
        return "Meningioma"

    if "pituitary" in caminho_minusculo:
        return "Pituitary"

    if (
        "notumor" in caminho_minusculo
        or "no_tumor" in caminho_minusculo
        or "sem_tumor" in caminho_minusculo
    ):
        return "Sem tumor"

    return "Tumor não especificado"


def copiar_imagem_errada(
    caminho_original,
    tipo,
    nome_arquivo,
    classe_real,
    classe_prevista
):
    """
    Copia as imagens classificadas incorretamente
    para a pasta false_predictions.
    """

    pasta_tipo = os.path.join(
        PASTA_ERROS,
        tipo.replace(" ", "_")
    )

    os.makedirs(pasta_tipo, exist_ok=True)

    nome_saida = (
        f"real_{classe_real}_"
        f"previsto_{classe_prevista}_"
        f"{nome_arquivo}"
    )

    destino = os.path.join(
        pasta_tipo,
        nome_saida
    )

    shutil.copy2(
        caminho_original,
        destino
    )


def avaliar_modelo_externo():
    print("=" * 60)
    print("AVALIAÇÃO EXTERNA DO NEUROSCAN AI")
    print("=" * 60)

    if not os.path.exists(CAMINHO_MODELO):
        print(
            f"\nErro: modelo não encontrado em:\n"
            f"{CAMINHO_MODELO}"
        )
        return

    if not os.path.exists(PASTA_EXTERNAL):
        print(
            f"\nErro: pasta externa não encontrada:\n"
            f"{PASTA_EXTERNAL}"
        )
        return

    print("\nCarregando o modelo...")

    modelo = load_model(CAMINHO_MODELO)

    print("Modelo carregado com sucesso!")

    os.makedirs(
        PASTA_ERROS,
        exist_ok=True
    )

    caminhos_imagens = []
    rotulos_reais = []
    tipos_imagens = []

    print("\nProcurando imagens externas...")

    for raiz, _, arquivos in os.walk(PASTA_EXTERNAL):

        for nome_arquivo in arquivos:

            if not nome_arquivo.lower().endswith(
                EXTENSOES_VALIDAS
            ):
                continue

            caminho_completo = os.path.join(
                raiz,
                nome_arquivo
            )

            rotulo = obter_rotulo_pela_pasta(
                caminho_completo
            )

            if rotulo is None:
                print(
                    "Imagem ignorada por não ser possível "
                    f"descobrir a classe: {caminho_completo}"
                )
                continue

            tipo = obter_tipo_imagem(
                caminho_completo
            )

            caminhos_imagens.append(
                caminho_completo
            )

            rotulos_reais.append(
                rotulo
            )

            tipos_imagens.append(
                tipo
            )

    total_imagens = len(caminhos_imagens)

    if total_imagens == 0:
        print(
            "\nNenhuma imagem válida foi encontrada."
        )
        return

    print(
        f"\nTotal de imagens encontradas: "
        f"{total_imagens}"
    )

    resultados = []
    previsoes = []
    probabilidades_tumor = []

    estatisticas_por_tipo = {}

    print("\nIniciando previsões...\n")

    for indice, caminho_imagem in enumerate(
        caminhos_imagens
    ):

        nome_arquivo = os.path.basename(
            caminho_imagem
        )

        classe_real = rotulos_reais[indice]
        tipo = tipos_imagens[indice]

        imagem = preprocessar_imagem(
            caminho_imagem
        )

        if imagem is None:
            print(
                f"Erro ao processar: {nome_arquivo}"
            )
            continue

        # Cria a dimensão do batch
        imagem_entrada = np.expand_dims(
            imagem,
            axis=0
        )

        resultado_modelo = modelo.predict(
            imagem_entrada,
            verbose=0
        )

        probabilidade_tumor = float(
            resultado_modelo[0][0]
        )

        classe_prevista = int(
            probabilidade_tumor >= 0.5
        )

        probabilidade_sem_tumor = (
            1 - probabilidade_tumor
        )

        if classe_prevista == 1:
            confianca = probabilidade_tumor
        else:
            confianca = probabilidade_sem_tumor

        acertou = (
            classe_prevista == classe_real
        )

        previsoes.append(
            classe_prevista
        )

        probabilidades_tumor.append(
            probabilidade_tumor
        )

        nome_classe_real = (
            "Tumor"
            if classe_real == 1
            else "Sem tumor"
        )

        nome_classe_prevista = (
            "Tumor"
            if classe_prevista == 1
            else "Sem tumor"
        )

        simbolo = "ACERTOU" if acertou else "ERROU"

        print(
            f"[{indice + 1}/{total_imagens}] "
            f"{nome_arquivo}"
        )

        print(
            f"Tipo: {tipo}"
        )

        print(
            f"Real: {nome_classe_real}"
        )

        print(
            f"Previsto: {nome_classe_prevista}"
        )

        print(
            f"Probabilidade de tumor: "
            f"{probabilidade_tumor * 100:.2f}%"
        )

        print(
            f"Confiança da previsão: "
            f"{confianca * 100:.2f}%"
        )

        print(
            f"Resultado: {simbolo}"
        )

        print("-" * 60)

        if tipo not in estatisticas_por_tipo:
            estatisticas_por_tipo[tipo] = {
                "total": 0,
                "acertos": 0,
                "erros": 0
            }

        estatisticas_por_tipo[tipo]["total"] += 1

        if acertou:
            estatisticas_por_tipo[tipo][
                "acertos"
            ] += 1

        else:
            estatisticas_por_tipo[tipo][
                "erros"
            ] += 1

            copiar_imagem_errada(
                caminho_original=caminho_imagem,
                tipo=tipo,
                nome_arquivo=nome_arquivo,
                classe_real=nome_classe_real,
                classe_prevista=nome_classe_prevista
            )

        resultados.append({
            "arquivo": nome_arquivo,
            "caminho": caminho_imagem,
            "tipo": tipo,
            "classe_real": nome_classe_real,
            "classe_prevista": nome_classe_prevista,
            "probabilidade_tumor": round(
                probabilidade_tumor,
                6
            ),
            "probabilidade_sem_tumor": round(
                probabilidade_sem_tumor,
                6
            ),
            "confianca": round(
                confianca,
                6
            ),
            "acertou": acertou
        })

    if len(previsoes) == 0:
        print(
            "Nenhuma imagem pôde ser avaliada."
        )
        return

    y_real = np.array(
        rotulos_reais[:len(previsoes)]
    )

    y_previsto = np.array(
        previsoes
    )

    acuracia = accuracy_score(
        y_real,
        y_previsto
    )

    matriz = confusion_matrix(
        y_real,
        y_previsto,
        labels=[0, 1]
    )

    relatorio = classification_report(
        y_real,
        y_previsto,
        labels=[0, 1],
        target_names=[
            "Sem tumor",
            "Tumor"
        ],
        zero_division=0
    )

    vn, fp, fn, vp = matriz.ravel()

    sensibilidade = (
        vp / (vp + fn)
        if (vp + fn) > 0
        else 0
    )

    especificidade = (
        vn / (vn + fp)
        if (vn + fp) > 0
        else 0
    )

    print("\n")
    print("=" * 60)
    print("RESULTADO FINAL")
    print("=" * 60)

    print(
        f"\nTotal avaliado: "
        f"{len(previsoes)}"
    )

    print(
        f"Acurácia externa: "
        f"{acuracia * 100:.2f}%"
    )

    print(
        f"Sensibilidade: "
        f"{sensibilidade * 100:.2f}%"
    )

    print(
        f"Especificidade: "
        f"{especificidade * 100:.2f}%"
    )

    print("\nMatriz de confusão:")

    print(matriz)

    print(
        "\nFormato da matriz:"
    )

    print(
        "[[Verdadeiro Negativo, Falso Positivo],"
    )

    print(
        " [Falso Negativo, Verdadeiro Positivo]]"
    )

    print(
        "\nRelatório de classificação:"
    )

    print(relatorio)

    print("=" * 60)
    print("RESULTADO POR TIPO")
    print("=" * 60)

    for tipo, dados in estatisticas_por_tipo.items():

        total = dados["total"]
        acertos = dados["acertos"]
        erros = dados["erros"]

        acuracia_tipo = (
            acertos / total
            if total > 0
            else 0
        )

        print(f"\n{tipo}")

        print(
            f"Total: {total}"
        )

        print(
            f"Acertos: {acertos}"
        )

        print(
            f"Erros: {erros}"
        )

        print(
            f"Acurácia: "
            f"{acuracia_tipo * 100:.2f}%"
        )

    salvar_resultados_csv(
        resultados
    )

    print(
        "\nAs imagens classificadas incorretamente "
        f"foram copiadas para: {PASTA_ERROS}"
    )

    print(
        "O relatório detalhado foi salvo em: "
        "resultados_externos.csv"
    )


def salvar_resultados_csv(resultados):
    campos = [
        "arquivo",
        "caminho",
        "tipo",
        "classe_real",
        "classe_prevista",
        "probabilidade_tumor",
        "probabilidade_sem_tumor",
        "confianca",
        "acertou"
    ]

    with open(
        "resultados_externos.csv",
        mode="w",
        newline="",
        encoding="utf-8-sig"
    ) as arquivo_csv:

        escritor = csv.DictWriter(
            arquivo_csv,
            fieldnames=campos
        )

        escritor.writeheader()

        escritor.writerows(
            resultados
        )


if __name__ == "__main__":
    avaliar_modelo_externo()
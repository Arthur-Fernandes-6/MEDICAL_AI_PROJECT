import os
import csv
import argparse
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src.gradcam_efficientnet_cropped import (
    preparar_imagem,
    gerar_gradcam,
    criar_sobreposicao
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

CAMINHO_MODELO_PADRAO = (
    "models/brain_tumor_efficientnet_cropped_best.keras"
)

PASTA_ENTRADA_PADRAO = "gradcam_test_images"

PASTA_RESULTADOS_PADRAO = (
    "results_gradcam_cropped_lote"
)

EXTENSOES_VALIDAS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
)

LIMIAR = 0.5


# ============================================================
# LISTAR IMAGENS
# ============================================================

def listar_imagens(pasta_entrada):
    imagens = []

    for raiz, _, arquivos in os.walk(pasta_entrada):
        for nome_arquivo in arquivos:
            if nome_arquivo.lower().endswith(
                EXTENSOES_VALIDAS
            ):
                caminho_completo = os.path.join(
                    raiz,
                    nome_arquivo
                )

                imagens.append(
                    caminho_completo
                )

    imagens.sort()

    return imagens


# ============================================================
# CRIAR FIGURA INDIVIDUAL
# ============================================================

def salvar_figura_gradcam(
    imagem_rgb,
    mapa_colorido,
    sobreposicao,
    probabilidade,
    caminho_imagem,
    pasta_entrada,
    pasta_resultados
):
    caminho_relativo = os.path.relpath(
        caminho_imagem,
        pasta_entrada
    )

    pasta_relativa = os.path.dirname(
        caminho_relativo
    )

    nome_original = os.path.basename(
        caminho_imagem
    )

    nome_sem_extensao = os.path.splitext(
        nome_original
    )[0]

    pasta_saida = os.path.join(
        pasta_resultados,
        pasta_relativa
    )

    os.makedirs(
        pasta_saida,
        exist_ok=True
    )

    classe_predita = (
        "Tumor"
        if probabilidade >= LIMIAR
        else "Sem tumor"
    )

    figura = plt.figure(
        figsize=(15, 5)
    )

    eixo1 = figura.add_subplot(
        1,
        3,
        1
    )

    eixo1.imshow(
        imagem_rgb
    )

    eixo1.set_title(
        "Imagem após Brain Crop"
    )

    eixo1.axis(
        "off"
    )

    eixo2 = figura.add_subplot(
        1,
        3,
        2
    )

    eixo2.imshow(
        mapa_colorido
    )

    eixo2.set_title(
        "Mapa Grad-CAM"
    )

    eixo2.axis(
        "off"
    )

    eixo3 = figura.add_subplot(
        1,
        3,
        3
    )

    eixo3.imshow(
        sobreposicao
    )

    eixo3.set_title(
        f"Predição: {classe_predita}\n"
        f"P(tumor): {probabilidade:.6f}"
    )

    eixo3.axis(
        "off"
    )

    plt.tight_layout()

    caminho_saida = os.path.join(
        pasta_saida,
        f"{nome_sem_extensao}_gradcam_cropped.png"
    )

    plt.savefig(
        caminho_saida,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close(
        figura
    )

    return caminho_saida, classe_predita


# ============================================================
# SALVAR MAPA E SOBREPOSIÇÃO SEPARADOS
# ============================================================

def salvar_componentes(
    mapa_colorido,
    sobreposicao,
    caminho_imagem,
    pasta_entrada,
    pasta_resultados
):
    caminho_relativo = os.path.relpath(
        caminho_imagem,
        pasta_entrada
    )

    pasta_relativa = os.path.dirname(
        caminho_relativo
    )

    nome_original = os.path.basename(
        caminho_imagem
    )

    nome_sem_extensao = os.path.splitext(
        nome_original
    )[0]

    pasta_saida = os.path.join(
        pasta_resultados,
        pasta_relativa,
        "componentes"
    )

    os.makedirs(
        pasta_saida,
        exist_ok=True
    )

    caminho_mapa = os.path.join(
        pasta_saida,
        f"{nome_sem_extensao}_mapa.png"
    )

    caminho_sobreposicao = os.path.join(
        pasta_saida,
        f"{nome_sem_extensao}_sobreposicao.png"
    )

    cv2.imwrite(
        caminho_mapa,
        cv2.cvtColor(
            mapa_colorido,
            cv2.COLOR_RGB2BGR
        )
    )

    cv2.imwrite(
        caminho_sobreposicao,
        cv2.cvtColor(
            sobreposicao,
            cv2.COLOR_RGB2BGR
        )
    )


# ============================================================
# SALVAR CSV
# ============================================================

def salvar_csv(resultados, pasta_resultados):
    caminho_csv = os.path.join(
        pasta_resultados,
        "resultados_gradcam_cropped.csv"
    )

    with open(
        caminho_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as arquivo_csv:

        escritor = csv.DictWriter(
            arquivo_csv,
            fieldnames=[
                "imagem",
                "probabilidade_tumor",
                "classe_predita",
                "status",
                "arquivo_resultado",
                "erro"
            ]
        )

        escritor.writeheader()
        escritor.writerows(
            resultados
        )

    return caminho_csv


# ============================================================
# EXECUÇÃO EM LOTE
# ============================================================

def gerar_gradcams_em_lote(
    caminho_modelo,
    pasta_entrada,
    pasta_resultados
):
    if not os.path.isfile(
        caminho_modelo
    ):
        raise FileNotFoundError(
            f"Modelo não encontrado: {caminho_modelo}"
        )

    if not os.path.isdir(
        pasta_entrada
    ):
        raise FileNotFoundError(
            f"Pasta de imagens não encontrada: {pasta_entrada}"
        )

    os.makedirs(
        pasta_resultados,
        exist_ok=True
    )

    imagens = listar_imagens(
        pasta_entrada
    )

    if len(imagens) == 0:
        raise RuntimeError(
            "Nenhuma imagem válida foi encontrada."
        )

    print("\n" + "=" * 65)
    print("CARREGANDO MODELO CROPPED")
    print("=" * 65)

    modelo = tf.keras.models.load_model(
        caminho_modelo
    )

    print("Modelo carregado com sucesso!")

    print("\n" + "=" * 65)
    print("GERAÇÃO DE GRAD-CAMS EM LOTE")
    print("=" * 65)

    print(
        f"Imagens encontradas: {len(imagens)}"
    )

    resultados = []

    total_sucesso = 0
    total_erros = 0
    total_tumor = 0
    total_sem_tumor = 0

    for indice, caminho_imagem in enumerate(
        imagens,
        start=1
    ):
        nome_exibicao = os.path.relpath(
            caminho_imagem,
            pasta_entrada
        )

        print(
            f"\n[{indice}/{len(imagens)}] "
            f"Processando: {nome_exibicao}"
        )

        try:
            imagem_rgb, imagem_modelo = preparar_imagem(
                caminho_imagem
            )

            mapa_calor, probabilidade = gerar_gradcam(
                modelo,
                imagem_modelo
            )

            mapa_colorido, sobreposicao = criar_sobreposicao(
                imagem_rgb,
                mapa_calor
            )

            caminho_resultado, classe_predita = (
                salvar_figura_gradcam(
                    imagem_rgb=imagem_rgb,
                    mapa_colorido=mapa_colorido,
                    sobreposicao=sobreposicao,
                    probabilidade=probabilidade,
                    caminho_imagem=caminho_imagem,
                    pasta_entrada=pasta_entrada,
                    pasta_resultados=pasta_resultados
                )
            )

            salvar_componentes(
                mapa_colorido=mapa_colorido,
                sobreposicao=sobreposicao,
                caminho_imagem=caminho_imagem,
                pasta_entrada=pasta_entrada,
                pasta_resultados=pasta_resultados
            )

            if classe_predita == "Tumor":
                total_tumor += 1
            else:
                total_sem_tumor += 1

            total_sucesso += 1

            resultados.append(
                {
                    "imagem": nome_exibicao,
                    "probabilidade_tumor": (
                        f"{probabilidade:.8f}"
                    ),
                    "classe_predita": classe_predita,
                    "status": "OK",
                    "arquivo_resultado": caminho_resultado,
                    "erro": ""
                }
            )

            print(
                f"Predição: {classe_predita}"
            )

            print(
                f"P(tumor): {probabilidade:.6f}"
            )

        except Exception as erro:
            total_erros += 1

            resultados.append(
                {
                    "imagem": nome_exibicao,
                    "probabilidade_tumor": "",
                    "classe_predita": "",
                    "status": "ERRO",
                    "arquivo_resultado": "",
                    "erro": str(erro)
                }
            )

            print(
                f"ERRO: {erro}"
            )

    caminho_csv = salvar_csv(
        resultados,
        pasta_resultados
    )

    print("\n" + "=" * 65)
    print("PROCESSAMENTO FINALIZADO")
    print("=" * 65)

    print(
        f"Total de imagens:       {len(imagens)}"
    )

    print(
        f"Processadas com sucesso: {total_sucesso}"
    )

    print(
        f"Imagens com erro:        {total_erros}"
    )

    print(
        f"Preditas como tumor:     {total_tumor}"
    )

    print(
        f"Preditas sem tumor:      {total_sem_tumor}"
    )

    print(
        f"\nResultados salvos em: {pasta_resultados}"
    )

    print(
        f"CSV salvo em: {caminho_csv}"
    )


# ============================================================
# ARGUMENTOS
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Gera Grad-CAM em lote usando a "
            "EfficientNet treinada com Brain Crop."
        )
    )

    parser.add_argument(
        "--pasta",
        default=PASTA_ENTRADA_PADRAO,
        help=(
            "Pasta que contém as imagens. "
            "Subpastas também serão percorridas."
        )
    )

    parser.add_argument(
        "--modelo",
        default=CAMINHO_MODELO_PADRAO,
        help="Caminho do modelo .keras."
    )

    parser.add_argument(
        "--saida",
        default=PASTA_RESULTADOS_PADRAO,
        help="Pasta onde os resultados serão salvos."
    )

    argumentos = parser.parse_args()

    gerar_gradcams_em_lote(
        caminho_modelo=argumentos.modelo,
        pasta_entrada=argumentos.pasta,
        pasta_resultados=argumentos.saida
    )


if __name__ == "__main__":
    main()
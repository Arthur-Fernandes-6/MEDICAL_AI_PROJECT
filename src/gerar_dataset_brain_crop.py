import os
import cv2
from src.preprocess_brain_crop import preprocessar_brain_crop


PASTA_ORIGEM = "brain_tumor_mri_dataset"
PASTA_DESTINO = "brain_tumor_mri_dataset_cropped"

CONJUNTOS = [
    "Training",
    "Testing"
]

CLASSES = [
    "glioma",
    "meningioma",
    "pituitary",
    "notumor"
]

EXTENSOES_VALIDAS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
)


def gerar_dataset_recortado():
    total_processadas = 0
    total_erros = 0

    for conjunto in CONJUNTOS:
        for nome_classe in CLASSES:

            pasta_entrada = os.path.join(
                PASTA_ORIGEM,
                conjunto,
                nome_classe
            )

            pasta_saida = os.path.join(
                PASTA_DESTINO,
                conjunto,
                nome_classe
            )

            if not os.path.isdir(pasta_entrada):
                print(
                    f"Pasta não encontrada: {pasta_entrada}"
                )
                continue

            os.makedirs(
                pasta_saida,
                exist_ok=True
            )

            arquivos = os.listdir(
                pasta_entrada
            )

            arquivos_validos = [
                arquivo
                for arquivo in arquivos
                if arquivo.lower().endswith(
                    EXTENSOES_VALIDAS
                )
            ]

            print("\n" + "=" * 60)
            print(f"Conjunto: {conjunto}")
            print(f"Classe: {nome_classe}")
            print(
                f"Imagens encontradas: "
                f"{len(arquivos_validos)}"
            )
            print("=" * 60)

            for indice, nome_arquivo in enumerate(
                arquivos_validos,
                start=1
            ):
                caminho_entrada = os.path.join(
                    pasta_entrada,
                    nome_arquivo
                )

                caminho_saida = os.path.join(
                    pasta_saida,
                    nome_arquivo
                )

                try:
                    # normalizar=False:
                    # a imagem será retornada em uint8,
                    # apropriada para salvar no disco.
                    imagem_recortada = preprocessar_brain_crop(
                        caminho_entrada,
                        normalizar=False
                    )

                    if imagem_recortada is None:
                        raise RuntimeError(
                            "O pré-processamento retornou None."
                        )

                    salvou = cv2.imwrite(
                        caminho_saida,
                        imagem_recortada
                    )

                    if not salvou:
                        raise RuntimeError(
                            "O OpenCV não conseguiu salvar."
                        )

                    total_processadas += 1

                    if (
                        indice % 100 == 0
                        or indice == len(arquivos_validos)
                    ):
                        print(
                            f"{indice}/"
                            f"{len(arquivos_validos)} "
                            "processadas"
                        )

                except Exception as erro:
                    total_erros += 1

                    print(
                        f"\nErro em {caminho_entrada}: "
                        f"{erro}"
                    )

    print("\n" + "=" * 60)
    print("DATASET FINALIZADO")
    print("=" * 60)
    print(
        f"Imagens processadas: {total_processadas}"
    )
    print(
        f"Imagens com erro: {total_erros}"
    )
    print(
        f"Dataset salvo em: {PASTA_DESTINO}"
    )


if __name__ == "__main__":
    gerar_dataset_recortado()
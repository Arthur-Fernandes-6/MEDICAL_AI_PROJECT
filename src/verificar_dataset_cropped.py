import os


PASTA_ORIGINAL = "brain_tumor_mri_dataset"
PASTA_CROPPED = "brain_tumor_mri_dataset_cropped"

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

EXTENSOES = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
)


def contar_imagens(pasta):
    if not os.path.isdir(pasta):
        return 0

    return sum(
        1
        for arquivo in os.listdir(pasta)
        if arquivo.lower().endswith(EXTENSOES)
    )


total_original = 0
total_cropped = 0

print("\n===== VERIFICAÇÃO DO DATASET =====\n")

for conjunto in CONJUNTOS:
    print(f"\n{conjunto}")

    for nome_classe in CLASSES:
        caminho_original = os.path.join(
            PASTA_ORIGINAL,
            conjunto,
            nome_classe
        )

        caminho_cropped = os.path.join(
            PASTA_CROPPED,
            conjunto,
            nome_classe
        )

        quantidade_original = contar_imagens(
            caminho_original
        )

        quantidade_cropped = contar_imagens(
            caminho_cropped
        )

        total_original += quantidade_original
        total_cropped += quantidade_cropped

        status = (
            "OK"
            if quantidade_original == quantidade_cropped
            else "DIFERENÇA"
        )

        print(
            f"{nome_classe:12s} | "
            f"Original: {quantidade_original:4d} | "
            f"Cropped: {quantidade_cropped:4d} | "
            f"{status}"
        )

print("\n" + "=" * 60)
print(f"Total original: {total_original}")
print(f"Total cropped:  {total_cropped}")
print("=" * 60)
# 🧠 NeuroScan AI

Sistema de Inteligência Artificial para classificação de tumores cerebrais em imagens de Ressonância Magnética (MRI) utilizando Redes Neurais Convolucionais (CNN).

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-green)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

# 📖 Sobre o Projeto

O **NeuroScan AI** é uma aplicação desenvolvida com o objetivo de realizar a classificação automática de exames de ressonância magnética cerebral, identificando a presença ou ausência de tumores utilizando técnicas de Inteligência Artificial.

O projeto foi desenvolvido como parte do curso de Inteligência Artificial da **OxeTech**, utilizando uma Rede Neural Convolucional (CNN) treinada em imagens reais de MRI.

Além da classificação, o sistema utiliza **LIME (Local Interpretable Model-Agnostic Explanations)** para fornecer explicações visuais das regiões da imagem que mais influenciaram a decisão da rede neural.

---

# ✨ Funcionalidades

- Classificação de tumores cerebrais
- Upload de imagens MRI
- Teste rápido utilizando imagens do dataset
- Exibição da probabilidade da classificação
- Confiança da previsão
- Explicabilidade utilizando LIME
- API REST desenvolvida em FastAPI
- Interface Web moderna
- Containerização com Docker

---

# 🖥️ Interface

## Tela Principal

*(adicione um print da aplicação aqui)*

---

## Resultado da Classificação

*(adicione outro print mostrando o resultado da IA)*

---

## Explicabilidade com LIME

*(adicione um print do LIME funcionando)*

---

# 🧠 Tecnologias Utilizadas

## Backend

- Python
- FastAPI
- TensorFlow/Keras
- NumPy
- Pillow
- LIME
- Scikit-image

## Frontend

- HTML5
- CSS3
- JavaScript

## Ferramentas

- Docker
- Git
- GitHub
- Git LFS

---

# 📂 Estrutura do Projeto

```text
BRAIN_TUMOR_DETECTION_AI
│
├── api/
│   ├── main.py
│   └── lime_explainer.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── models/
│   └── brain_tumor_cnn_v2.keras
│
├── datasets/
│
├── notebooks/
│
├── src/
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# ⚙️ Como executar

## Clone o projeto

```bash
git clone https://github.com/SEU-USUARIO/BRAIN_TUMOR_DETECTION_AI.git
```

Entre na pasta

```bash
cd BRAIN_TUMOR_DETECTION_AI
```

Instale as dependências

```bash
pip install -r requirements.txt
```

Execute

```bash
python -m uvicorn api.main:app --reload
```

Acesse

```
http://127.0.0.1:8000
```

---

# 🐳 Executando com Docker

Construir a imagem

```bash
docker build -t brain-tumor-ai .
```

Executar

```bash
docker run --rm -p 8000:10000 brain-tumor-ai
```

---

# 📊 Dataset

O modelo foi treinado utilizando um conjunto de imagens reais de Ressonância Magnética contendo exames classificados em duas categorias:

- Tumor
- Sem Tumor

As imagens foram utilizadas exclusivamente para fins educacionais e de pesquisa.

---

# 🔍 Explicabilidade (LIME)

O projeto utiliza o algoritmo **LIME (Local Interpretable Model-Agnostic Explanations)** para gerar explicações visuais da decisão da CNN.

As regiões destacadas representam as áreas da imagem que mais contribuíram para a classificação realizada pelo modelo.

---

# ⚠️ Aviso

Este projeto possui finalidade exclusivamente acadêmica e educacional.

Os resultados apresentados **não substituem avaliação médica profissional** e **não devem ser utilizados para diagnóstico clínico**.

---

# 👨‍💻 Autor

Arthur Fernandes
Fábio luis

Curso de Sistemas de Informação

Projeto desenvolvido durante o curso de Inteligência Artificial da OxeTech.

---

# 📄 Licença

Este projeto está licenciado sob a licença MIT.

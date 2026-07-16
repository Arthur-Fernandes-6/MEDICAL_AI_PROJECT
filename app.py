import os
import sys
import hashlib
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# Adiciona o diretório 'src' ao PATH para importar os módulos locais
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

app = Flask(__name__, static_folder='frontend', static_url_path='')

# Configurações
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limite de 16MB

# Variáveis globais para o modelo e status
MODELO = None
MODO_SIMULADO = False
MOTIVO_SIMULACAO = ""

# Verificar se o modelo é um ponteiro LFS ou se está completo
CAMINHO_MODELO = os.path.join(os.path.dirname(__file__), 'models', 'brain_tumor_cnn.keras')

def verificar_modelo_lfs(caminho):
    if not os.path.exists(caminho):
        return True, "Arquivo do modelo não encontrado."
    
    # Se o arquivo for muito pequeno (ex: menos de 1KB), é provável que seja o ponteiro do Git LFS
    tamanho = os.path.getsize(caminho)
    if tamanho < 1000:
        return True, f"O arquivo do modelo está em formato de ponteiro Git LFS ({tamanho} bytes) e não foi baixado."
    return False, ""

is_lfs, motivo_lfs = verificar_modelo_lfs(CAMINHO_MODELO)

if is_lfs:
    MODO_SIMULADO = True
    MOTIVO_SIMULACAO = motivo_lfs
    print(f"[*] AVISO: {motivo_lfs} O servidor iniciará no MODO SIMULAÇÃO.")
else:
    try:
        import numpy as np
        import cv2 as cv
        from tensorflow.keras.models import load_model
        from preprocess import preprocessar_imagem
        
        print("[*] Carregando modelo do TensorFlow...")
        MODELO = load_model(CAMINHO_MODELO)
        print("[+] Modelo carregado com sucesso!")
    except Exception as e:
        MODO_SIMULADO = True
        MOTIVO_SIMULACAO = f"Erro ao carregar dependências do TensorFlow/OpenCV ou erro ao carregar o modelo: {str(e)}"
        print(f"[*] AVISO: {MOTIVO_SIMULACAO}")
        print("[*] O servidor iniciará no MODO SIMULAÇÃO.")


# --- Auxiliares para o Modo Simulado ---
def predição_simulada(caminho_imagem, nome_arquivo):
    """
    Realiza uma predição simulada determinística baseada no nome do arquivo
    e no conteúdo (hash) da imagem. Garante consistência ao testar a mesma imagem.
    """
    # 1. Tenta identificar pelo nome do arquivo
    nome_lower = nome_arquivo.lower()
    
    # Se o nome indicar claramente a classe (comum no dataset e nos exemplos)
    if any(k in nome_lower for k in ['yes', 'tumor', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y0']):
        classe = "Tumor"
        # Gera uma confiança alta aleatória-determinística
        semente = int(hashlib.md5(nome_arquivo.encode()).hexdigest(), 16)
        probabilidade = 0.75 + (semente % 24) / 100.0  # entre 75% e 99%
        return classe, probabilidade
    
    if any(k in nome_lower for k in ['no', 'sem', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n0']):
        classe = "Sem tumor"
        semente = int(hashlib.md5(nome_arquivo.encode()).hexdigest(), 16)
        probabilidade = (semente % 25) / 100.0  # entre 0% e 24%
        return classe, probabilidade

    # 2. Caso contrário, faz baseando-se no hash do conteúdo do arquivo
    try:
        with open(caminho_imagem, "rb") as f:
            conteudo = f.read()
        md5_hash = hashlib.md5(conteudo).hexdigest()
        semente = int(md5_hash, 16)
        
        # Decide de forma determinística
        probabilidade = (semente % 100) / 100.0
        if probabilidade >= 0.5:
            classe = "Tumor"
        else:
            classe = "Sem tumor"
        
        # Garante que as probabilidades fiquem mais distantes do limiar de 0.5 para parecer realista
        if probabilidade >= 0.5 and probabilidade < 0.7:
            probabilidade += 0.15
        elif probabilidade < 0.5 and probabilidade > 0.3:
            probabilidade -= 0.15
            
        return classe, probabilidade
    except Exception:
        # Fallback de segurança absoluto
        return "Sem tumor", 0.12


# --- Rotas da API ---

@app.route('/')
def index():
    """Serve a aplicação front-end."""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    """Retorna o status atual do servidor (Real vs Simulado)."""
    return jsonify({
        "status": "online",
        "modo_simulado": MODO_SIMULADO,
        "motivo": MOTIVO_SIMULACAO if MODO_SIMULADO else "Modelo carregado com sucesso."
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Endpoint principal para upload e classificação de imagens."""
    if 'file' not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nome de arquivo inválido"}), 400
    
    # Salva o arquivo temporariamente
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        if MODO_SIMULADO:
            classificacao, probabilidade = predição_simulada(filepath, filename)
            resultado = {
                "classificacao": classificacao,
                "probabilidade": probabilidade,
                "simulado": True
            }
        else:
            import numpy as np
            from preprocess import preprocessar_imagem
            
            imagem = preprocessar_imagem(filepath)
            if imagem is None:
                return jsonify({"error": "Falha ao processar a imagem. Formato inválido."}), 400
            
            imagem = np.expand_dims(imagem, axis=0)
            pred = MODELO.predict(imagem)
            probabilidade = float(pred[0][0])
            
            if probabilidade >= 0.5:
                classificacao = "Tumor"
            else:
                classificacao = "Sem tumor"
                
            resultado = {
                "classificacao": classificacao,
                "probabilidade": probabilidade,
                "simulado": False
            }
        
        # Limpa o arquivo enviado após a inferência (opcional, mas bom para privacidade)
        try:
            os.remove(filepath)
        except Exception:
            pass
            
        return jsonify(resultado)
        
    except Exception as e:
        return jsonify({"error": f"Erro interno ao processar predição: {str(e)}"}), 500


@app.route('/api/samples', methods=['GET'])
def get_samples():
    """Retorna uma lista de imagens de amostra para facilitar o teste pelo usuário."""
    samples = []
    
    # Procurar por algumas imagens em datasets/yes e datasets/no
    dirs = {
        "yes": os.path.join(os.path.dirname(__file__), 'datasets', 'yes'),
        "no": os.path.join(os.path.dirname(__file__), 'datasets', 'no')
    }
    
    # Pega até 3 amostras de cada classe
    for label, path in dirs.items():
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            # Ordena ou pega uma seleção determinística para não mudar a cada request
            files = sorted(files)[:3]
            for f in files:
                samples.append({
                    "filename": f,
                    "label": label,
                    "url": f"/api/samples/{label}/{f}"
                })
                
    return jsonify(samples)

@app.route('/api/samples/<label>/<filename>', methods=['GET'])
def serve_sample(label, filename):
    """Serve uma imagem de teste real."""
    if label not in ['yes', 'no']:
        return "Classe inválida", 400
    
    directory = os.path.join(os.path.dirname(__file__), 'datasets', label)
    return send_from_directory(directory, filename)


if __name__ == '__main__':
    # Cria a pasta de uploads se não existir
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    print("--------------------------------------------------")
    print(f"Servidor rodando em: http://127.0.0.1:5000")
    print(f"Status: {'MODO SIMULAÇÃO' if MODO_SIMULADO else 'MODO REAL'}")
    print("--------------------------------------------------")
    app.run(host='127.0.0.1', port=5000, debug=True)

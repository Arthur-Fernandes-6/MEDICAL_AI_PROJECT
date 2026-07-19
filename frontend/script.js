document.addEventListener("DOMContentLoaded", () => {
    // URL base da API
    const API_URL = "http://127.0.0.1:8000"; // Caminho relativo, pois o backend serve o frontend estático

    // Elementos DOM - Status
    const statusBadge = document.getElementById("status-badge");
    const statusText = document.getElementById("status-text");

    // Elementos DOM - Upload & Drag & Drop
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const browseBtn = document.getElementById("browse-btn");
    const samplesGallery = document.getElementById("samples-gallery");

    // Elementos DOM - Estados dos Resultados
    const resultsPanel = document.getElementById("results-panel");
    const stateEmpty = document.getElementById("result-state-empty");
    const stateLoading = document.getElementById("result-state-loading");
    const stateSuccess = document.getElementById("result-state-success");
    const simulationWarning = document.getElementById("simulation-warning");

    // Elementos DOM - Laudo Médico
    const previewImage = document.getElementById("preview-image");
    const classificationBadge = document.getElementById("classification-badge");
    const classificationText = document.getElementById("classification-text");
    const confidencePercentage = document.getElementById("confidence-percentage");
    const confidenceBarFill = document.getElementById("confidence-bar-fill");
    
    const reportExamId = document.getElementById("report-exam-id");
    const reportTimestamp = document.getElementById("report-timestamp");
    const reportTextDesc = document.getElementById("report-text-desc");
    const recommendationsList = document.getElementById("recommendations-list");
    
    const printReportBtn = document.getElementById("print-report-btn");
    const resetAnalysisBtn = document.getElementById("reset-analysis-btn");

    // Elementos DOM - Gráfico de Performance
    const chartToggleBtns = document.querySelectorAll(".chart-toggle-btn");
    let performanceChart = null;

    // Dados das curvas de aprendizado
    const epochs = [1, 2, 3, 4, 5, 6];

const trainingData = {
    accuracy: {
        train: [
            0.8728,
            0.9625,
            0.9835,
            0.9891,
            0.9908,
            0.9967
        ],

        val: [
            0.9250,
            0.9732,
            0.9723,
            0.9777,
            0.9741,
            0.9768
        ]
    },

    loss: {
        train: [
            0.3794,
            0.1300,
            0.0559,
            0.0334,
            0.0242,
            0.0092
        ],

        val: [
            0.2396,
            0.1026,
            0.0901,
            0.1083,
            0.1042,
            0.1203
        ]
    }
};

    // ==========================================================================
    // 1. Inicialização e Status da Conexão
    // ==========================================================================
    async function verificarStatusServidor() {
        try {
            const res = await fetch(`${API_URL}/api/status`);
            if (!res.ok) throw new Error("Erro de resposta do servidor");
            
            const data = await res.json();
            
            statusBadge.className = "status-badge"; // Reseta classes
            if (data.modo_simulado) {
                statusBadge.classList.add("online-simulated");
                statusText.innerText = "Modo Simulação Ativo";
                statusBadge.title = `O modelo real não está disponível: ${data.motivo}. Rodando em modo simulado.`;
            } else {
                statusBadge.classList.add("online-real");
                statusText.innerText = "Servidor Online (Modelo Real)";
            }
        } catch (err) {
            console.error("Erro ao conectar com o back-end:", err);
            statusBadge.className = "status-badge offline";
            statusText.innerText = "Servidor Offline";
        }
    }

    // ==========================================================================
    // 2. Carregar Galeria de Imagens de Teste
    // ==========================================================================
    async function carregarGaleriaExemplos() {
        try {
            const res = await fetch(`${API_URL}/api/samples`);
            if (!res.ok) throw new Error();
            
            const samples = await res.json();
            samplesGallery.innerHTML = ""; // Limpa skeletons

            if (samples.length === 0) {
                samplesGallery.innerHTML = "<p class='text-dark' style='grid-column: 1/-1;'>Nenhuma amostra encontrada.</p>";
                return;
            }

            samples.forEach(sample => {
                const card = document.createElement("div");
                card.className = "sample-card";
                
                const labelText = sample.label === "yes" ? "Tumor" : "Sem Tumor";
                const labelClass = sample.label === "yes" ? "yes" : "no";

                card.innerHTML = `
                    <div class="sample-img-container">
                        <img src="${sample.url}" alt="${sample.filename}" loading="lazy">
                    </div>
                    <div class="sample-info">
                        <span class="sample-name" title="${sample.filename}">${sample.filename}</span>
                        <span class="sample-badge ${labelClass}">${labelText}</span>
                    </div>
                `;

                card.addEventListener("click", () => analisarImagemAmostra(sample.url, sample.filename));
                samplesGallery.appendChild(card);
            });
        } catch (err) {
            console.error("Erro ao carregar imagens de exemplo:", err);
            samplesGallery.innerHTML = "<p class='text-coral' style='grid-column: 1/-1;'><i class='fa-solid fa-circle-exclamation'></i> Erro ao carregar exemplos.</p>";
        }
    }

    // ==========================================================================
    // 3. Processamento de Arquivos e Upload
    // ==========================================================================
    
    // Clique no botão de seleção
    browseBtn.addEventListener("click", () => fileInput.click());
    
    // Mudança no Input
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            processarArquivo(e.target.files[0]);
        }
    });

    // Eventos de Drag & Drop
    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add("drag-over");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove("drag-over");
        }, false);
    });

    dropZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            processarArquivo(files[0]);
        }
    });

    function processarArquivo(file) {
        if (!file.type.startsWith("image/")) {
            alert("Por favor, selecione apenas arquivos de imagem.");
            return;
        }
        classificarMRI(file);
    }

    // Fluxo de análise a partir de uma amostra da galeria
    async function analisarImagemAmostra(url, filename) {
        exibirEstado("loading");
        try {
            const res = await fetch(url);
            const blob = await res.blob();
            const file = new File([blob], filename, { type: blob.type });
            classificarMRI(file);
        } catch (err) {
            console.error("Erro ao processar amostra:", err);
            exibirEstado("empty");
            alert("Erro ao carregar imagem de exemplo para análise.");
        }
    }

    // ==========================================================================
    // 4. Classificação de Imagem & Laudo Médico
    // ==========================================================================
    async function classificarMRI(file) {
        exibirEstado("loading");

        // Cria preview local imediato da imagem
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
        };
        reader.readAsDataURL(file);

        // Prepara dados de envio
        const formData = new FormData();
        formData.append("arquivo", file);
        const startTime = Date.now();

        try {
            const response = await fetch(`${API_URL}/api/predict`, {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || "Erro na análise.");
            }

            const resultado = await response.json();
            
            // Simula um delay visual de escaneamento de pelo menos 1.5s para melhor experiência clínica (WOW)
            const elapsedTime = Date.now() - startTime;
            const delay = Math.max(1500 - elapsedTime, 0);

            setTimeout(() => {
                renderizarResultados(resultado, file.name);
            }, delay);

        } catch (err) {
            console.error("Erro ao classificar exame:", err);
            exibirEstado("empty");
            alert(`Falha na análise: ${err.message}`);
        }
    }

    function renderizarResultados(res, filename) {
        exibirEstado("success");

        // Status de Simulação
        if (res.simulado) {
            simulationWarning.classList.remove("hide");
        } else {
            simulationWarning.classList.add("hide");
        }

        // Configuração de Badges e Cores
        classificationBadge.className = "classification-badge";
        const hasTumor = res.classificacao === "Tumor";

        // Confiança formatada
        // Se Tumor: probabilidade * 100
        // Se Sem Tumor: (1 - probabilidade) * 100
        const confRaw = hasTumor ? res.probabilidade : (1 - res.probabilidade);
        const confPercent = (confRaw * 100).toFixed(2);
        
        confidencePercentage.innerText = `${confPercent}%`;
        confidenceBarFill.style.width = "0%"; // Reseta para animar de novo

        if (hasTumor) {
            classificationBadge.classList.add("tumor");

            classificationText.innerHTML = `
                <i class="fa-solid fa-triangle-exclamation"></i>
                Padrão associado à classe Tumor
            `;

            reportTextDesc.innerHTML = `
                O modelo de visão computacional classificou a imagem
                <strong>${filename}</strong> como pertencente à classe
                <strong>Tumor</strong>, com confiança de
                <strong>${confPercent}%</strong>.

                Este resultado é experimental e representa apenas a classificação
                produzida pelo modelo a partir dos padrões aprendidos no dataset.
                Ele não constitui diagnóstico médico.
            `;

            recommendationsList.innerHTML = `
                <li>
                    <strong>Interpretação profissional:</strong>
                    A imagem deve ser avaliada por um médico especialista.
                </li>

                <li>
                    <strong>Limitação do modelo:</strong>
                    O sistema não identifica localização, tamanho, estágio ou tipo específico de tumor.
                </li>

                <li>
                    <strong>Finalidade acadêmica:</strong>
                    O resultado não deve ser usado isoladamente para decisões clínicas.
                </li>
            `;
        } else {
            classificationBadge.classList.add("sem-tumor");

            classificationText.innerHTML = `
                <i class="fa-solid fa-circle-check"></i>
                Padrão associado à classe Sem Tumor
            `;

            reportTextDesc.innerHTML = `
                O modelo de visão computacional classificou a imagem
                <strong>${filename}</strong> como pertencente à classe
                <strong>Sem Tumor</strong>, com confiança de
                <strong>${confPercent}%</strong>.

                Isso significa apenas que os padrões visuais observados foram mais
                semelhantes às imagens da classe sem tumor utilizadas no treinamento.
                O resultado não exclui doenças ou outras alterações médicas.
            `;

            recommendationsList.innerHTML = `
                <li>
                    <strong>Avaliação especializada:</strong>
                    O resultado deve ser interpretado por um profissional de saúde.
                </li>

                <li>
                    <strong>Possibilidade de erro:</strong>
                    O modelo pode produzir falsos negativos e falsos positivos.
                </li>

                <li>
                    <strong>Finalidade acadêmica:</strong>
                    Esta ferramenta não substitui exames, laudos ou avaliação médica.
                </li>
            `;
}

        // Animação suave de enchimento da barra
        setTimeout(() => {
            confidenceBarFill.style.width = `${confPercent}%`;
        }, 100);

        // Preenche metadados do laudo
        const randomID = Math.floor(10000 + Math.random() * 90000);
        reportExamId.innerText = `#NS-${randomID}-MRI`;
        
        const agora = new Date();
        const dataFormatada = agora.toLocaleDateString("pt-BR") + " - " + agora.toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit" });
        reportTimestamp.innerText = dataFormatada;
    }

    function exibirEstado(state) {
        stateEmpty.classList.remove("active");
        stateLoading.classList.remove("active");
        stateSuccess.classList.remove("active");

        if (state === "empty") stateEmpty.classList.add("active");
        if (state === "loading") stateLoading.classList.add("active");
        if (state === "success") stateSuccess.classList.add("active");
    }

    // Botões de Ação do Laudo
    resetAnalysisBtn.addEventListener("click", () => {
        fileInput.value = "";
        exibirEstado("empty");
    });

    printReportBtn.addEventListener("click", () => {
        window.print();
    });

    // ==========================================================================
    // 5. Gráfico de Performance (Chart.js)
    // ==========================================================================
    function inicializarGraficoPerformance(metrica = "accuracy") {
        const ctx = document.getElementById("performance-chart").getContext("2d");
        
        // Se o gráfico já existir, destrói para poder recriar
        if (performanceChart) {
            performanceChart.destroy();
        }

        const isAcc = metrica === "accuracy";
        const labelText = isAcc ? "Acurácia" : "Perda (Loss)";
        const trainColor = isAcc ? "#06b6d4" : "#8b5cf6";
        const valColor = isAcc ? "#10b981" : "#ef4444";

        performanceChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: epochs,
                datasets: [
                    {
                        label: `${labelText} - Treino`,
                        data: trainingData[metrica].train,
                        borderColor: trainColor,
                        backgroundColor: `${trainColor}1a`, // transparência
                        borderWidth: 3,
                        pointRadius: 4,
                        pointBackgroundColor: trainColor,
                        fill: true,
                        tension: 0.3
                    },
                    {
                        label: `${labelText} - Validação`,
                        data: trainingData[metrica].val,
                        borderColor: valColor,
                        backgroundColor: "transparent",
                        borderWidth: 3,
                        pointRadius: 4,
                        pointBackgroundColor: valColor,
                        borderDash: [5, 5],
                        tension: 0.3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: "top",
                        labels: {
                            color: "#94a3b8",
                            font: {
                                family: "Plus Jakarta Sans",
                                weight: 600
                            }
                        }
                    },
                    tooltip: {
                        mode: "index",
                        intersect: false
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: "rgba(255, 255, 255, 0.05)"
                        },
                        ticks: {
                            color: "#94a3b8",
                            font: {
                                family: "Plus Jakarta Sans"
                            }
                        },
                        title: {
                            display: true,
                            text: "Época",
                            color: "#94a3b8",
                            font: {
                                family: "Plus Jakarta Sans",
                                weight: 600
                            }
                        }
                    },
                    y: {
                        grid: {
                            color: "rgba(255, 255, 255, 0.05)"
                        },
                        ticks: {
                            color: "#94a3b8",
                            font: {
                                family: "Plus Jakarta Sans"
                            }
                        },
                        title: {
                            display: true,
                            text: labelText,
                            color: "#94a3b8",
                            font: {
                                family: "Plus Jakarta Sans",
                                weight: 600
                            }
                        }
                    }
                }
            }
        });
    }

    // Toggle de Métricas do Gráfico
    chartToggleBtns.forEach(btn => {
        btn.addEventListener("click", (e) => {
            chartToggleBtns.forEach(b => b.classList.remove("active"));
            e.target.classList.add("active");
            
            const metrica = e.target.dataset.chart;
            inicializarGraficoPerformance(metrica);
        });
    });

    // ==========================================================================
    // Inicialização da Página
    // ==========================================================================
    verificarStatusServidor();
    carregarGaleriaExemplos();
    inicializarGraficoPerformance("accuracy");
});

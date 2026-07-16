document.addEventListener("DOMContentLoaded", () => {
    // URL base da API
    const API_URL = ""; // Caminho relativo, pois o backend serve o frontend estático

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
    const epochs = Array.from({ length: 15 }, (_, i) => i + 1); // 15 épocas exibidas
    const trainingData = {
        accuracy: {
            train: [0.68, 0.75, 0.81, 0.84, 0.87, 0.89, 0.91, 0.93, 0.94, 0.95, 0.96, 0.965, 0.97, 0.975, 0.98],
            val: [0.70, 0.73, 0.78, 0.82, 0.85, 0.86, 0.89, 0.90, 0.91, 0.92, 0.93, 0.935, 0.94, 0.945, 0.9512]
        },
        loss: {
            train: [0.65, 0.54, 0.45, 0.38, 0.32, 0.28, 0.24, 0.20, 0.18, 0.15, 0.13, 0.11, 0.10, 0.09, 0.08],
            val: [0.61, 0.56, 0.48, 0.41, 0.35, 0.32, 0.28, 0.25, 0.22, 0.20, 0.18, 0.17, 0.16, 0.155, 0.1482]
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
        formData.append("file", file);

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
            classificationText.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Tumor Detectado`;
            
            // Relatório de Tumor
            reportTextDesc.innerHTML = `O algoritmo de visão computacional identificou anormalidade estrutural de alta densidade no exame de ressonância magnética (MRI) <strong>${filename}</strong>. A assinatura de contraste e bordas é compatível com tecidos tumorais típicos, apresentando um nível de confiança estatística de <strong>${confPercent}%</strong>.`;
            
            recommendationsList.innerHTML = `
                <li><strong>Encaminhamento prioritário:</strong> Direcionar o paciente imediatamente a um Neurologista ou Neurocirurgião.</li>
                <li><strong>Exame complementar:</strong> Recomenda-se realizar ressonância magnética cerebral com contraste de gadolínio para melhor delimitação anatômica tridimensional.</li>
                <li><strong>Correlação de Sintomas:</strong> Avaliar clinicamente o paciente para correlacionar o achado de imagem com possíveis cefaleias persistentes, alterações de visão ou déficits neurológicos focais.</li>
            `;
        } else {
            classificationBadge.classList.add("sem-tumor");
            classificationText.innerHTML = `<i class="fa-solid fa-circle-check"></i> Sem Tumor Detectado`;

            // Relatório de Sem Tumor
            reportTextDesc.innerHTML = `O algoritmo de visão computacional analisou o exame de ressonância magnética (MRI) <strong>${filename}</strong> e não detectou padrões sugestivos de lesões expansivas ou tecidos tumorais primários ou secundários nas imagens fornecidas, atingindo uma confiança de <strong>${confPercent}%</strong> para exame normal.`;
            
            recommendationsList.innerHTML = `
                <li><strong>Acompanhamento preventivo:</strong> Manter a rotina padrão de cuidados clínicos de acordo com a idade e o histórico familiar do paciente.</li>
                <li><strong>Persistência de Sintomas:</strong> Caso persistam sintomas neurológicos (mesmo com MRI normal), aconselha-se uma consulta especializada para investigar outras patologias não associadas a massas encefálicas.</li>
                <li><strong>Arquivamento Seguro:</strong> Salvar este laudo preliminar no Prontuário Eletrônico do Paciente (PEP) para registros de baseline.</li>
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

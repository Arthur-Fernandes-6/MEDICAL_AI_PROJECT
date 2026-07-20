FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/models && \
    curl -L \
    "https://huggingface.co/arthurfrp123/neuroscan-brain-tumor-cnn/resolve/main/brain_tumor_cnn_v2.keras?download=true" \
    -o /app/models/brain_tumor_cnn_v2.keras

EXPOSE 8080

CMD ["sh", "-c", "python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
# Aplikasi web deteksi prompt injection (FastAPI + DeBERTa-v3-base biner)
# Build dari folder induk `skripsi/` agar folder model ikut ter-copy:
#   docker build -t pi-detector -f webapp/Dockerfile .
#   docker run -p 8000:8000 pi-detector
FROM python:3.10-slim

WORKDIR /app

# Dependensi
COPY webapp/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kode aplikasi + model biner
COPY webapp/ ./webapp/
COPY best_model_biner/ ./best_model_biner/

ENV MODEL_PATH=/app/best_model_biner
EXPOSE 8000
WORKDIR /app/webapp

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

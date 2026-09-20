# HF Spaces (Docker SDK) — Deteksi Prompt Injection (DeBERTa-v3-base biner)
FROM python:3.10-slim

WORKDIR /app

# Cache HF ke lokasi yang bisa ditulis di dalam container Space
ENV HF_HOME=/tmp/hf \
    MODEL_PATH=/app/best_model_biner \
    PYTHONUNBUFFERED=1

# torch CPU-only agar image jauh lebih ringan & build lebih cepat
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Kode aplikasi + model (best_model_biner/ ikut ter-copy dari repo Space)
COPY . .

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

FROM python:3.13-slim
WORKDIR /app
RUN useradd --create-home --uid 1000 user
COPY requirements-spaces.txt .
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-spaces.txt
COPY --chown=user app.py common.py inference.py serve_spaces.py ./
USER user
ENV HOME=/home/user HF_HOME=/home/user/.cache/huggingface \
    HOST=0.0.0.0 PORT=7860 NOONCHI_MODEL=koelectra \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["python", "serve_spaces.py"]

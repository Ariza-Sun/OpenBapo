FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENBAPO_DATA_DIR=/data \
    OPENBAPO_DEVICE=cuda \
    HF_HOME=/cache/hf \
    TORCH_HOME=/cache/torch \
    XDG_CACHE_HOME=/cache \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git build-essential libcairo2 libsndfile1 fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir torch==2.14.1 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu126 \
    && python deploy/install.py \
    && useradd --uid 10001 --create-home openbapo \
    && mkdir -p /data /cache && chown openbapo:openbapo /data /cache \
    && chmod +x /app/deploy/entrypoint.sh
USER openbapo
ENTRYPOINT ["/app/deploy/entrypoint.sh"]
CMD ["openbapo-bot"]

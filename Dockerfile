FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime@sha256:c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2

ARG LTX_CODE_REVISION=9377758131b1ffde4b7f766804590a6617bf2ab9

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_OFFLINE=1 \
    DRAKO_ALLOWED_STORAGE_HOSTS=a76220a52aaf357ce8909685181757af.r2.cloudflarestorage.com

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Lightricks/LTX-2.git /opt/ltx \
    && cd /opt/ltx \
    && git checkout "$LTX_CODE_REVISION" \
    && python -m pip install --upgrade pip \
    && python -m pip install \
        -e packages/ltx-core \
        -e packages/ltx-pipelines

WORKDIR /app
COPY requirements.txt download_models.py ./
RUN python -m pip install -r requirements.txt \
    && HF_HUB_OFFLINE=0 python download_models.py \
    && python -m pip install \
        transformers==4.57.6 \
        huggingface-hub==0.36.0 \
    && rm download_models.py

COPY handler.py contract.py ./

CMD ["python", "-u", "handler.py"]

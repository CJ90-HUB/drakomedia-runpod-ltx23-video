FROM ghcr.io/cj90-hub/drakomedia-ltx23-video-public:video-e057e1364fb353aa10c7ac653625cc245b12f581@sha256:bec5ee9f0de143ff603c299a4a0cfbf0805a91dfe5fd9c5aadfa1ffed86be53d

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_OFFLINE=1 \
    DRAKO_ALLOWED_STORAGE_HOSTS=a76220a52aaf357ce8909685181757af.r2.cloudflarestorage.com

WORKDIR /app
COPY handler.py video_contract.py video_only_distilled.py ./

CMD ["python", "-u", "handler.py"]

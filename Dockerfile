FROM ghcr.io/cj90-hub/drakomedia-ltx23-video-public@sha256:05506e48ed5d3602bbcb02f3e0f8d0b566082b29724b1585c3ff602d0f596ed7

WORKDIR /app
COPY handler.py contract.py ./

CMD ["python", "-u", "handler.py"]

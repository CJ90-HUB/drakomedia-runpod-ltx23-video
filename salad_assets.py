from __future__ import annotations

import hashlib
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


MAX_ASSET_BYTES = 40_000_000_000
MAX_TOTAL_BYTES = 80_000_000_000
ALLOWED_ROOTS = {"gemma", "ltx"}
_asset_lock = threading.Lock()
_session = requests.Session()


@dataclass(frozen=True)
class ModelAsset:
    path: str
    url: str
    sha256: str
    size_bytes: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_asset(value: Any) -> ModelAsset:
    if not isinstance(value, dict):
        raise ValueError("Cada modelo debe describirse como un objeto.")
    relative = str(value.get("path", "")).strip().replace("\\", "/")
    parts = relative.split("/")
    if (
        len(parts) < 2
        or parts[0] not in ALLOWED_ROOTS
        or any(part in {"", ".", ".."} for part in parts)
        or not re.fullmatch(r"[A-Za-z0-9._/-]+", relative)
    ):
        raise ValueError("La ruta de un modelo no es válida.")
    url = str(value.get("url", "")).strip()
    if not url.startswith("https://"):
        raise ValueError("Los modelos deben descargarse mediante HTTPS.")
    sha256 = str(value.get("sha256", "")).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("Falta la huella SHA-256 de un modelo.")
    size = int(value.get("size_bytes", 0))
    if size <= 0 or size > MAX_ASSET_BYTES:
        raise ValueError("El tamaño declarado de un modelo no es válido.")
    return ModelAsset(relative, url, sha256, size)


def _download(asset: ModelAsset, root: Path) -> None:
    destination = (root / asset.path).resolve()
    if root.resolve() not in destination.parents:
        raise ValueError("La ruta del modelo sale de la caché autorizada.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        destination.is_file()
        and destination.stat().st_size == asset.size_bytes
        and _sha256(destination) == asset.sha256
    ):
        return

    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.is_file() else 0
    if offset > asset.size_bytes:
        partial.unlink()
        offset = 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with _session.get(
        asset.url,
        headers=headers,
        stream=True,
        timeout=(20, 900),
        allow_redirects=False,
    ) as response:
        if offset and response.status_code != 206:
            partial.unlink(missing_ok=True)
            offset = 0
            response.close()
            return _download(asset, root)
        response.raise_for_status()
        mode = "ab" if offset else "wb"
        written = offset
        with partial.open(mode) as output:
            for chunk in response.iter_content(8 * 1024 * 1024):
                if not chunk:
                    continue
                written += len(chunk)
                if written > asset.size_bytes:
                    raise ValueError("Un modelo descargado supera su tamaño declarado.")
                output.write(chunk)
    if partial.stat().st_size != asset.size_bytes:
        raise ValueError("La descarga de un modelo quedó incompleta.")
    if _sha256(partial) != asset.sha256:
        partial.unlink(missing_ok=True)
        raise ValueError("La huella de un modelo descargado no coincide.")
    os.replace(partial, destination)


def ensure_model_assets(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("Falta el manifiesto firmado de modelos de R2.")
    raw_assets = manifest.get("assets")
    if not isinstance(raw_assets, list) or not raw_assets:
        raise ValueError("El manifiesto de modelos está vacío.")
    assets = tuple(_parse_asset(value) for value in raw_assets)
    total = sum(asset.size_bytes for asset in assets)
    if total > MAX_TOTAL_BYTES:
        raise ValueError("El manifiesto de modelos supera el límite de seguridad.")

    root = Path(os.environ.get("DRAKO_MODEL_CACHE", "/workspace/model-cache"))
    root.mkdir(parents=True, exist_ok=True)
    with _asset_lock:
        workers = min(4, len(assets))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda asset: _download(asset, root), assets))

    checkpoint = root / "ltx" / "ltx-2.3-22b-distilled-fp8.safetensors"
    gemma = root / "gemma"
    if not checkpoint.is_file() or not (gemma / "config.json").is_file():
        raise ValueError("La caché no contiene todos los modelos necesarios.")
    os.environ["LTX_DISTILLED_MODEL"] = str(checkpoint)
    os.environ["LTX_GEMMA_ROOT"] = str(gemma)
    return {
        "version": str(manifest.get("version", "")).strip(),
        "asset_count": len(assets),
        "total_bytes": total,
    }

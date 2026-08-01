from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


MAX_ASSET_BYTES = 40_000_000_000
MAX_TOTAL_BYTES = 80_000_000_000
ALLOWED_ROOTS = {"gemma", "ltx"}
PARALLEL_ASSET_THRESHOLD_BYTES = 8_000_000_000
RANGE_PART_BYTES = 512 * 1024 * 1024
DEFAULT_RANGE_WORKERS = 4
MAX_RANGE_WORKERS = 8
_asset_lock = threading.Lock()
_progress_lock = threading.Lock()
_session = requests.Session()
_progress: dict[str, Any] = {
    "phase": "idle",
    "operation": "",
    "required_roots": [],
    "total_bytes": 0,
    "completed_bytes": 0,
    "active_assets": {},
    "completed_assets": 0,
    "asset_count": 0,
    "updated_at": time.time(),
}


@dataclass(frozen=True)
class ModelAsset:
    path: str
    url: str
    sha256: str
    size_bytes: int


def _log(message: str) -> None:
    print(f"DRAKO_SALAD {message}", flush=True)


def _set_progress(**values: Any) -> None:
    with _progress_lock:
        _progress.update(values)
        _progress["updated_at"] = time.time()


def _update_asset_progress(asset: ModelAsset, written: int) -> None:
    with _progress_lock:
        active = dict(_progress.get("active_assets", {}))
        active[asset.path] = written
        _progress["active_assets"] = active
        _progress["completed_bytes"] = sum(active.values())
        _progress["updated_at"] = time.time()


def asset_progress_snapshot() -> dict[str, Any]:
    with _progress_lock:
        snapshot = dict(_progress)
        snapshot["active_assets"] = dict(_progress.get("active_assets", {}))
        return snapshot


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


def _download_serial(asset: ModelAsset, root: Path) -> None:
    destination = (root / asset.path).resolve()
    if root.resolve() not in destination.parents:
        raise ValueError("La ruta del modelo sale de la caché autorizada.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        destination.is_file()
        and destination.stat().st_size == asset.size_bytes
        and _sha256(destination) == asset.sha256
    ):
        _update_asset_progress(asset, asset.size_bytes)
        _log(f"asset_cached path={asset.path} bytes={asset.size_bytes}")
        return

    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.is_file() else 0
    if offset > asset.size_bytes:
        partial.unlink()
        offset = 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    _update_asset_progress(asset, offset)
    _log(f"asset_download_start path={asset.path} offset={offset} bytes={asset.size_bytes}")
    last_reported = offset
    last_report_time = time.monotonic()
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
            return _download_serial(asset, root)
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
                now = time.monotonic()
                if written - last_reported >= 256 * 1024 * 1024 or now - last_report_time >= 15:
                    _update_asset_progress(asset, written)
                    _log(
                        f"asset_download_progress path={asset.path} "
                        f"bytes={written} total={asset.size_bytes}"
                    )
                    last_reported = written
                    last_report_time = now
    if partial.stat().st_size != asset.size_bytes:
        raise ValueError("La descarga de un modelo quedó incompleta.")
    _set_progress(phase="verifying")
    _log(f"asset_verify_start path={asset.path} bytes={asset.size_bytes}")
    if _sha256(partial) != asset.sha256:
        partial.unlink(missing_ok=True)
        raise ValueError("La huella de un modelo descargado no coincide.")
    os.replace(partial, destination)
    _update_asset_progress(asset, asset.size_bytes)
    with _progress_lock:
        _progress["completed_assets"] = int(_progress.get("completed_assets", 0)) + 1
    _log(f"asset_ready path={asset.path} bytes={asset.size_bytes}")


def _range_specs(
    size: int,
    part_bytes: int = RANGE_PART_BYTES,
) -> tuple[tuple[int, int], ...]:
    if size <= 0 or part_bytes <= 0:
        raise ValueError("El tamano del rango no es valido.")
    return tuple(
        (start, min(size - 1, start + part_bytes - 1))
        for start in range(0, size, part_bytes)
    )


def _range_workers() -> int:
    try:
        configured = int(
            os.environ.get("DRAKO_MODEL_RANGE_WORKERS", DEFAULT_RANGE_WORKERS)
        )
    except (TypeError, ValueError):
        configured = DEFAULT_RANGE_WORKERS
    return max(1, min(MAX_RANGE_WORKERS, configured))


def _download_parallel(asset: ModelAsset, root: Path) -> None:
    destination = (root / asset.path).resolve()
    if root.resolve() not in destination.parents:
        raise ValueError("La ruta del modelo sale de la cache autorizada.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        destination.is_file()
        and destination.stat().st_size == asset.size_bytes
        and _sha256(destination) == asset.sha256
    ):
        for stale in destination.parent.glob(f"{destination.name}.part.range-*"):
            stale.unlink(missing_ok=True)
        _update_asset_progress(asset, asset.size_bytes)
        _log(f"asset_cached path={asset.path} bytes={asset.size_bytes}")
        return

    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    specs = _range_specs(asset.size_bytes)
    range_paths = tuple(
        partial.with_name(f"{partial.name}.range-{index:04d}")
        for index in range(len(specs))
    )
    written_by_range: dict[int, int] = {}
    for index, ((start, end), path) in enumerate(zip(specs, range_paths)):
        expected = end - start + 1
        if path.is_file() and path.stat().st_size == expected:
            written_by_range[index] = expected
        else:
            path.unlink(missing_ok=True)
            written_by_range[index] = 0

    progress_guard = threading.Lock()
    initial = sum(written_by_range.values())
    _update_asset_progress(asset, initial)
    workers = min(_range_workers(), len(specs))
    _log(
        f"asset_parallel_start path={asset.path} ranges={len(specs)} "
        f"workers={workers} resumed={initial} bytes={asset.size_bytes}"
    )
    last_reported = initial
    last_report_time = time.monotonic()

    def download_range(index: int) -> None:
        nonlocal last_reported, last_report_time
        start, end = specs[index]
        destination_part = range_paths[index]
        expected = end - start + 1
        if written_by_range[index] == expected:
            return
        temporary = destination_part.with_suffix(destination_part.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        written = 0
        with requests.get(
            asset.url,
            headers={"Range": f"bytes={start}-{end}"},
            stream=True,
            timeout=(20, 900),
            allow_redirects=False,
        ) as response:
            if response.status_code != 206:
                raise ValueError("El servidor de modelos no acepto rangos HTTPS.")
            content_range = response.headers.get("Content-Range", "").lower()
            if not content_range.startswith(f"bytes {start}-{end}/"):
                raise ValueError("El servidor devolvio un rango inesperado.")
            with temporary.open("wb") as output:
                for chunk in response.iter_content(8 * 1024 * 1024):
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > expected:
                        raise ValueError("Un rango supera su tamano declarado.")
                    output.write(chunk)
                    report = 0
                    with progress_guard:
                        written_by_range[index] = written
                        total_written = sum(written_by_range.values())
                        now = time.monotonic()
                        if (
                            total_written - last_reported >= 256 * 1024 * 1024
                            or now - last_report_time >= 15
                        ):
                            last_reported = total_written
                            last_report_time = now
                            report = total_written
                    if report:
                        _update_asset_progress(asset, report)
                        _log(
                            f"asset_download_progress path={asset.path} "
                            f"bytes={report} total={asset.size_bytes}"
                        )
        if written != expected:
            temporary.unlink(missing_ok=True)
            raise ValueError("La descarga de un rango quedo incompleta.")
        os.replace(temporary, destination_part)
        with progress_guard:
            written_by_range[index] = expected

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(download_range, range(len(specs))))

    if sum(path.stat().st_size for path in range_paths) != asset.size_bytes:
        raise ValueError("Los rangos no completan el modelo.")
    _set_progress(phase="assembling")
    _log(f"asset_assemble_start path={asset.path} ranges={len(range_paths)}")
    assembled = 0
    last_assembled = 0
    last_assemble_time = time.monotonic()
    with partial.open("wb") as output:
        for range_path in range_paths:
            with range_path.open("rb") as source:
                while chunk := source.read(8 * 1024 * 1024):
                    output.write(chunk)
                    assembled += len(chunk)
                    now = time.monotonic()
                    if (
                        assembled - last_assembled >= 256 * 1024 * 1024
                        or now - last_assemble_time >= 15
                    ):
                        _log(
                            f"asset_assemble_progress path={asset.path} "
                            f"bytes={assembled} total={asset.size_bytes}"
                        )
                        last_assembled = assembled
                        last_assemble_time = now
    if assembled != asset.size_bytes:
        partial.unlink(missing_ok=True)
        raise ValueError("El ensamblado paralelo quedo incompleto.")

    _set_progress(phase="verifying")
    _log(f"asset_verify_start path={asset.path} bytes={asset.size_bytes}")
    if _sha256(partial) != asset.sha256:
        partial.unlink(missing_ok=True)
        for range_path in range_paths:
            range_path.unlink(missing_ok=True)
        raise ValueError("La huella del modelo descargado no coincide.")
    os.replace(partial, destination)
    for range_path in range_paths:
        range_path.unlink(missing_ok=True)
    _update_asset_progress(asset, asset.size_bytes)
    with _progress_lock:
        _progress["completed_assets"] = int(_progress.get("completed_assets", 0)) + 1
    _log(f"asset_ready path={asset.path} bytes={asset.size_bytes}")


def _download(asset: ModelAsset, root: Path) -> None:
    destination = (root / asset.path).resolve()
    partial = destination.with_suffix(destination.suffix + ".part")
    if (
        asset.size_bytes >= PARALLEL_ASSET_THRESHOLD_BYTES
        and not partial.is_file()
    ):
        return _download_parallel(asset, root)
    result = _download_serial(asset, root)
    for stale in destination.parent.glob(f"{destination.name}.part.range-*"):
        stale.unlink(missing_ok=True)
    return result


def ensure_model_assets(
    manifest: Any,
    *,
    required_roots: set[str] | None = None,
    operation: str = "",
) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("Falta el manifiesto firmado de modelos de R2.")
    raw_assets = manifest.get("assets")
    if not isinstance(raw_assets, list) or not raw_assets:
        raise ValueError("El manifiesto de modelos está vacío.")
    assets = tuple(_parse_asset(value) for value in raw_assets)
    manifest_total = sum(asset.size_bytes for asset in assets)
    if manifest_total > MAX_TOTAL_BYTES:
        raise ValueError("El manifiesto de modelos supera el límite de seguridad.")

    requested = required_roots or ALLOWED_ROOTS
    if not requested or not requested.issubset(ALLOWED_ROOTS):
        raise ValueError("La selección de modelos no es válida.")
    selected = tuple(
        asset for asset in assets if asset.path.split("/", 1)[0] in requested
    )
    if not selected:
        raise ValueError("El manifiesto no contiene los modelos solicitados.")
    total = sum(asset.size_bytes for asset in selected)

    _set_progress(
        phase="downloading",
        operation=operation,
        required_roots=sorted(requested),
        total_bytes=total,
        completed_bytes=0,
        active_assets={},
        completed_assets=0,
        asset_count=len(selected),
    )
    _log(
        f"cache_prepare operation={operation or 'unknown'} "
        f"roots={','.join(sorted(requested))} assets={len(selected)} bytes={total}"
    )

    root = Path(os.environ.get("DRAKO_MODEL_CACHE", "/workspace/model-cache"))
    root.mkdir(parents=True, exist_ok=True)
    with _asset_lock:
        workers = min(4, len(selected))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda asset: _download(asset, root), selected))

    checkpoint = root / "ltx" / "ltx-2.3-22b-distilled-fp8.safetensors"
    gemma = root / "gemma"
    if "ltx" in requested and not checkpoint.is_file():
        raise ValueError("La caché no contiene el checkpoint LTX necesario.")
    if "gemma" in requested and not (gemma / "config.json").is_file():
        raise ValueError("La caché no contiene todos los modelos Gemma necesarios.")
    if "ltx" in requested:
        os.environ["LTX_DISTILLED_MODEL"] = str(checkpoint)
    if "gemma" in requested:
        os.environ["LTX_GEMMA_ROOT"] = str(gemma)
    _set_progress(phase="ready", completed_bytes=total, completed_assets=len(selected))
    _log(
        f"cache_ready operation={operation or 'unknown'} "
        f"roots={','.join(sorted(requested))} assets={len(selected)} bytes={total}"
    )
    return {
        "version": str(manifest.get("version", "")).strip(),
        "asset_count": len(selected),
        "total_bytes": total,
        "required_roots": sorted(requested),
    }

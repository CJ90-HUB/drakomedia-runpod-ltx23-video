from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import requests

from video_contract import (
    GenerationStageError,
    MAX_IMAGE_BYTES,
    MAX_OUTPUT_BYTES,
    VideoRequest,
    parse_request,
    public_error,
)


LTX_CODE_REVISION = "9377758131b1ffde4b7f766804590a6617bf2ab9"
LTX_MODEL_REVISION = "1d756cd27fa11c0896c4dfee093cd1bf36c7f7a1"
LTX_MODEL_FILE = "ltx-2.3-22b-distilled-fp8.safetensors"
SPATIAL_UPSAMPLER = (
    "/models/ltx-2.3/"
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
)
_session = requests.Session()
_pipeline: Any | None = None
_pipeline_load_seconds = 0.0
_pipeline_generation_count = 0
_pipeline_lock = threading.Lock()
_generation_lock = threading.Lock()


def _locate_checkpoint() -> Path:
    configured = os.environ.get("LTX_DISTILLED_MODEL", "").strip()
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            Path("/models") / LTX_MODEL_FILE,
            Path("/runpod-volume") / LTX_MODEL_FILE,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    for root in (
        Path("/runpod-volume/huggingface-cache"),
        Path("/runpod-volume"),
    ):
        if not root.exists():
            continue
        matches = list(root.rglob(LTX_MODEL_FILE))
        if matches:
            return matches[0]
    raise RuntimeError(
        "El modelo LTX-2.3 FP8 no está disponible en la caché de RunPod."
    )


def _locate_gemma() -> Path:
    configured = os.environ.get("LTX_GEMMA_ROOT", "").strip()
    if configured and (Path(configured) / "config.json").is_file():
        return Path(configured)
    candidates = [
        Path("/models/gemma-3-12b"),
        Path("/runpod-volume/gemma-3-12b"),
        Path(
            "/runpod-volume/huggingface-cache/hub/"
            "models--Lightricks--gemma-3-12b-it-qat-q4_0-unquantized"
        ),
        Path(
            "/runpod-volume/huggingface-cache/"
            "models--Lightricks--gemma-3-12b-it-qat-q4_0-unquantized"
        ),
    ]
    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return candidate
        snapshots = candidate / "snapshots"
        if snapshots.is_dir():
            for snapshot in snapshots.iterdir():
                if (snapshot / "config.json").is_file():
                    return snapshot
    raise RuntimeError(
        "Gemma 3 no está disponible en la caché de modelos de RunPod."
    )


def _validate_runtime_dependencies() -> None:
    # The immutable, working production image is the compatibility authority.
    # Import the exact module that exposed the original broken dependency pair.
    importlib.import_module("transformers.utils.hub")

    actual = {
        package: importlib.metadata.version(package)
        for package in ("transformers", "huggingface-hub")
    }
    print(
        "DrakoMedia LTX-2.3 · runtime verificado · "
        f"transformers={actual['transformers']} · "
        f"huggingface-hub={actual['huggingface-hub']}",
        flush=True,
    )


def _download_image(url: str, destination: Path) -> None:
    with _session.get(
        url,
        stream=True,
        timeout=(10, 120),
        allow_redirects=False,
    ) as response:
        response.raise_for_status()
        expected = int(response.headers.get("content-length", 0) or 0)
        if expected > MAX_IMAGE_BYTES:
            raise ValueError("La imagen inicial es demasiado grande.")
        total = 0
        with destination.open("wb") as output:
            for chunk in response.iter_content(256 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_IMAGE_BYTES:
                    raise ValueError(
                        "La imagen inicial es demasiado grande."
                    )
                output.write(chunk)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _upload(url: str, source: Path) -> str:
    size = source.stat().st_size
    if size <= 0 or size > MAX_OUTPUT_BYTES:
        raise ValueError("El vídeo generado tiene un tamaño no permitido.")
    digest = _sha256(source)
    with source.open("rb") as payload:
        response = _session.put(
            url,
            data=payload,
            headers={
                "Content-Type": "video/mp4",
                "Content-Length": str(size),
            },
            timeout=(10, 900),
            allow_redirects=False,
        )
    response.raise_for_status()
    return digest


def _stage(stage: str, action: Any) -> Any:
    try:
        return action()
    except GenerationStageError:
        raise
    except Exception as exc:
        raise GenerationStageError(stage, exc) from exc


def _load_pipeline() -> Any:
    global _pipeline
    global _pipeline_load_seconds

    if _pipeline is not None:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline
        try:
            _stage("runtime", _validate_runtime_dependencies)
            quantization_module = _stage(
                "pipeline_import",
                lambda: importlib.import_module(
                    "ltx_pipelines.utils.quantization_factory"
                ),
            )
            pipeline_module = _stage(
                "pipeline_import",
                lambda: importlib.import_module("video_only_distilled"),
            )
            checkpoint = str(_stage("checkpoint", _locate_checkpoint))
            gemma_root = str(_stage("gemma", _locate_gemma))
            started = time.monotonic()
            print(
                "DrakoMedia LTX-2.3 · "
                "cargando el modelo persistente una sola vez",
                flush=True,
            )
            _pipeline = _stage(
                "model_load",
                lambda: pipeline_module.VideoOnlyDistilledPipeline(
                    distilled_checkpoint_path=checkpoint,
                    spatial_upsampler_path=SPATIAL_UPSAMPLER,
                    gemma_root=gemma_root,
                    loras=(),
                    quantization=(
                        quantization_module.QuantizationKind.FP8_SCALED_MM
                        .to_policy(checkpoint_path=checkpoint)
                    ),
                ),
            )
        except Exception as exc:
            print(
                "DrakoMedia LTX-2.3 · fallo de carga · "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            raise
        _pipeline_load_seconds = time.monotonic() - started
        print(
            "DrakoMedia LTX-2.3 · modelo persistente preparado en "
            f"{_pipeline_load_seconds:.1f} s",
            flush=True,
        )
        return _pipeline


def _run_pipeline(request: VideoRequest, folder: Path) -> tuple[Path, float, bool]:
    global _pipeline_generation_count

    import torch
    from ltx_core.model.video_vae import (
        TilingConfig,
        get_video_chunks_number,
    )
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.constants import (
        DISTILLED_SIGMAS,
        STAGE_2_DISTILLED_SIGMAS,
    )
    from ltx_pipelines.utils.media_io import encode_video

    reused = _pipeline is not None
    pipeline = _load_pipeline()
    output = folder / "result.mp4"
    images = []
    if request.image_url:
        image = folder / "source-image"
        _download_image(request.image_url, image)
        images.append(
            ImageConditioningInput(
                path=str(image),
                frame_idx=0,
                strength=0.95,
                crf=0,
            )
        )

    started = time.monotonic()
    with _generation_lock, torch.inference_mode():
        tiling = TilingConfig.default()
        video_chunks = get_video_chunks_number(request.frames, tiling)
        pipeline_args: dict[str, Any] = {}
        if request.profile == "ultra-cheap-plus":
            pipeline_args = {
                "stage_1_sigmas": DISTILLED_SIGMAS[[0, 4, 6, 7, 8]],
                "stage_2_sigmas": STAGE_2_DISTILLED_SIGMAS[[0, 2, 3]],
            }
        video = pipeline(
            prompt=request.prompt,
            seed=request.seed,
            height=request.height,
            width=request.width,
            num_frames=request.frames,
            frame_rate=request.fps,
            images=images,
            tiling_config=tiling,
            enhance_prompt=False,
            **pipeline_args,
        )
        encode_video(
            video=video,
            fps=request.fps,
            audio=None,
            output_path=str(output),
            video_chunks_number=video_chunks,
            crf=22 if request.profile == "ultra-cheap-plus" else 19,
            preset=(
                "ultrafast"
                if request.profile == "ultra-cheap-plus"
                else "veryfast"
            ),
        )
        del video

    generation_seconds = time.monotonic() - started
    _pipeline_generation_count += 1
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not output.is_file():
        raise RuntimeError("LTX-2.3 no produjo el archivo de vídeo.")
    return output, generation_seconds, reused


def _generate(request: VideoRequest, folder: Path) -> dict[str, Any]:
    output, generation_seconds, pipeline_reused = _run_pipeline(
        request,
        folder,
    )
    sha256 = _upload(request.upload_url, output)
    return {
        "ok": True,
        "profile": request.profile,
        "request_id": request.request_id,
        "object_key": request.object_key,
        "sha256": sha256,
        "size_bytes": output.stat().st_size,
        "width": request.width,
        "height": request.height,
        "frames": request.frames,
        "fps": request.fps,
        "duration_ms": round(
            request.frames / request.fps * 1_000
        ),
        "seed": request.seed,
        "engine": "ltx-2.3-distilled-fp8-video-only",
        "pipeline_mode": "persistent",
        "pipeline_reused": pipeline_reused,
        "pipeline_load_ms": round(_pipeline_load_seconds * 1_000),
        "generation_ms": round(generation_seconds * 1_000),
        "generation_count": _pipeline_generation_count,
        "code_revision": LTX_CODE_REVISION,
        "model_revision": LTX_MODEL_REVISION,
    }


def handler(event: dict[str, Any]) -> dict[str, Any]:
    try:
        request = parse_request(event)
        print(
            "DrakoMedia LTX-2.3 · "
            f"solicitud {request.request_id} · "
            f"{request.profile} · "
            f"{request.width}x{request.height} · "
            f"{request.frames} frames",
            flush=True,
        )
        with tempfile.TemporaryDirectory(
            prefix="drakomedia-ltx23-"
        ) as name:
            return _generate(request, Path(name))
    except Exception as exc:
        print(
            "DrakoMedia LTX-2.3 · error "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return public_error(exc)


if __name__ == "__main__":
    import runpod

    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )
    runpod.serverless.start({"handler": handler})

from __future__ import annotations

import gc
import json
import os
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from contract import (
    MAX_IMAGE_BYTES,
    MotionAnalysisRequest,
    parse_motion_request,
    public_error,
)


GEMMA_REPOSITORY = "Lightricks/gemma-3-12b-it-qat-q4_0-unquantized"
GEMMA_REVISION = "d62fe4f1995ade703b49a0f3c0d0f161237ef437"
GEMMA_ROOT = Path("/runpod-volume/gemma-3-12b")
GEMMA_READY = GEMMA_ROOT / ".drakomedia-ready"
_session = requests.Session()
_model_lock = threading.Lock()

MOTION_SYSTEM_PROMPT = """
You are a conservative motion director for LTX-2.3 image-to-video.
Inspect the actual supplied image and the scene context. Return ONLY one
compact JSON object:
{"motion_prompt":"...","confidence":0.0,"risk":"low|medium|high","reason":"..."}

motion_prompt must contain 20-55 English words and describe one continuous
documentary shot. State one restrained camera behavior and only physically
plausible motion of elements visibly present in the image. Never introduce,
remove, duplicate or transform subjects or objects. Never request a scene
transition, a hidden area, a new action, readable text, fast camera motion,
camera roll, aggressive zoom, morphing or complex choreography. Preserve
identity, subject count, geometry and composition. For people in close or
medium shots, prefer a locked camera with breathing, blinking and tiny
existing gestures. confidence measures how certain you are that the motion
matches visible content. risk is high when faces, hands, dense machinery,
text or ambiguous geometry make animation fragile. reason must be a short
Spanish explanation. Motion level: safe means locked or nearly imperceptible;
natural means one subtle documentary camera move and plausible visible
motion; dynamic means one visible but controlled camera move and stronger
existing environmental motion, without inventing actions or elements.
""".strip()


def _gemma_is_ready(path: Path) -> bool:
    base_files = (
        (path / "config.json").is_file()
        and (path / "model.safetensors.index.json").is_file()
        and (path / "tokenizer.model").is_file()
    )
    if not base_files:
        return False
    if path != GEMMA_ROOT:
        return True
    return (
        GEMMA_READY.is_file()
        and GEMMA_READY.read_text(encoding="utf-8").strip()
        == GEMMA_REVISION
    )


def _ensure_gemma() -> Path:
    configured = os.environ.get("LTX_GEMMA_ROOT", "").strip()
    if configured and _gemma_is_ready(Path(configured)):
        return Path(configured)
    baked = Path("/models/gemma-3-12b")
    if _gemma_is_ready(baked):
        return baked
    if _gemma_is_ready(GEMMA_ROOT):
        return GEMMA_ROOT

    with _model_lock:
        if _gemma_is_ready(GEMMA_ROOT):
            return GEMMA_ROOT
        if not GEMMA_ROOT.parent.is_dir():
            raise RuntimeError(
                "El volumen compartido de Gemma 3 no está montado."
            )
        from huggingface_hub import snapshot_download

        temporary = GEMMA_ROOT.with_name("gemma-3-12b.partial")
        shutil.rmtree(temporary, ignore_errors=True)
        print(
            "DrakoMedia Cloud Motion · preparando Gemma 3 una sola vez "
            "en el volumen compartido",
            flush=True,
        )
        snapshot_download(
            GEMMA_REPOSITORY,
            revision=GEMMA_REVISION,
            local_dir=temporary,
            ignore_patterns=["README.md", ".gitattributes"],
        )
        if not _gemma_is_ready(temporary):
            raise RuntimeError(
                "La descarga verificada de Gemma 3 quedó incompleta."
            )
        shutil.rmtree(GEMMA_ROOT, ignore_errors=True)
        temporary.rename(GEMMA_ROOT)
        GEMMA_READY.write_text(GEMMA_REVISION, encoding="utf-8")
        return GEMMA_ROOT


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
                    raise ValueError("La imagen inicial es demasiado grande.")
                output.write(chunk)


def _extract_motion_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text.strip(),
        flags=re.I | re.S,
    )
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("El director visual no devolvió JSON.")
    return json.loads(cleaned[start : end + 1])


def _validate_motion(value: dict[str, Any]) -> dict[str, Any]:
    prompt = " ".join(str(value.get("motion_prompt", "")).split())
    words = prompt.split()
    forbidden = (
        "new person",
        "new people",
        "new object",
        "scene transition",
        "camera roll",
        "fast pan",
        "fast tilt",
        "aggressive zoom",
        "morph",
        "transform into",
    )
    if (
        len(words) < 20
        or len(words) > 55
        or any(term in prompt.lower() for term in forbidden)
    ):
        raise ValueError("El prompt de movimiento no superó la validación.")
    confidence = max(0.0, min(1.0, float(value.get("confidence", 0.0))))
    risk = str(value.get("risk", "high")).strip().lower()
    if risk not in {"low", "medium", "high"}:
        risk = "high"
    return {
        "motion_prompt": prompt,
        "confidence": confidence,
        "risk": risk,
        "reason": " ".join(str(value.get("reason", "")).split())[:240],
    }


def _motion_context(scene: Any) -> str:
    return (
        f"{MOTION_SYSTEM_PROMPT}\n\n"
        f"Scene title: {scene.title}\n"
        f"Narration: {scene.narration}\n"
        f"Image prompt: {scene.image_prompt}\n"
        f"Visual intention: {scene.visual_intent}\n"
        f"Motion level: {scene.level}"
    )


def _prepare_motion_image(path: Path) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail((768, 768), Image.Resampling.LANCZOS)
    return image


def _generate_result(
    model: Any,
    processor: Any,
    image: Image.Image,
    context: str,
) -> dict[str, Any]:
    return _generate_results(
        model,
        processor,
        [image],
        [context],
    )[0]


def _generate_results(
    model: Any,
    processor: Any,
    images: list[Image.Image],
    contexts: list[str],
) -> list[dict[str, Any]]:
    import torch

    if not images or len(images) != len(contexts):
        raise ValueError("El lote visual no es válido.")
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": context},
                ],
            }
        ]
        for image, context in zip(images, contexts, strict=True)
    ]
    prompts = [
        processor.apply_chat_template(
            message,
            add_generation_prompt=True,
            tokenize=False,
        )
        for message in messages
    ]
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"
    inputs = processor(
        text=prompts,
        images=images,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    generated = None
    trimmed = None
    try:
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=112,
                do_sample=False,
            )
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        raw_values = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        results: list[dict[str, Any]] = []
        for raw in raw_values:
            results.append(
                _validate_motion(_extract_motion_object(raw))
            )
        return results
    finally:
        del inputs
        del generated
        del trimmed


def _analyze_motion(
    request: MotionAnalysisRequest,
    folder: Path,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_root = str(_ensure_gemma())
    print(
        "DrakoMedia Cloud Motion · cargando el director visual Gemma 3",
        flush=True,
    )
    started = time.monotonic()
    processor = AutoProcessor.from_pretrained(
        model_root,
        local_files_only=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_root,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    load_seconds = time.monotonic() - started
    results: list[dict[str, Any]] = []
    analysis_started = time.monotonic()
    batch_size = max(
        1,
        min(24, int(os.environ.get("DRAKO_MOTION_BATCH_SIZE", "16"))),
    )
    try:
        for batch_start in range(0, len(request.scenes), batch_size):
            scenes = request.scenes[batch_start : batch_start + batch_size]
            paths: list[Path] = []
            images: list[Image.Image] = []
            try:
                for offset, scene in enumerate(scenes):
                    index = batch_start + offset + 1
                    image_path = folder / f"motion-{index:03d}.image"
                    _download_image(scene.image_url, image_path)
                    paths.append(image_path)
                    images.append(_prepare_motion_image(image_path))
                batch_results = _generate_results(
                    model,
                    processor,
                    images,
                    [_motion_context(scene) for scene in scenes],
                )
            finally:
                for image in images:
                    image.close()
                for image_path in paths:
                    image_path.unlink(missing_ok=True)
            for scene, result in zip(
                scenes,
                batch_results,
                strict=True,
            ):
                result.update(
                    {
                        "scene_id": scene.scene_id,
                        "analysis_signature": scene.analysis_signature,
                        "ok": True,
                    }
                )
                results.append(result)
            print(
                "DrakoMedia Cloud Motion · "
                f"analizadas {len(results)} de {len(request.scenes)} "
                f"(lote GPU {len(scenes)})",
                flush=True,
            )
    finally:
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "ok": True,
        "request_id": request.request_id,
        "operation": "analyze_motion",
        "results": results,
        "director_engine": "gemma-3-12b-vision",
        "director_mode": "cloud-batch-split",
        "model_revision": GEMMA_REVISION,
        "model_load_ms": round(load_seconds * 1_000),
        "analysis_ms": round(
            (time.monotonic() - analysis_started) * 1_000
        ),
        "scene_count": len(results),
    }


def handler(event: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = event.get("input") if isinstance(event, dict) else None
        if (
            isinstance(payload, dict)
            and payload.get("operation") == "prepare_model"
        ):
            request_id = str(payload.get("request_id", "")).strip()
            if not request_id or len(request_id) > 160:
                raise ValueError("Falta request_id para preparar Gemma 3.")
            started = time.monotonic()
            root = _ensure_gemma()
            model_bytes = sum(
                path.stat().st_size
                for path in root.rglob("*")
                if path.is_file()
            )
            return {
                "ok": True,
                "request_id": request_id,
                "operation": "prepare_model",
                "director_engine": "gemma-3-12b-vision",
                "model_revision": GEMMA_REVISION,
                "model_bytes": model_bytes,
                "prepare_ms": round(
                    (time.monotonic() - started) * 1_000
                ),
            }
        request = parse_motion_request(event)
        print(
            "DrakoMedia Cloud Motion · "
            f"solicitud {request.request_id} · "
            f"{len(request.scenes)} imágenes",
            flush=True,
        )
        with tempfile.TemporaryDirectory(
            prefix="drakomedia-motion-"
        ) as name:
            return _analyze_motion(request, Path(name))
    except Exception as exc:
        print(
            f"DrakoMedia Cloud Motion · error {type(exc).__name__}",
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

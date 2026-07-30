from __future__ import annotations

import ipaddress
import os
import socket
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


MAX_PROMPT_CHARS = 4_000
MAX_IMAGE_BYTES = 30 * 1024 * 1024
MAX_OUTPUT_BYTES = 2 * 1024 * 1024 * 1024
MAX_MOTION_SCENES = 200
DEFAULT_ALLOWED_HOST = (
    "a76220a52aaf357ce8909685181757af.r2.cloudflarestorage.com"
)


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class VideoRequest:
    request_id: str
    prompt: str
    seed: int
    width: int
    height: int
    frames: int
    fps: int
    image_url: str
    upload_url: str
    object_key: str


@dataclass(frozen=True)
class MotionSceneRequest:
    scene_id: str
    analysis_signature: str
    image_url: str
    title: str
    narration: str
    image_prompt: str
    visual_intent: str
    level: str


@dataclass(frozen=True)
class MotionAnalysisRequest:
    request_id: str
    scenes: tuple[MotionSceneRequest, ...]


def _text(
    value: Any,
    name: str,
    *,
    required: bool = False,
    limit: int = 1_000,
) -> str:
    result = str(value or "").strip()
    if required and not result:
        raise ContractError(f"Falta {name}.")
    if len(result) > limit:
        raise ContractError(f"{name} supera el tamaño permitido.")
    return result


def _integer(
    value: Any,
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} debe ser un número entero.") from exc
    if result < minimum or result > maximum:
        raise ContractError(
            f"{name} debe estar entre {minimum} y {maximum}."
        )
    return result


def _is_public_host(host: str) -> bool:
    try:
        addresses = socket.getaddrinfo(
            host,
            443,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise ContractError(
            "No se pudo resolver el servidor de almacenamiento."
        ) from exc
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False
    return True


def validate_storage_url(value: Any, name: str) -> str:
    url = _text(value, name, required=True, limit=8_192)
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ContractError(
            f"{name} debe ser un enlace HTTPS temporal válido."
        )
    allowed_hosts = tuple(
        item.strip().lower()
        for item in os.environ.get(
            "DRAKO_ALLOWED_STORAGE_HOSTS",
            DEFAULT_ALLOWED_HOST,
        ).split(",")
        if item.strip()
    )
    host = parsed.hostname.lower()
    if host not in allowed_hosts:
        raise ContractError(
            f"{name} no pertenece al almacenamiento autorizado."
        )
    if (
        os.environ.get("DRAKO_SKIP_DNS_GUARD", "0") != "1"
        and not _is_public_host(host)
    ):
        raise ContractError(
            f"{name} no apunta a una dirección pública."
        )
    return url


def parse_request(event: dict[str, Any]) -> VideoRequest:
    if not isinstance(event, dict) or not isinstance(
        event.get("input"),
        dict,
    ):
        raise ContractError(
            "La solicitud de RunPod no contiene input."
        )
    payload = event["input"]
    output = payload.get("output") or {}
    source = payload.get("source") or {}
    if not isinstance(output, dict) or not isinstance(source, dict):
        raise ContractError(
            "source y output deben ser objetos."
        )

    width = _integer(
        payload.get("width", 1920),
        "width",
        minimum=256,
        maximum=1920,
    )
    height = _integer(
        payload.get("height", 1088),
        "height",
        minimum=256,
        maximum=1088,
    )
    if width % 64 or height % 64:
        raise ContractError(
            "width y height deben ser múltiplos de 64."
        )

    frames = _integer(
        payload.get("frames", 121),
        "frames",
        minimum=9,
        maximum=241,
    )
    if (frames - 1) % 8:
        raise ContractError(
            "frames debe cumplir 8 × K + 1."
        )

    image_url = ""
    if source.get("image_url"):
        image_url = validate_storage_url(
            source.get("image_url"),
            "source.image_url",
        )

    object_key = _text(
        output.get("object_key"),
        "output.object_key",
        required=True,
        limit=1_024,
    )
    if not object_key.startswith("safe-to-delete/"):
        raise ContractError(
            "El resultado debe guardarse bajo safe-to-delete/."
        )

    return VideoRequest(
        request_id=_text(
            payload.get("request_id"),
            "request_id",
            required=True,
            limit=160,
        ),
        prompt=_text(
            payload.get("prompt"),
            "prompt",
            required=True,
            limit=MAX_PROMPT_CHARS,
        ),
        seed=_integer(
            payload.get("seed", 42),
            "seed",
            minimum=0,
            maximum=2_147_483_647,
        ),
        width=width,
        height=height,
        frames=frames,
        fps=_integer(
            payload.get("fps", 24),
            "fps",
            minimum=8,
            maximum=60,
        ),
        image_url=image_url,
        upload_url=validate_storage_url(
            output.get("upload_url"),
            "output.upload_url",
        ),
        object_key=object_key,
    )


def parse_motion_request(event: dict[str, Any]) -> MotionAnalysisRequest:
    if not isinstance(event, dict) or not isinstance(
        event.get("input"),
        dict,
    ):
        raise ContractError(
            "La solicitud de RunPod no contiene input."
        )
    payload = event["input"]
    if payload.get("operation") != "analyze_motion":
        raise ContractError("La operación de movimiento no es válida.")
    raw_scenes = payload.get("scenes")
    if (
        not isinstance(raw_scenes, list)
        or not raw_scenes
        or len(raw_scenes) > MAX_MOTION_SCENES
    ):
        raise ContractError(
            f"scenes debe contener entre 1 y {MAX_MOTION_SCENES} imágenes."
        )

    scenes: list[MotionSceneRequest] = []
    for index, raw in enumerate(raw_scenes, start=1):
        if not isinstance(raw, dict):
            raise ContractError(f"La escena {index} no es válida.")
        level = _text(
            raw.get("level", "natural"),
            f"scenes[{index}].level",
            required=True,
            limit=16,
        ).lower()
        if level not in {"safe", "natural", "dynamic"}:
            raise ContractError(
                f"scenes[{index}].level no es válido."
            )
        signature = _text(
            raw.get("analysis_signature"),
            f"scenes[{index}].analysis_signature",
            required=True,
            limit=64,
        ).upper()
        if (
            len(signature) != 64
            or any(
                character not in "0123456789ABCDEF"
                for character in signature
            )
        ):
            raise ContractError(
                f"scenes[{index}].analysis_signature no es válida."
            )
        scenes.append(
            MotionSceneRequest(
                scene_id=_text(
                    raw.get("scene_id"),
                    f"scenes[{index}].scene_id",
                    required=True,
                    limit=160,
                ),
                analysis_signature=signature,
                image_url=validate_storage_url(
                    raw.get("image_url"),
                    f"scenes[{index}].image_url",
                ),
                title=_text(
                    raw.get("title"),
                    f"scenes[{index}].title",
                    limit=300,
                ),
                narration=_text(
                    raw.get("narration"),
                    f"scenes[{index}].narration",
                    limit=4_000,
                ),
                image_prompt=_text(
                    raw.get("image_prompt"),
                    f"scenes[{index}].image_prompt",
                    limit=4_000,
                ),
                visual_intent=_text(
                    raw.get("visual_intent"),
                    f"scenes[{index}].visual_intent",
                    limit=2_000,
                ),
                level=level,
            )
        )
    return MotionAnalysisRequest(
        request_id=_text(
            payload.get("request_id"),
            "request_id",
            required=True,
            limit=160,
        ),
        scenes=tuple(scenes),
    )


def public_error(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, ContractError):
        return {
            "ok": False,
            "error_code": "INVALID_INPUT",
            "message": str(exc),
        }
    return {
        "ok": False,
        "error_code": "GENERATION_FAILED",
        "message": "LTX-2.3 no pudo completar la generación.",
    }

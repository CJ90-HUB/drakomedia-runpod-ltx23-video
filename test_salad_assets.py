from pathlib import Path

import pytest

from salad_assets import ALLOWED_ROOTS, _parse_asset


def test_asset_path_is_scoped() -> None:
    asset = _parse_asset(
        {
            "path": "gemma/config.json",
            "url": "https://example.invalid/config.json",
            "sha256": "a" * 64,
            "size_bytes": 12,
        }
    )
    assert asset.path == "gemma/config.json"


@pytest.mark.parametrize("path", ["../secret", "gemma/../../secret", "other/model.bin"])
def test_asset_path_rejects_traversal(path: str) -> None:
    with pytest.raises(ValueError):
        _parse_asset(
            {
                "path": path,
                "url": "https://example.invalid/model",
                "sha256": "a" * 64,
                "size_bytes": 12,
            }
        )


def test_salad_server_is_reachable_by_platform_probes() -> None:
    start_script = (Path(__file__).parent / "start-salad.sh").read_text(
        encoding="utf-8"
    )
    assert "uvicorn salad_server:app --host 0.0.0.0 --port 8080" in start_script
    assert "--host 127.0.0.1" not in start_script


def test_motion_analysis_downloads_only_gemma() -> None:
    server = (Path(__file__).parent / "salad_server.py").read_text(encoding="utf-8")
    assert 'required_roots = {"gemma"} if operation == "analyze_motion"' in server
    assert 'else {"gemma", "ltx"}' in server


def test_all_model_roots_remain_explicitly_scoped() -> None:
    assert ALLOWED_ROOTS == {"gemma", "ltx"}


def test_salad_worker_has_unbuffered_progress_logs() -> None:
    dockerfile = (Path(__file__).parent / "Dockerfile.salad").read_text(
        encoding="utf-8"
    )
    assets = (Path(__file__).parent / "salad_assets.py").read_text(
        encoding="utf-8"
    )
    assert "PYTHONUNBUFFERED=1" in dockerfile
    assert "asset_download_progress" in assets
    assert "cache_ready" in assets

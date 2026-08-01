from pathlib import Path

import pytest

from salad_assets import (
    ALLOWED_ROOTS,
    PARALLEL_ASSET_THRESHOLD_BYTES,
    _parse_asset,
    _range_specs,
)


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


def test_parallel_ranges_cover_asset_without_gaps() -> None:
    specs = _range_specs(1_025, 256)
    assert specs == (
        (0, 255),
        (256, 511),
        (512, 767),
        (768, 1_023),
        (1_024, 1_024),
    )
    assert sum(end - start + 1 for start, end in specs) == 1_025


def test_ltx_checkpoint_qualifies_for_parallel_download() -> None:
    assert 29_531_884_062 >= PARALLEL_ASSET_THRESHOLD_BYTES


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

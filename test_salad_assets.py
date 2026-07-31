from pathlib import Path

import pytest

from salad_assets import _parse_asset


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

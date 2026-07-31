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

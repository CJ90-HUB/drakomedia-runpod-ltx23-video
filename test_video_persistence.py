import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import handler


class _Quantization:
    def to_policy(self, checkpoint_path=None):
        return ("fp8", checkpoint_path)


class _QuantizationKind:
    FP8_SCALED_MM = _Quantization()


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        handler._pipeline = None
        handler._pipeline_load_seconds = 0

    def test_pipeline_is_loaded_only_once(self):
        created = []

        class FakePipeline:
            def __init__(self, **settings):
                created.append(settings)

        distilled = types.ModuleType("video_only_distilled")
        distilled.VideoOnlyDistilledPipeline = FakePipeline
        quantization = types.ModuleType(
            "ltx_pipelines.utils.quantization_factory"
        )
        quantization.QuantizationKind = _QuantizationKind

        modules = {
            "ltx_pipelines": types.ModuleType("ltx_pipelines"),
            "video_only_distilled": distilled,
            "ltx_pipelines.utils": types.ModuleType("ltx_pipelines.utils"),
            "ltx_pipelines.utils.quantization_factory": quantization,
        }
        with (
            patch.dict(sys.modules, modules),
            patch.object(handler, "_validate_runtime_dependencies"),
            patch.object(
                handler,
                "_locate_checkpoint",
                return_value=Path("/models/model.safetensors"),
            ),
            patch.object(
                handler,
                "_locate_gemma",
                return_value=Path("/models/gemma"),
            ),
        ):
            first = handler._load_pipeline()
            second = handler._load_pipeline()

        self.assertIs(first, second)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0]["quantization"][0], "fp8")
        self.assertTrue(
            created[0]["quantization"][1].replace("\\", "/").endswith(
                "/models/model.safetensors"
            )
        )

    def test_runtime_check_accepts_proven_production_versions(self):
        versions = {
            "transformers": "4.57.6",
            "huggingface-hub": "0.36.0",
        }
        with (
            patch.object(handler.importlib, "import_module"),
            patch.object(
                handler.importlib.metadata,
                "version",
                side_effect=lambda package: versions[package],
            ),
        ):
            handler._validate_runtime_dependencies()

    def test_gemma_locator_includes_shared_volume_layout(self):
        source = Path(handler.__file__).read_text(encoding="utf-8")
        self.assertIn('Path("/runpod-volume/gemma-3-12b")', source)

    def test_worker_registers_before_lazy_model_load(self):
        source = Path(handler.__file__).read_text(encoding="utf-8")
        main_block = source.split('if __name__ == "__main__":', 1)[1]

        self.assertIn('runpod.serverless.start({"handler": handler})', main_block)
        self.assertNotIn("_load_pipeline()", main_block)


if __name__ == "__main__":
    unittest.main()

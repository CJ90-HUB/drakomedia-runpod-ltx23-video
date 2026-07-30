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

        distilled = types.ModuleType("ltx_pipelines.distilled")
        distilled.DistilledPipeline = FakePipeline
        quantization = types.ModuleType(
            "ltx_pipelines.utils.quantization_factory"
        )
        quantization.QuantizationKind = _QuantizationKind

        modules = {
            "ltx_pipelines": types.ModuleType("ltx_pipelines"),
            "ltx_pipelines.distilled": distilled,
            "ltx_pipelines.utils": types.ModuleType("ltx_pipelines.utils"),
            "ltx_pipelines.utils.quantization_factory": quantization,
        }
        with (
            patch.dict(sys.modules, modules),
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


if __name__ == "__main__":
    unittest.main()

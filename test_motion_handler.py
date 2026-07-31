import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import motion_handler


class MotionHandlerTests(unittest.TestCase):
    def test_accepts_individualized_safe_prompt(self):
        result = motion_handler._validate_motion(
            {
                "motion_prompt": (
                    "The locked camera holds the scientist at center frame "
                    "while subtle breathing and a slight movement of loose "
                    "fabric preserve the original composition and identity."
                ),
                "confidence": 0.82,
                "risk": "medium",
                "reason": "Movimiento mínimo adecuado para un retrato.",
            }
        )
        self.assertEqual(result["risk"], "medium")
        self.assertGreaterEqual(len(result["motion_prompt"].split()), 20)

    def test_rejects_generic_short_prompt(self):
        with self.assertRaises(ValueError):
            motion_handler._validate_motion(
                {
                    "motion_prompt": "Slow camera movement.",
                    "confidence": 1,
                    "risk": "low",
                }
            )

    def test_shared_model_requires_revision_marker(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({}),
                encoding="utf-8",
            )
            (root / "tokenizer.model").write_bytes(b"tokenizer")
            with (
                patch.object(motion_handler, "GEMMA_ROOT", root),
                patch.object(
                    motion_handler,
                    "GEMMA_READY",
                    root / ".drakomedia-ready",
                ),
            ):
                self.assertFalse(motion_handler._gemma_is_ready(root))
                (root / ".drakomedia-ready").write_text(
                    motion_handler.GEMMA_REVISION,
                    encoding="utf-8",
                )
                self.assertTrue(motion_handler._gemma_is_ready(root))

    def test_prepare_model_returns_verified_size(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "weights.bin").write_bytes(b"verified-model")
            with patch.object(
                motion_handler,
                "_ensure_gemma",
                return_value=root,
            ):
                output = motion_handler.handler(
                    {
                        "input": {
                            "operation": "prepare_model",
                            "request_id": "prepare-test",
                        }
                    }
                )
        self.assertTrue(output["ok"])
        self.assertEqual(output["operation"], "prepare_model")
        self.assertEqual(output["model_bytes"], 14)

    def test_batch_passes_one_image_list_per_prompt(self):
        class FakeTensor:
            shape = (2, 8)

        class FakeInputs(dict):
            def to(self, _device):
                return self

        processor = MagicMock()
        processor.tokenizer = MagicMock()
        processor.apply_chat_template.side_effect = ["prompt-1", "prompt-2"]
        processor.return_value = FakeInputs(input_ids=FakeTensor())
        processor.batch_decode.return_value = [
            json.dumps(
                {
                    "motion_prompt": (
                        "The camera remains steady while existing dust drifts "
                        "gently across the visible rails and distant heat haze "
                        "shimmers without changing the original composition."
                    ),
                    "confidence": 0.9,
                    "risk": "low",
                    "reason": "Safe visible environmental motion.",
                }
            ),
            json.dumps(
                {
                    "motion_prompt": (
                        "A slow forward camera glide follows the existing "
                        "viaduct while sunlight shifts subtly across its "
                        "concrete pillars and the surrounding savanna stays "
                        "fully consistent."
                    ),
                    "confidence": 0.85,
                    "risk": "medium",
                    "reason": "Controlled infrastructure reveal.",
                }
            ),
        ]
        model = MagicMock()
        model.generate.return_value = MagicMock()
        model.generate.return_value.__getitem__.return_value = MagicMock()
        images = [MagicMock(), MagicMock()]

        results = motion_handler._generate_results(
            model,
            processor,
            images,
            ["context-1", "context-2"],
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(
            processor.call_args.kwargs["images"],
            [[images[0]], [images[1]]],
        )


if __name__ == "__main__":
    unittest.main()

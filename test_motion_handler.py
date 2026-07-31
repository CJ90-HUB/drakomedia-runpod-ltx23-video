import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()

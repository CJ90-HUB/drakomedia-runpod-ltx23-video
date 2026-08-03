import ast
import unittest
from pathlib import Path


class VideoOnlySourceTests(unittest.TestCase):
    def test_pipeline_omits_audio_modality_and_decoder(self):
        source = Path(__file__).with_name(
            "video_only_distilled.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        ]
        self.assertFalse(
            any(
                keyword.arg == "audio"
                for call in calls
                for keyword in call.keywords
            )
        )
        self.assertNotIn("audio_decoder", source)
        self.assertNotIn("AudioDecoder", source)
        self.assertGreaterEqual(source.count("SimpleDenoiser(video_context, None)"), 2)


if __name__ == "__main__":
    unittest.main()

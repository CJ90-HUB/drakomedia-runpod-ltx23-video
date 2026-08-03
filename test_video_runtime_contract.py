import unittest
from pathlib import Path


class RuntimeContractTests(unittest.TestCase):
    def test_requirements_match_the_official_pinned_ltx_runtime(self):
        requirements = {
            line.strip()
            for line in Path(__file__).with_name("requirements.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

        self.assertIn("transformers==4.57.6", requirements)
        self.assertIn("huggingface-hub==0.36.0", requirements)
        self.assertNotIn("huggingface-hub==0.34.4", requirements)

    def test_docker_uses_pinned_validated_video_base_and_checks_imports(self):
        dockerfile = (
            Path(__file__).with_name("Dockerfile").read_text(encoding="utf-8")
        )

        self.assertIn(
            "video-e057e1364fb353aa10c7ac653625cc245b12f581"
            "@sha256:bec5ee9f0de143ff603c299a4a0cfbf0805a91dfe5fd9c5aadfa1ffed86be53d",
            dockerfile,
        )
        self.assertNotIn("pip install", dockerfile)
        self.assertNotIn("download_models", dockerfile)
        self.assertIn("COPY handler.py video_contract.py video_only_distilled.py ./", dockerfile)


if __name__ == "__main__":
    unittest.main()

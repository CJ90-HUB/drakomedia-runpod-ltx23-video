import os
import unittest

os.environ["DRAKO_SKIP_DNS_GUARD"] = "1"

from contract import ContractError, parse_request


HOST = (
    "a76220a52aaf357ce8909685181757af."
    "r2.cloudflarestorage.com"
)


def event(**overrides):
    payload = {
        "request_id": "test-1",
        "prompt": "A still camera observes a calm lake.",
        "seed": 42,
        "width": 768,
        "height": 512,
        "frames": 49,
        "fps": 24,
        "source": {},
        "output": {
            "upload_url": f"https://{HOST}/signed",
            "object_key": "safe-to-delete/video/test/result.mp4",
        },
    }
    payload.update(overrides)
    return {"input": payload}


class ContractTests(unittest.TestCase):
    def test_valid_request(self):
        request = parse_request(event())
        self.assertEqual(request.width, 768)
        self.assertEqual(request.frames, 49)

    def test_rejects_non_multiple_resolution(self):
        with self.assertRaises(ContractError):
            parse_request(event(width=770))

    def test_rejects_invalid_frame_count(self):
        with self.assertRaises(ContractError):
            parse_request(event(frames=48))

    def test_rejects_external_storage(self):
        bad = event()
        bad["input"]["output"]["upload_url"] = (
            "https://example.com/upload"
        )
        with self.assertRaises(ContractError):
            parse_request(bad)

    def test_rejects_non_temporary_prefix(self):
        bad = event()
        bad["input"]["output"]["object_key"] = "video/result.mp4"
        with self.assertRaises(ContractError):
            parse_request(bad)


if __name__ == "__main__":
    unittest.main()

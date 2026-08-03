import os
import unittest

os.environ["DRAKO_SKIP_DNS_GUARD"] = "1"

from video_contract import (
    ContractError,
    GenerationStageError,
    parse_request,
    public_error,
)


HOST = (
    "a76220a52aaf357ce8909685181757af."
    "r2.cloudflarestorage.com"
)


def event(**overrides):
    payload = {
        "profile": "cheap",
        "request_id": "test-1",
        "prompt": "A still camera observes a calm lake.",
        "seed": 42,
        "width": 1344,
        "height": 768,
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
        self.assertEqual(request.width, 1344)
        self.assertEqual(request.frames, 49)

    def test_rejects_non_multiple_resolution(self):
        with self.assertRaises(ContractError):
            parse_request(event(width=770))

    def test_rejects_resolution_outside_profile(self):
        with self.assertRaises(ContractError):
            parse_request(event(width=1024, height=576))

    def test_ultra_cheap_plus_profile(self):
        request = parse_request(event(
            profile="ultra-cheap-plus",
            width=1024,
            height=576,
        ))
        self.assertEqual(request.profile, "ultra-cheap-plus")

    def test_best_video_only_profile(self):
        request = parse_request(event(
            profile="best-video-only",
            width=1920,
            height=1088,
        ))
        self.assertEqual(request.profile, "best-video-only")

    def test_normal_profile(self):
        request = parse_request(event(
            profile="normal",
            width=1600,
            height=896,
        ))
        self.assertEqual(request.profile, "normal")

    def test_normal_profile_accepts_vertical_orientation(self):
        request = parse_request(event(
            profile="normal",
            width=896,
            height=1600,
        ))
        self.assertEqual((request.width, request.height), (896, 1600))

    def test_profiles_preserve_editorial_inputs(self):
        prompt = "Exact prompt that must not change."
        request = parse_request(event(
            profile="ultra-cheap-plus",
            width=1024,
            height=576,
            prompt=prompt,
            seed=487066030,
            frames=113,
            fps=24,
        ))
        self.assertEqual(request.prompt, prompt)
        self.assertEqual(request.seed, 487066030)
        self.assertEqual(request.frames, 113)
        self.assertEqual(request.fps, 24)

    def test_generation_stage_is_visible_without_internal_message(self):
        result = public_error(GenerationStageError(
            "checkpoint",
            FileNotFoundError("private path"),
        ))
        self.assertEqual(result["stage"], "checkpoint")
        self.assertEqual(result["cause_type"], "FileNotFoundError")
        self.assertNotIn("private path", result["message"])

    def test_rejects_invalid_frame_count(self):
        with self.assertRaises(ContractError):
            parse_request(event(frames=48))

    def test_accepts_fifteen_second_frame_count(self):
        request = parse_request(event(frames=361, fps=24))
        self.assertEqual(request.frames, 361)

    def test_rejects_frame_count_above_fifteen_seconds(self):
        with self.assertRaises(ContractError):
            parse_request(event(frames=369, fps=24))

    def test_rejects_more_than_fifteen_seconds_at_low_fps(self):
        with self.assertRaises(ContractError):
            parse_request(event(frames=129, fps=8))

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

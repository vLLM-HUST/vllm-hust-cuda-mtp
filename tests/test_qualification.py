import unittest
from unittest.mock import patch

from vllm_hust_cuda_mtp.qualification import Completion, qualify


class QualificationTests(unittest.TestCase):
    def test_paired_smoke_passes_without_logging_output(self):
        with patch(
            "vllm_hust_cuda_mtp.qualification.completion",
            return_value=Completion("answer", "stop", 0.1, 2),
        ) as call:
            result = qualify("http://127.0.0.1:18184", "http://127.0.0.1:18185")
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["cases"]), 6)
        self.assertGreaterEqual(call.call_count, 12)
        self.assertNotIn("answer", repr(result))

    def test_mismatch_fails_closed(self):
        with patch(
            "vllm_hust_cuda_mtp.qualification.completion",
            side_effect=[
                Completion("A", "stop", 0.1, 1),
                Completion("B", "stop", 0.1, 1),
            ] * 6,
        ):
            result = qualify("http://127.0.0.1:18184", "http://127.0.0.1:18185")
        self.assertFalse(result["passed"])

    def test_external_endpoint_refused(self):
        with self.assertRaisesRegex(ValueError, "loopback"):
            qualify("https://example.com", "http://127.0.0.1:18185")


if __name__ == "__main__":
    unittest.main()

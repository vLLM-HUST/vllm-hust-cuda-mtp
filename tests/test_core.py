import unittest

from vllm_hust_cuda_mtp.core import HostReport, build_command


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.report = HostReport(
            "0.30.1rc1.dev535",
            {
                "qwen35_mtp_supports_pp": True,
                "hybrid_trailing_mtp_groups": True,
                "hybrid_exact_group_partition": True,
            },
        )
        self.baseline = [
            "vllm", "serve", "Qwen/Qwen3.5-35B-A3B",
            "--pipeline-parallel-size", "4", "--tensor-parallel-size", "1",
            "--distributed-executor-backend", "ray",
            "--host", "127.0.0.1", "--port", "18185",
            "--max-model-len", "131072", "--max-num-batched-tokens", "8192",
        ]

    def test_safe_command(self):
        command = build_command(self.baseline, self.report)
        self.assertIn("--no-enable-prefix-caching", command)
        self.assertEqual(
            command[command.index("--speculative-config") + 1],
            '{"method":"mtp","num_speculative_tokens":1}',
        )
        self.assertEqual(self.baseline[-1], "8192")

    def test_current_a6000_local_checkpoint(self):
        baseline = self.baseline.copy()
        baseline[2] = "/models/Qwen3.5-35B-A3B"
        baseline.extend(["--served-model-name", "Qwen3.5-35B-A3B"])
        self.assertIn("--speculative-config", build_command(baseline, self.report))
        baseline[-1] = "unapproved-alias"
        with self.assertRaisesRegex(ValueError, "approved served-model-name"):
            build_command(baseline, self.report)

    def test_apc_refused(self):
        with self.assertRaisesRegex(ValueError, "not qualified"):
            build_command(self.baseline, self.report, prefix_caching=True)
        with self.assertRaisesRegex(ValueError, "enables prefix caching"):
            build_command(self.baseline + ["--enable-prefix-caching"], self.report)

    def test_abi_refused(self):
        incomplete = HostReport(self.report.vllm_version, {**self.report.features, "qwen35_mtp_supports_pp": False})
        with self.assertRaisesRegex(ValueError, "host lacks"):
            build_command(self.baseline, incomplete)

    def test_nonisolated_refused(self):
        for arg, value in [
            ("--host", "0.0.0.0"),
            ("--port", "8000"),
            ("--pipeline-parallel-size", "5"),
            ("--distributed-executor-backend", "mp"),
        ]:
            candidate = self.baseline.copy()
            candidate[candidate.index(arg) + 1] = value
            with self.subTest(arg=arg, value=value), self.assertRaises(ValueError):
                build_command(candidate, self.report)

    def test_existing_speculation_and_secret_refused(self):
        with self.assertRaisesRegex(ValueError, "already sets speculative"):
            build_command(self.baseline + ["--speculative-config", "{}"], self.report)
        with self.assertRaisesRegex(ValueError, "credentials"):
            build_command(self.baseline + ["--api-key", "not-a-real-key"], self.report)


if __name__ == "__main__":
    unittest.main()

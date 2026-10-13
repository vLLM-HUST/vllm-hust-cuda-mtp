"""Non-production command line interface for the CUDA MTP experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .core import HostReport, build_command, probe_host
from .qualification import qualify


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vllm-hust-cuda-mtp")
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("probe", help="inspect local vLLM Python APIs without using GPU")
    plan = commands.add_parser("plan", help="produce isolated PP4 MTP1 launch argv")
    plan.add_argument("--baseline-argv", type=Path, required=True)
    plan.add_argument("--host-report", type=Path, required=True)
    launch = commands.add_parser("launch", help="exec an explicitly isolated MTP test server")
    launch.add_argument("--baseline-argv", type=Path, required=True)
    launch.add_argument("--host-report", type=Path, required=True)
    launch.add_argument("--acknowledge-isolated-gpu-test", action="store_true", required=True)
    canary = commands.add_parser("qualify", help="paired loopback correctness smoke test")
    canary.add_argument("--baseline-url", required=True)
    canary.add_argument("--candidate-url", required=True)
    args = parser.parse_args(argv)
    if args.action == "probe":
        report = probe_host()
        print(json.dumps({"vllm_version": report.vllm_version, "features": report.features}))
        return 0 if report.ready else 2
    if args.action == "qualify":
        result = qualify(args.baseline_url, args.candidate_url)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["passed"] else 2
    baseline = json.loads(args.baseline_argv.read_text())
    if not isinstance(baseline, list) or not all(isinstance(arg, str) for arg in baseline):
        parser.error("baseline argv must be a JSON array of strings")
    report = HostReport.from_json(json.loads(args.host_report.read_text()))
    try:
        command = build_command(baseline, report)
    except ValueError as exc:
        parser.error(str(exc))
    if args.action == "launch":
        # Execution is opt-in; the caller must separately reserve the GPUs.
        # Never invoke this against the shared production port or host binding.
        os.execvp(command[0], command)
        raise AssertionError("os.execvp unexpectedly returned")
    print(json.dumps(command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

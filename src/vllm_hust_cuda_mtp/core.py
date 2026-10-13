"""Host checks and launch planning; never patch a live vLLM process."""

from __future__ import annotations

import importlib
import inspect
import json
from dataclasses import dataclass
from typing import Sequence

MODEL = "Qwen/Qwen3.5-35B-A3B"
LOCAL_MODEL = "/models/Qwen3.5-35B-A3B"
SERVED_MODELS = {"Qwen3.5-35B-A3B", MODEL}
REQUIRED_FEATURES = (
    "qwen35_mtp_supports_pp",
    "hybrid_trailing_mtp_groups",
    "hybrid_exact_group_partition",
)


@dataclass(frozen=True)
class HostReport:
    vllm_version: str
    features: dict[str, bool]

    @property
    def ready(self) -> bool:
        return all(self.features.get(name, False) for name in REQUIRED_FEATURES)

    @classmethod
    def from_json(cls, value: object) -> "HostReport":
        if not isinstance(value, dict):
            raise ValueError("host report must be a JSON object")
        features = value.get("features")
        if not isinstance(features, dict) or not all(
            isinstance(features.get(name), bool) for name in REQUIRED_FEATURES
        ):
            raise ValueError("host report is missing boolean feature results")
        return cls(str(value.get("vllm_version", "unknown")), features)


def probe_host() -> HostReport:
    """Import only host Python APIs; no model load or GPU allocation."""
    import importlib.metadata

    features = dict.fromkeys(REQUIRED_FEATURES, False)
    try:
        model = importlib.import_module("vllm.model_executor.models.qwen3_5_mtp")
        pp = importlib.import_module("vllm.model_executor.models.interfaces")
        mtp_class = getattr(model, "Qwen3_5MTP")
        features["qwen35_mtp_supports_pp"] = getattr(pp, "SupportsPP") in mtp_class.__mro__
    except (ImportError, AttributeError, TypeError):
        pass
    try:
        kv = importlib.import_module("vllm.v1.core.kv_cache_utils")
        source = inspect.getsource(kv)
        annotation = inspect.getsource(getattr(kv, "_annotate_eagle_groups"))
        features["hybrid_trailing_mtp_groups"] = (
            "def _uses_trailing_mtp_layers" in source
            and "use_trailing_layer_fallback" in annotation
            and "group.is_eagle_group = True" in annotation
        )
        features["hybrid_exact_group_partition"] = (
            "def _groups_partition_layers_exactly" in source
            and "_groups_partition_layers_exactly(kv_cache_spec, kv_cache_groups)" in annotation
        )
    except (ImportError, AttributeError, OSError, TypeError):
        pass
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        version = "unavailable"
    return HostReport(version, features)


def _flag(argv: Sequence[str], name: str) -> str | None:
    values = [argv[index + 1] for index, item in enumerate(argv[:-1]) if item == name]
    values += [item.partition("=")[2] for item in argv if item.startswith(name + "=")]
    if len(values) > 1:
        raise ValueError(f"duplicate {name}")
    return values[0] if values else None


def build_command(
    baseline: Sequence[str], report: HostReport, *, prefix_caching: bool = False
) -> list[str]:
    """Return an isolated PP4 MTP1 command or fail closed.

    Prefix caching deliberately cannot be enabled here until the hybrid
    Mamba + MTP corruption case has a separate runtime correctness gate.
    """
    if not report.ready:
        raise ValueError("host lacks required CUDA MTP/PP/hybrid group features")
    if prefix_caching:
        raise ValueError("MTP + prefix caching is not qualified; refusing unsafe combination")
    args = list(baseline)
    if len(args) < 3 or args[:2] != ["vllm", "serve"] or args[2] not in {MODEL, LOCAL_MODEL}:
        raise ValueError("baseline must serve the approved Qwen3.5-35B-A3B model")
    if args[2] == LOCAL_MODEL and _flag(args, "--served-model-name") not in SERVED_MODELS:
        raise ValueError("local checkpoint requires an approved served-model-name")
    if _flag(args, "--pipeline-parallel-size") != "4" or _flag(
        args, "--tensor-parallel-size"
    ) != "1":
        raise ValueError("this profile requires PP4 x TP1")
    if _flag(args, "--distributed-executor-backend") != "ray":
        raise ValueError("this four-host profile requires the Ray executor")
    if _flag(args, "--host") != "127.0.0.1":
        raise ValueError("experimental service must bind loopback only")
    port = _flag(args, "--port")
    if port is None or not port.isdecimal() or not 18000 <= int(port) <= 18999:
        raise ValueError("experimental service must use an 18xxx port")
    if _flag(args, "--speculative-config") is not None:
        raise ValueError("baseline already sets speculative decoding")
    if "--enable-prefix-caching" in args or any(
        item.startswith("--enable-prefix-caching=") for item in args
    ):
        raise ValueError("baseline enables prefix caching")
    if "--no-enable-prefix-caching" in args:
        raise ValueError("baseline already disables prefix caching")
    secret_flags = ("--api-key", "--token", "--password", "--hf-token")
    if any(item == flag or item.startswith(flag + "=") for item in args for flag in secret_flags):
        raise ValueError("command may contain credentials; refuse to handle it")
    args.extend(
        [
            "--no-enable-prefix-caching",
            "--speculative-config",
            json.dumps({"method": "mtp", "num_speculative_tokens": 1}, separators=(",", ":")),
        ]
    )
    return args

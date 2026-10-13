"""Fail-closed paired completion canary; no responses or secrets are logged."""

from __future__ import annotations

import concurrent.futures
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

from .core import MODEL

CASES = (
    ("short", "请只回答：校验通过。", 48),
    (
        "repeated-prefix",
        "系统背景：研究 vLLM 推理引擎的 KV 缓存和投机解码。" * 24
        + "\n问题：解释为什么候选 token 仍需目标模型验证。",
        192,
    ),
    (
        "long-decode",
        "用中文分六点解释流水线并行的吞吐与延迟权衡，每点写两句。",
        384,
    ),
)


@dataclass(frozen=True)
class Completion:
    text: str
    finish_reason: str
    seconds: float
    output_tokens: int


def _require_loopback(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("qualification endpoints must be loopback HTTP")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("endpoint must be an origin URL without a path or query")
    return url.rstrip("/")


def completion(origin: str, prompt: str, max_tokens: int) -> Completion:
    payload = json.dumps(
        {
            "model": MODEL,
            "prompt": prompt,
            "temperature": 0,
            "seed": 42,
            "max_tokens": max_tokens,
            "stream": False,
        }
    ).encode()
    request = urllib.request.Request(
        _require_loopback(origin) + "/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        document = json.load(response)
    elapsed = time.monotonic() - started
    choice = document["choices"][0]
    return Completion(
        str(choice["text"]),
        str(choice["finish_reason"]),
        elapsed,
        int(document.get("usage", {}).get("completion_tokens", -1)),
    )


def qualify(baseline_url: str, candidate_url: str) -> dict[str, object]:
    """Paired exact-output canary, not a production performance proof."""
    baseline_url = _require_loopback(baseline_url)
    candidate_url = _require_loopback(candidate_url)
    if baseline_url == candidate_url:
        raise ValueError("baseline and candidate must be distinct endpoints")
    results: list[dict[str, object]] = []
    for name, prompt, limit in CASES:
        baseline = completion(baseline_url, prompt, limit)
        candidate = completion(candidate_url, prompt, limit)
        results.append(
            {
                "case": name,
                "exact_output": baseline.text == candidate.text,
                "nonempty": bool(baseline.text.strip()) and bool(candidate.text.strip()),
                "finished": baseline.finish_reason == candidate.finish_reason == "stop",
                "baseline_seconds": round(baseline.seconds, 3),
                "candidate_seconds": round(candidate.seconds, 3),
                "baseline_output_tokens": baseline.output_tokens,
                "candidate_output_tokens": candidate.output_tokens,
            }
        )
    # Second identical long-prefix request tests reuse behavior; no hit claim is
    # made because this MOD deliberately turns prefix caching off.
    _, prompt, limit = CASES[1]
    repeat_a = completion(baseline_url, prompt, limit)
    repeat_b = completion(candidate_url, prompt, limit)
    results.append(
        {
            "case": "repeated-prefix-second-pass",
            "exact_output": repeat_a.text == repeat_b.text,
            "nonempty": bool(repeat_a.text.strip()) and bool(repeat_b.text.strip()),
            "finished": repeat_a.finish_reason == repeat_b.finish_reason == "stop",
        }
    )
    # Concurrent requests catch some request-state contamination bugs. Each
    # pair is compared only with its own exact prompt.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending = [
            pool.submit(completion, origin, f"并发校验 {index}：只回复数字 {index}", 32)
            for index in range(2)
            for origin in (baseline_url, candidate_url)
        ]
        concurrent_results = [future.result() for future in pending]
    for index in range(2):
        baseline, candidate = concurrent_results[index * 2 : index * 2 + 2]
        results.append(
            {
                "case": f"concurrent-{index}",
                "exact_output": baseline.text == candidate.text,
                "nonempty": bool(baseline.text.strip()) and bool(candidate.text.strip()),
                "finished": baseline.finish_reason == candidate.finish_reason == "stop",
            }
        )
    passed = all(
        item["exact_output"] and item["nonempty"] and item["finished"]
        for item in results
    )
    return {"passed": passed, "cases": results, "scope": "smoke-only-no-apc-or-performance-claim"}

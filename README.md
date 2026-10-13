# CUDA MTP for Qwen3.5 (experimental)

This MOD is a fail-closed activation and correctness carrier for the **native**
vLLM MTP path on Qwen/Qwen3.5-35B-A3B, PP4 × TP1 CUDA. It does not implement a
new drafter and it does not patch a running vLLM process. It is deliberately
separate from the Ascend-only vSpec runtime.

Current status: source and CPU tests pass, and a temporary PP4 MTP1 instance
was launched and tested on the A6000 fleet on 2026-10-13. It was then stopped
and the original service restored. Synthetic decode throughput improved, but
the candidate lost 34% of KV-cache capacity and one fixed long answer differed
from the baseline. **This is not production-qualified.** See
[`experiments/a6000-pp4/RESULTS.md`](experiments/a6000-pp4/RESULTS.md).

## What it does

- `probe` checks the installed Qwen3.5 MTP class implements `SupportsPP` and
  checks that the host KV-cache source contains the trailing-drafter annotation
  and exact-group-partition guard. This is an ABI smoke check, not a runtime
  correctness proof.
- `plan` accepts a JSON array of baseline `vllm serve` arguments and a probe
  report, then emits a separate loopback-only, port 18xxx command with MTP1.
- `launch` runs that command only with an explicit isolated-test acknowledgement.
- `qualify` compares baseline and candidate `/v1/chat/completions` responses
  with native thinking disabled, using fixed short, repeated-prefix,
  long-decode, concurrent and cancel/recover prompts. It emits
  only metadata, not prompts or model answers. Any mismatch fails.

The launcher **forces prefix caching off** for MTP. A known hybrid Mamba/MTP
correctness issue remains open upstream, and the earlier A100 overlay also
lost prefix-cache hits under MTP. This conservative switch is only for an
isolated test; it is not an acceptable production workaround if workload
performance depends on APC. Cache-on requires a separate fix and full
correctness/cancellation/cache-hit qualification before support is added.

## Example isolated workflow

Run `probe` in the *same image* as the candidate engine; save its JSON report
in the test workspace. Prepare a JSON array of baseline `vllm serve` arguments
for an allocated PP4 × TP1 test cell. The array must bind `127.0.0.1` to a
port from 18000–18999, use the Ray executor, identify either
`Qwen/Qwen3.5-35B-A3B` or the verified
A6000 checkpoint path `/models/Qwen3.5-35B-A3B` (the latter also requires an
approved `--served-model-name`), and contain no credentials. Do not point it at
a live service's port or GPUs.

```bash
vllm-hust-cuda-mtp probe
vllm-hust-cuda-mtp plan --baseline-argv baseline.json --host-report host.json
vllm-hust-cuda-mtp launch --baseline-argv baseline.json --host-report host.json --acknowledge-isolated-gpu-test
vllm-hust-cuda-mtp qualify --baseline-url http://127.0.0.1:18184 --candidate-url http://127.0.0.1:18185
```

`qualify` is a correctness smoke test only: it does not establish statistical
quality, prefix-cache compatibility, cancellation safety, draft acceptance, or
throughput benefit. The maintenance test above provides a single ON/OFF
synthetic comparison, not repeated or cache-on production evidence. Do not
enable this on the shared A6000 PP4 service on the strength of that result.

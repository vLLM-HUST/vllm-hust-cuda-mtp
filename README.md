# CUDA MTP for Qwen3.5 (experimental)

This MOD is a fail-closed activation and correctness carrier for the **native**
vLLM MTP path on Qwen/Qwen3.5-35B-A3B, PP4 × TP1 CUDA. It does not implement a
new drafter and it does not patch a running vLLM process. It is deliberately
separate from the Ascend-only vSpec runtime.

Current status: source and CPU tests pass; the three required host APIs were
read-only checked in the A6000 image. **No PP4 MTP model has been launched, no
paired GPU canary has passed, and this is not production-qualified.**

## What it does

- `probe` checks the installed Qwen3.5 MTP class implements `SupportsPP` and
  checks that the host KV-cache source contains the trailing-drafter annotation
  and exact-group-partition guard. This is an ABI smoke check, not a runtime
  correctness proof.
- `plan` accepts a JSON array of baseline `vllm serve` arguments and a probe
  report, then emits a separate loopback-only, port 18xxx command with MTP1.
- `launch` runs that command only with an explicit isolated-test acknowledgement.
- `qualify` compares baseline and candidate `/v1/completions` responses on
  fixed short, repeated-prefix, long-decode and concurrent prompts. It emits
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
port from 18000–18999, identify exactly `Qwen/Qwen3.5-35B-A3B`, and contain
no credentials. Do not point it at a live service's port or GPUs.

```bash
vllm-hust-cuda-mtp probe
vllm-hust-cuda-mtp plan --baseline-argv baseline.json --host-report host.json
vllm-hust-cuda-mtp launch --baseline-argv baseline.json --host-report host.json --acknowledge-isolated-gpu-test
vllm-hust-cuda-mtp qualify --baseline-url http://127.0.0.1:18184 --candidate-url http://127.0.0.1:18185
```

`qualify` is a correctness smoke test only: it does not establish statistical
quality, prefix-cache compatibility, cancellation safety, draft acceptance, or
throughput benefit. Those require a separately authorized GPU test with ON/OFF
matched workloads, metrics, and repeated runs. Do not enable this on the
shared A6000 PP4 service on the strength of the static probe or smoke test.

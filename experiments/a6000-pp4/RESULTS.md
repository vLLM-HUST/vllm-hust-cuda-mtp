# A6000 PP4 MTP1 maintenance experiment — 2026-10-13

Status: **experimental; not admitted to the shared service**. The user
authorized a reversible short maintenance test. The original Docker head
container was stopped but preserved, a temporary loopback-only MTP1 head was
run on the same four Ray workers, and the original container was restarted.
No other model or router configuration was changed.
The included launcher uses `MTP_HEAD_IP` rather than publishing the lab's
internal Ray address; provide it only in the authorized test environment.

## Configuration and functional checks

Both cells used the same Qwen3.5-35B-A3B checkpoint, four RTX A6000 hosts,
PP4 × TP1, 131,072-token context, and GPU memory utilization 0.90. The OFF
cell had the existing automatic prefix cache on; the ON cell used native vLLM
MTP with one speculative token and deliberately disabled prefix caching due
to unresolved hybrid Mamba/MTP correctness risk. Thus this comparison tests
the safe candidate configuration, **not an MTP-only toggle**.

The ON instance loaded `Qwen3_5MoeMTP`, joined four Ray nodes/four GPUs, and
returned HTTP 200 on `/health`. Its self-control canary passed 7/7 cases:
short reply, repeated prefix, long decode, two concurrent replies, and
stream-cancel/recovery. This is not a baseline-versus-candidate paired pass:
the two full PP4 instances could not coexist on the allocated GPUs. Fixed
non-thinking, temperature-zero short/prefix replies matched the saved OFF
hashes, but one 314-token OFF long answer became a different 315-token ON
answer. Both finished normally. Therefore exact answer equivalence failed
for that prompt; quality equivalence is unproven.

ON draft metrics after the canary showed 603 drafts and 387 accepted (64.2%)
across mixed prompts. APC queries/hits were zero by design; preemptions were
zero. The OFF instance also had zero cache hits in this new epoch, so these
observations do not isolate an MTP-caused hit-rate loss.

## Matched synthetic serving benchmark

The installed vLLM `bench serve` client sent random 128-input/256-output
token requests, seed 42, temperature 0, `ignore_eos`, to each cell in sequence.
All requests completed and each measured 256 output tokens. Values below
are one run per cell; ON was measured before restored OFF.

| Concurrency / requests | ON output tok/s | OFF output tok/s | ON mean TTFT | OFF mean TTFT | ON mean TPOT | OFF mean TPOT | ON draft acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C4 / 12 | 159.85 | 120.69 | 263.77 ms | 275.09 ms | 23.97 ms | 32.17 ms | 98.90% (1525/1542) |
| C1 / 6 | 91.40 | 56.27 | 178.53 ms | 182.10 ms | 10.28 ms | 17.12 ms | 94.55% (746/789) |

Observed output throughput was +32.4% at C4 and +62.4% at C1 versus the
restored OFF run. A separate **unpaired** ON run at the server-default
temperature delivered 107.79 output tok/s at C4 with 56.68% acceptance;
sampling settings materially affect acceptance and potential benefit.
These forced-length random requests do not establish answer quality,
real-user gains, or statistical confidence. Run order, warm state, and live
traffic were not controlled by repeated randomized trials.

## Capacity and disposition

ON KV-cache capacity was 2,203,210 tokens (16.81 theoretical concurrent
131K requests), versus 3,344,837 tokens (25.52×) after OFF restoration: a
34.1% capacity reduction. APC remains unavailable in the safe ON profile.
Those losses, the differing long answer, and single-run evidence preclude
production enablement despite the synthetic throughput improvement.

After the test, the temporary head was stopped and the original
`vllm-hust-ray` head restarted. Restoration checks: `/health` HTTP 200,
`speculative_config=None`, `enable_prefix_caching=True`, 3,344,837 KV-cache
tokens, and a real non-thinking chat request returned the exact requested
`restore-mtp-ok` text with `finish_reason=stop`. The worker fleet rejoined.

Next gate: isolate the long-answer difference, repair/qualify hybrid
Mamba + MTP + APC, then run repeated, order-rotated, realistic mixed
prefill/decode and cache-hit workloads with output-quality checks. No
automatic production switch is warranted from this experiment.

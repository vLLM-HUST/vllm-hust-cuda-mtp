#!/usr/bin/env bash
# Isolated maintenance test only. The original vllm-hust-ray container is rollback.
set -euo pipefail

head_ip=${MTP_HEAD_IP:?Set MTP_HEAD_IP to the allocated Ray head address}
ray start \
  --head \
  --node-ip-address="${head_ip}" \
  --num-cpus=8 \
  --num-gpus=1 \
  --port=6379 \
  --include-dashboard=false

for attempt in $(seq 1 120); do
  if python3 - <<'PY'
import ray
ray.init(address="auto", logging_level="ERROR")
resources = ray.cluster_resources()
gpus = int(resources.get("GPU", 0))
nodes = sum(1 for node in ray.nodes() if node.get("Alive"))
raise SystemExit(0 if nodes >= 4 and gpus >= 4 else 1)
PY
  then
    break
  fi
  if [[ "${attempt}" == 120 ]]; then
    echo "Timed out waiting for four Ray nodes/GPUs" >&2
    exit 3
  fi
  sleep 5
done

export PYTHONPATH=/opt/vllm-hust/cuda-mtp.whl
exec python3 - <<'PY'
import os
from vllm_hust_cuda_mtp.core import build_command, probe_host

baseline = [
    "vllm", "serve", "/models/Qwen3.5-35B-A3B",
    "--served-model-name", "Qwen3.5-35B-A3B", "Qwen/Qwen3.5-35B-A3B",
    "--host", "127.0.0.1", "--port", "18185",
    "--tensor-parallel-size", "1", "--pipeline-parallel-size", "4",
    "--distributed-executor-backend", "ray",
    "--max-model-len", "131072", "--gpu-memory-utilization", "0.90",
    "--reasoning-parser", "qwen3", "--enable-auto-tool-choice",
    "--tool-call-parser", "qwen3_coder",
]
command = build_command(baseline, probe_host())
os.execvp(command[0], command)
PY

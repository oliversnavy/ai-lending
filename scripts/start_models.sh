#!/bin/bash
# Start both vLLM model servers
# Primary: Qwen3.6-35B-A3B-NVFP4 on port 8000
# Advisor: Qwen3.6-27B-NVFP4 on port 8001
#
# --gpu-memory-utilization guidance:
#   0.45 / 0.38  →  ~115GB used, ~6GB free   (max KV cache, experiment runs only)
#   0.35 / 0.28  →  ~82GB used,  ~28GB free  (default, leaves room for data workload)
#   0.25 / 0.20  →  ~55GB used,  ~55GB free  (lightweight, other heavy projects running)

GPU_MEM_PRIMARY=${1:-0.40}
GPU_MEM_ADVISOR=${2:-0.32}

echo "Starting primary with gpu-memory-utilization=${GPU_MEM_PRIMARY}"
docker run --gpus all -d \
  --name vllm-primary \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  nvcr.io/nvidia/vllm:26.04-py3 \
  vllm serve unsloth/Qwen3.6-35B-A3B-NVFP4 \
  --tokenizer Qwen/Qwen3.6-35B-A3B \
  --dtype bfloat16 \
  --max-model-len 32768 --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization "${GPU_MEM_PRIMARY}"

echo "Starting advisor with gpu-memory-utilization=${GPU_MEM_ADVISOR}"
docker run --gpus all -d \
  --name vllm-advisor \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8001:8000 \
  nvcr.io/nvidia/vllm:26.04-py3 \
  vllm serve unsloth/Qwen3.6-27B-NVFP4 \
  --tokenizer Qwen/Qwen3.6-27B \
  --dtype bfloat16 \
  --max-model-len 32768 --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization "${GPU_MEM_ADVISOR}"

echo "Waiting for models to load..."
until docker logs vllm-primary 2>&1 | grep -q "Application startup complete"; do sleep 5; done
echo "Primary ready on :8000"
until docker logs vllm-advisor 2>&1 | grep -q "Application startup complete"; do sleep 5; done
echo "Advisor ready on :8001"

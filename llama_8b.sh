export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MODEL_PATH="${MODEL_PATH:-./models/Llama-3.1-8B-Instruct}"
PORT="${PORT:-6010}"

python -m vllm.entrypoints.openai.api_server \
--served-model-name llama3_8b \
--model "${MODEL_PATH}" \
--gpu-memory-utilization 0.95 \
--tensor-parallel-size 4 \
--max-model-len 4096 \
--port "${PORT}"

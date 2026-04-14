export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MODEL_PATH="${MODEL_PATH:-./models/Llama-3.3-70B-Instruct}"
PORT="${PORT:-6010}"

python -m vllm.entrypoints.openai.api_server \
--served-model-name llama3_70b \
--model "${MODEL_PATH}" \
--gpu-memory-utilization 0.9 \
--tensor-parallel-size 8 \
--max-model-len 8192 \
--port "${PORT}"

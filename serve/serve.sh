vllm serve  meta-llama/Meta-Llama-3.1-8B-Instruct --port 1234 --chat-template llama3.jinja \
    --served-model-name Meta-Llama-3.1-8B-Instruct \

python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-405B-Instruct-FP8 --tp 8

python -m sglang.launch_server --model-path mistralai/Mistral-Large-Instruct-2407 --tp 8
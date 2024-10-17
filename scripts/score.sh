python -m src.rank \
    --data_path results_synthetic/SlimOrca-Dedup_rejection-sampling_0-20000.json \
    --output_path results_rank/SlimOrca-Dedup_rejection-sampling_0-20000.json


python -m src.rank \
    --data_path results_synthetic/Magpie-Reasoning_rejection-sampling_0-20000.json \
    --output_path results_rank/Magpie-Reasoning_rejection-sampling_0-20000.json

# slm-research-vn/slm-instruct-synthetic-v0.1
python -m src.rank \
    --data_path results_synthetic/slm-instruct-synthetic-v0.1_rejection-sampling_0-20000.json \
    --output_path results_rank/slm-instruct-synthetic-v0.1_rejection-sampling_0-20000.json

# slm-research-vn/slm-instruct-synthetic-v0.2
python -m src.rank \
    --data_path results_synthetic/slm-instruct-synthetic-v0.2_rejection-sampling_0-20000.json \
    --output_path results_rank/slm-instruct-synthetic-v0.2_rejection-sampling_0-20000.json

# slm-research-vn/Magpie-Llama-3.1-Pro-MT
python -m src.rank \
    --data_path results_synthetic/Magpie-Llama-3.1-Pro-MT_rejection-sampling_0-20000.json \
    --output_path results_rank/Magpie-Llama-3.1-Pro-MT_rejection-sampling_0-20000.json


python -m src.rank \
    --data_path results_synthetic/slm-code-synthetic-v0.1_rejection-sampling_0-50000.json \
    --output_path results_rank/slm-code-synthetic-v0.1_rejection-sampling_0-50000.json
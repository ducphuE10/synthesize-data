# python -m src.gen_rs \
#     --data_path slm-research-vn/slm-code-synthetic-v0.1 --start 0 --offset 25000 \
#     --output_path results_synthetic/slm-code-synthetic-v0.1_rejection-sampling_0-25000.json


# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m src.gen_rs \
#     --data_path slm-research-vn/slm-code-synthetic-v0.1 --start 25000 --offset 25000 \
#     --output_path results_synthetic/slm-code-synthetic-v0.1_rejection-sampling_25000-50000.json


# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m src.gen_rs \
#     --data_path slm-research-vn/slm-code-synthetic-v0.1 --start 0 --offset 20 \
#     --output_path results_synthetic/slm-code-synthetic-v0.1_rejection-sampling_0-20.json


python -m src.gen_rs \
    --data_path slm-research-vn/SlimOrca-Dedup --start 0 --offset 20000 --n 8 \
    --output_path results_synthetic/SlimOrca-Dedup_rejection-sampling_0-20000.json


python -m src.gen_rs \
    --data_path slm-research-vn/Magpie-Reasoning --start 0 --offset 20000 --n 8 \
    --output_path results_synthetic/Magpie-Reasoning_rejection-sampling_0-20000.json

# slm-research-vn/slm-instruct-synthetic-v0.1
python -m src.gen_rs \
    --data_path slm-research-vn/slm-instruct-synthetic-v0.1 --start 0 --offset 20000 --n 8 \
    --output_path results_synthetic/slm-instruct-synthetic-v0.1_rejection-sampling_0-20000.json

# slm-research-vn/slm-instruct-synthetic-v0.2
python -m src.gen_rs \
    --data_path slm-research-vn/slm-instruct-synthetic-v0.2 --start 0 --offset 20000 --n 8 \
    --output_path results_synthetic/slm-instruct-synthetic-v0.2_rejection-sampling_0-20000.json

# slm-research-vn/Magpie-Llama-3.1-Pro-MT
python -m src.gen_rs \
    --data_path slm-research-vn/Magpie-Llama-3.1-Pro-MT --start 0 --offset 20000 --n 8 \
    --output_path results_synthetic/Magpie-Llama-3.1-Pro-MT_rejection-sampling_0-20000.json
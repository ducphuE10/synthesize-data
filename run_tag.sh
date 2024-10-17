python tag_magpie/wrapper.py \
    --model google/gemma-2-27b-it \
    --data_path Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered \
    --instruction_column instruction \
    --output_dir results_tag/Magpie-Llama-3.1-Pro-MT-300K-Filtered \
    --mission quality,difficulty \
    --gpu_ids 0
python tag_magpie/wrapper.py \
    --model google/gemma-2-27b-it \
    --data_path UCLA-AGI/data-mistral-7b-instruct-sppo-iter1 \
    --instruction_column prompt \
    --output_dir results_tag/sppo-iter1 \
    --gpu_ids 0 \

python tag_magpie/wrapper.py \
    --model google/gemma-2-27b-it \
    --data_path UCLA-AGI/data-mistral-7b-instruct-sppo-iter2 \
    --instruction_column prompt \
    --output_dir results_tag/sppo-iter2 \
    --gpu_ids 0 

python tag_magpie/wrapper.py \
    --model google/gemma-2-27b-it \
    --data_path UCLA-AGI/data-mistral-7b-instruct-sppo-iter3 \
    --instruction_column prompt \
    --output_dir results_tag/sppo-iter3 \
    --gpu_ids 0 
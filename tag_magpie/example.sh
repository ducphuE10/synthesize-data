python main.py --data_path collect_data/data/combined_ds_raw --start 0 --offset 100 --output_dir results


python wrapper.py --data_path slm-research-vn/slm-instruct-v0.1 --output_dir results --gpu_ids "0,1,2,3,4,5,6,7"


python tag/tag_quality_v2.py --data_path slm-research-vn/slm-instruct-v0.1 --start 0 --offset 20 --output_dir results_v2 --batch_size 8

python tag/wrapper.py --data_path argilla/magpie-ultra-v0.1 --output_dir results_v2 --gpu_ids "0,1,2,3,4,5,6,7" --batch_size 8
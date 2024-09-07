import os
import subprocess
import argparse

from datasets import load_from_disk, load_dataset


def get_data(path):
    try:
        data = load_from_disk(path)
    except FileNotFoundError:
        data = load_dataset(path, split="train")

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--instruction_column", type=str, default="instruction")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--mission", type=str, default="all")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--test_mode", action="store_true")

    args = parser.parse_args()

    # parse the gpu ids
    gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpu_ids.split(",")]

    # get the data
    data = get_data(args.data_path)
    if args.test_mode:
        data = data.select(range(100))
        
    num_samples = len(data)

    # each gpu will process a subset of the data
    chunk_size = num_samples // len(gpu_ids)

    runs = []
    for i, gpu_id in enumerate(gpu_ids):
        start = i * chunk_size
        offset = chunk_size if i < len(gpu_ids) - 1 else num_samples - start

        command = f"CUDA_VISIBLE_DEVICES={gpu_id} python tag_magpie/main.py --model {args.model} --data_path {args.data_path} --instruction_column {args.instruction_column} --start {start} --offset {offset} --mission {args.mission} --output_dir {args.output_dir}"

        run = subprocess.Popen(command, shell=True)

        runs.append(run)

    for run in runs:
        run.wait()

import os
import subprocess
import argparse

from datasets import load_from_disk, load_dataset, Dataset, concatenate_datasets


def load_data(path, start=None, offset=None):
    if path.endswith(".json"):
        data = Dataset.from_json(path)
    elif path.endswith(".csv"):
        data = Dataset.from_csv(path)
    elif path.endswith(".parquet"):
        data = Dataset.from_parquet(path)
    else:
        try:
            data = load_from_disk(path)
        except FileNotFoundError:
            data = load_dataset(path, split="train")

    if start:
        data = data.select(range(start, len(data)))
    if offset:
        data = data.select(range(0, offset))

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_model", type=str, default="Skywork/Skywork-Reward-Gemma-2-27B")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--input_column", type=str, default="candidates")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_mode", action="store_true")
    

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # parse the gpu ids
    gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpu_ids.split(",")]

    # get the data
    data = load_data(args.data_path)
    if args.test_mode:
        data = data.select(range(100))
        
    num_samples = len(data)

    # each gpu will process a subset of the data
    chunk_size = num_samples // len(gpu_ids)

    runs = []
    output_paths = []
    for i, gpu_id in enumerate(gpu_ids):
        start = i * chunk_size
        offset = chunk_size if i < len(gpu_ids) - 1 else num_samples - start
        output_path = os.path.join(args.output_dir, f"output_{i}.json")

        command = f"CUDA_VISIBLE_DEVICES={gpu_id} python -m src.rank --reward_model {args.reward_model} --data_path {args.data_path} --input_column {args.input_column} --start {start} --offset {offset} --output_path {output_path} --batch_size {args.batch_size}"

        
        run = subprocess.Popen(command, shell=True)

        runs.append(run)
        output_paths.append(output_path)

    for run in runs:
        try:
            run.wait()
        except Exception as e:
            raise e
        
        
    # if successful, merge the outputs
    output_data = concatenate_datasets([Dataset.from_json(output_path) for output_path in output_paths])
    
    output_path = os.path.join(args.output_dir, "output.json")
    output_data.to_json(output_path)

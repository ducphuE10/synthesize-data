import numpy as np
import warnings
import argparse
import os

from src import utils
from typing import List, Callable, Dict
from datasets import load_from_disk, load_dataset, Dataset

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


def prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["dpo", "sppo", "rso"])
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--input_column", type=str, default="candidates")
    parser.add_argument("--score_column", type=str, default="reward_scores")
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--offset", type=int, default=None)
    parser.add_argument("--hf_hub", type=str, default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token", type=str, default=None)
    
    
    args = parser.parse_args()
    data = load_data(args.data_path, args.start, args.offset)
    scores = data[args.score_column]
    candidates = data[args.input_column]
    
    chosen_indices = [np.argmax(score) for score in scores]
    rejected_indices = [np.argmin(score) for score in scores]
    
    chosen_candidates = [choices[i] for i, choices in zip(chosen_indices, candidates)]
    rejected_candidates = [choices[i] for i, choices in zip(rejected_indices, candidates)]
    
    if args.task == "dpo":
        data = data.add_column("chosen", chosen_candidates)
        data = data.add_column("rejected", rejected_candidates)
    
    elif args.task == "sppo":
        pass
    
    elif args.task == "rso":
        data = data.add_column("messages", chosen_candidates)
        data = data.add_column("conversations", [utils.fmt_openAI_to_shareGPT(messages) for messages in chosen_candidates])
    
    if args.hf_hub:
        data.push_to_hub(args.hf_hub, private=args.private, token=args.token)
        return
    
    data.to_json(args.output_path)
    
if __name__ == '__main__':
    prepare()
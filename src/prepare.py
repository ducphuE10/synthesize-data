import torch
import llm_blender
import numpy as np
import warnings
import argparse
import os

from src import utils
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
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
    args = ""
    scores = []
    candidates = []
    
    chosen_indices = np.argmax(scores, axis=1)
    rejected_indices = np.argmin(scores, axis=1)
    
    chosen_candidates = [choices[i] for i, choices in zip(chosen_indices, candidates)]
    rejected_candidates = [choices[i] for i, choices in zip(rejected_indices, candidates)]
    
    if args.task == "dpo":
        data = data.add_column("chosen", chosen_candidates)
        data = data.add_column("rejected", rejected_candidates)
    
    elif args.task == "sppo":
        pass
    
    elif args.task == "rft":
        data = data.add_column("messages", chosen_candidates)
    
    data.to_json(args.output_path)
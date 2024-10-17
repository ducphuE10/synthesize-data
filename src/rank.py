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


class BaseRanker:
    def __init__(self):
        pass

    def rank(
        self, candidates: List[List[Dict]], batch_size: int = 8
    ):
        pass


class PairRM(BaseRanker):
    def __init__(self):
        super().__init__()
        self.blender = llm_blender.Blender()
        self.blender.loadranker("llm-blender/PairRM")

    def rank(
        self, candidates: List[List[Dict]], batch_size: int = 8
    ):
        
        prompts = [
            utils.get_instruction_openAI(c[0]) for c in candidates
            ]
        
        responses = [
            [msg[-1]["content"] for msg in c] for c in candidates
            ] # list of list of responses, i.e. [[rsp1, rsp2, ...], [rsp1, rsp2, ...], ...]
        
        assert len(prompts) == len(responses), f"Length of prompts and responses must be the same {len(prompts)} != {len(responses)}"
        for i, rsp in enumerate(responses):
            if len(rsp) != len(responses[0]):
                print(i)
                print(rsp)
                exit()
        
        ranks = self.blender.rank(
            prompts, responses, return_scores=True, batch_size=batch_size
        )
        
        return list(ranks)


class SeqClsRM(BaseRanker):
    def __init__(self, model_name: str):
        super().__init__()
        self.device = "cuda"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            num_labels=1,
            trust_remote_code=True,
            device_map=self.device,
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def rank(
        self, candidates: List[List[Dict]], batch_size: int = 8
    ):
        flatten_messages = []
        for messages in candidates:
            flatten_messages.extend(messages)
            
        inputs = [
            self.tokenizer.apply_chat_template(msg, tokenize=False) for msg in flatten_messages
        ]

        outputs = []
        for i in tqdm(range(0, len(inputs), batch_size)):
            inputs_batch = inputs[i : i + batch_size]
            inputs_batch = self.tokenizer(
                inputs_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs_batch).logits
                outputs.extend([s[0].item() for s in logits])

        scores = []
        lst_len_candidates = [len(c) for c in candidates]
        for l in lst_len_candidates:
            scores.append(outputs[:l])
            outputs = outputs[l:]

        return scores
    
def load_reward_model(model_name: str) -> BaseRanker:
    if model_name == "llm-blender/PairRM":
        return PairRM()
    else:
        return SeqClsRM(model_name)
    
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
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_model", type=str, default="Skywork/Skywork-Reward-Gemma-2-27B")
    
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--input_column", type=str, default="candidates")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--offset", type=int, default=None)
    
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    # parser.add_argument("--task", type=str, choices=["dpo", "sppo", "rft"])
    
    args = parser.parse_args()
    
    if os.path.exists(args.output_path):
        raise FileExistsError(f"Output file {args.output_path} already")
    
    # create directory if not exists
    dir_path = os.path.dirname(args.output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    reward_model = load_reward_model(args.reward_model)
    
    data = load_data(args.data_path, args.start, args.offset)
    candidates = data[args.input_column]
    
    scores = reward_model.rank(candidates, args.batch_size)
    data = data.add_column("reward_scores", scores)
    data = data.add_column("reward_model", [args.reward_model] * len(data))
    
    data.to_json(args.output_path)
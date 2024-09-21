import os
import argparse
import json

from vllm import LLM, SamplingParams
from datasets import load_from_disk, load_dataset
from loguru import logger
from transformers import AutoTokenizer


def get_data(path, start=None, offset=None):
    try:
        data = load_from_disk(path)
    except FileNotFoundError:
        data = load_dataset(path, split="train")
        
    if start:
        data = data.select(range(start, len(data)))
    if offset:
        data = data.select(range(0, offset))

    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-Large-Instruct-2407")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--instruction_column", type=str, default="instruction")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--offset", type=int, default=None)
    parser.add_argument("--output_path", type=str)
    
    args = parser.parse_args()
    if os.path.exists(args.output_path):
        raise FileExistsError(f"Output file {args.output_path} already")
    # create directory if not exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    data = get_data(args.data_path, args.start, args.offset)
    lst_instructions = data[args.instruction_column]
    
    
    llm = LLM(model=args.model_name, max_model_len=4096, tensor_parallel_size=8)
    params = SamplingParams(
        temperature=0.7,
        max_tokens=2048,
        top_p=0.9,
        stop=["</s>"],
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    logger.info(f"Processing {len(lst_instructions)} instructions")
    
    lst_user_messages = [
            [{"role": "user", "content": i}] for i in lst_instructions
        ]
    
    inputs = [
            tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in lst_user_messages
        ]
    
    outputs = llm.generate(inputs, params)
    
    results = []
    for i in range(len(outputs)):
        model_response = outputs[i].outputs[0].text.strip()
        fmt_messages = [
            {"role": "user", "content": lst_instructions[i]},
            {"role": "assistant", "content": model_response},
        ]
        fmt_shareGPT = [
            {"from": "human", "value": lst_instructions[i]},
            {"from": "gpt", "value": model_response},
        ]
        
        item = {
            "instruction": lst_instructions[i],
            "response": model_response,
            "messages": fmt_messages,
            "conversations": fmt_shareGPT,
        }
        results.append(item)
        
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)
            

if __name__ == "__main__":
    main()
            
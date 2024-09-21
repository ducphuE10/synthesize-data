import os
import argparse
import json
import warnings

from src import utils
from vllm import LLM, SamplingParams
from datasets import load_from_disk, load_dataset
from loguru import logger
from transformers import AutoTokenizer


def load_data(path, start=None, offset=None):
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
    # model parameters
    parser.add_argument(
        "--model_name", type=str, default="mistralai/Mistral-Large-Instruct-2407"
    )
    parser.add_argument("--tp", type=int, default=1)

    # generation parameters
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--n", type=int, default=1)

    # data parameters
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--input_column", type=str, default="instruction")
    parser.add_argument(
        "--input_type", type=str, choices=["str", "openai", "sharegpt"], default="str"
    )
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--offset", type=int, default=None)
    parser.add_argument("--output_path", type=str)

    args = parser.parse_args()

    if os.path.exists(args.output_path):
        raise FileExistsError(f"Output file {args.output_path} already")
    # create directory if not exists
    dir_path = os.path.dirname(args.output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    data = load_data(args.data_path, args.start, args.offset)

    if args.input_type == "str":
        lst_user_messages = [
            [{"role": "user", "content": i}] for i in data[args.instruction_column]
        ]
    elif args.input_type == "openai":
        lst_user_messages = [
            utils.get_user_messages_openAI(i) for i in data[args.input_column]
        ]
    elif args.input_type == "sharegpt":
        lst_user_messages = [
            utils.get_user_messages_shareGPT(i) for i in data[args.input_column]
        ]

    llm = LLM(model=args.model_name, max_model_len=4096, tensor_parallel_size=args.tp)
    params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    logger.info(f"Processing {len(lst_user_messages)} instructions")

    inputs = [
        tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
        for c in lst_user_messages
    ]

    outputs = llm.generate(inputs, params)

    results = []
    for i, rsp in enumerate(outputs):
        lst_responses = [item.text for item in rsp.outputs]
        lst_candidates = [
            lst_user_messages[i] + [{"role": "assistant", "content": r}]
            for r in lst_responses
        ]
        results.append(
            {
                "candidates": lst_candidates,
                "model": args.model_name,
                "sampling_params": {
                    "n": args.n,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "top_p": args.top_p,
                },
            }
        )

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()

from vllm import LLM, SamplingParams
from tqdm import tqdm
from datasets import load_from_disk, load_dataset
from utils import input_difficulty_rating, input_quality_rating, input_classification
from transformers import AutoTokenizer
from loguru import logger

import os
import argparse
import json
import pandas as pd


logger.add("unitag.log")


def process_engine_responses(response, mission):
    item = {}
    try:
        tags_json = json.loads(response)
        if mission == "difficulty":
            item["intent"] = tags_json["intent"]
            item["knowledge"] = tags_json["knowledge"]
            item["difficulty"] = tags_json["difficulty"]
            item["difficulty_generator"] = model_name
            
        elif mission == "quality":
            item["input_quality"] = tags_json["input_quality"]
            item["quality_explanation"] = tags_json["explanation"]
            item["quality_generator"] = model_name
            
        elif mission == "classification":
            item["task_category"] = tags_json["primary_tag"]
            item["other_task_category"] = tags_json["other_tags"]
            item["task_category_generator"] = model_name
            
    except Exception as e:
        logger.info(f"[unitag.py] Failed to process item with error: {str(e)}")
        logger.info(f"[unitag.py] Raw response from LLM tagger: {response}")
        if mission == "difficulty":
            item["intent"] = None
            item["knowledge"] = None
            item["difficulty"] = None
            item["difficulty_generator"] = None
        elif mission == "quality":
            item["input_quality"] = None
            item["quality_explanation"] = None
            item["quality_generator"] = None
        elif mission == "classification":
            item["task_category"] = None
            item["other_task_category"] = None
            item["task_category_generator"] = None

    return item


def template_generator(input, mission):
    if mission == "difficulty":
        return input_difficulty_rating(input)
    elif mission == "quality":
        return input_quality_rating(input)
    elif mission == "classification":
        return input_classification(input)
    else:
        raise ValueError(
            "Invalid mission. Available missions: difficulty, quality, classification"
        )


def get_instructions(path, instruction_column, start, offset):
    try:
        data = load_from_disk(path)
    except FileNotFoundError:
        data = load_dataset(path, split="train")

    if start is not None:
        data = data.select(range(start, len(data)))

    if offset is not None:
        data = data.select(range(0, offset))

    return data[instruction_column]


def vllm_batch_generate(llm, lst_inputs, params):
    outputs = llm.generate(lst_inputs, params)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--instruction_column", type=str, default="instruction")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--offset", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="unitag_output")
    parser.add_argument("--mission", type=str, default="all")
    args = parser.parse_args()


    os.makedirs(args.output_dir, exist_ok=True)
    
    # ===============================
    # Parse the mission
    if args.mission == "all":
        missions = ["difficulty", "quality", "classification"]
    else:
        # parse the mission by comma
        missions = [m.strip() for m in args.mission.split(",")]
        assert all(m in {"difficulty", "quality", "classification"} for m in missions)
    logger.info(f"Missions: {missions}")
    # ===============================
    
    
    # ===============================
    # Get the instructions
    lst_instructions = get_instructions(
        args.data_path, args.instruction_column, args.start, args.offset
    )
    logger.info(f"Processing {len(lst_instructions)} instructions")
    # ===============================
    
    
    # ===============================
    # Initialize the model
    global model_name
    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if "Meta-Llama-3.1" in model_name:
        tokenizer.chat_template = "{{- bos_token }}{%- for message in messages %}{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{%- endif %}"

    llm = LLM(model=model_name, max_model_len=4096)
    params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        stop=["}"],
        include_stop_str_in_output=True,
    )
    # ===============================


    for mission in missions:        
        logger.info(f"Start processing mission: {mission}")

        lst_user_messages = [
            [{"role": "user", "content": template_generator(i, mission)}]
            for i in lst_instructions
        ]

        lst_inputs = [
            tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
            + "{"
            for c in lst_user_messages
        ]

        logger.info(f"Generating tags for {len(lst_inputs)} inputs")

        outputs = llm.generate(lst_inputs, params)


        results = []
        for i, _ in tqdm(enumerate(outputs), total=len(outputs)):

            model_response = "{\n" + outputs[i].outputs[0].text.strip()
            # Remove additional information at the end of the response
            model_response = model_response[: model_response.rfind("}") + 1]

            item = process_engine_responses(model_response, mission)

            result = {"instruction": lst_instructions[i], **item}
            results.append(result)
        
        df = pd.DataFrame(results)
        output_path = os.path.join(args.output_dir, f"{mission}_start-{args.start}_offset-{args.offset}.csv")
        df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()
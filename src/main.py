import argparse
import json
import os
import asyncio

from src import utils
from glob import glob
from loguru import logger
from transformers import AutoTokenizer
from typing import List, Dict, Callable
from src.config import load_config
from src.data import get_dataset
from src.llm import AsyncOpenAIBackend
from tqdm import tqdm

logger.add("logs/data_synthesis.log", rotation="10 MB", backtrace=True, diagnose=True, level="ERROR")

async def is_related(current_messages: List[Dict[str, str]], next_message: Dict[str, str], tokenizer: AutoTokenizer, verification_model: AsyncOpenAIBackend) -> bool:
    conversation = "\n".join([f"{m['role']}: {m['content']}" for m in current_messages])
    message = f"{next_message['role']}: {next_message['content']}"

    prompt = f'''Given the following conversation:
### Conversation
```
{conversation}
```

Can we continue the conversation with the following message without breaking the flow of the conversation?
### Message
```
{message}
```

### Output Format
Begin your evaluation by providing a short explanation why the message can or cannot be continued in the conversation. Then, provide the output as either "YES" or "NO".

Please provide your answer in the following format:
{{
    "explanation": "Your explanation here.",
    "output": "YES" or "NO"
}}
'''
    fmt_prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) + "{"
    
    response = await verification_model.generate(fmt_prompt, sampling_params={"temperature": 0.0, "max_tokens": 1024, "stop": ["}"]})
    
    response = "{" + response + "}"
    
    try:
        response = json.loads(response)
        
        if response["output"] == "YES":
            return True
        return False
    
    except json.JSONDecodeError:
        logger.error(f"Failed to decode response: {response}")
        return False
    

async def generate_bot_message(messages: List[Dict[str, str]], synthesis_model: AsyncOpenAIBackend) -> str:
    assert messages[-1]["role"] == "user"
    
    response_content = await synthesis_model.chat_generate(messages)
    
    return {"role": "assistant", "content": response_content}
    

async def generate_user_message(messages: List[Dict[str, str]], synthesis_model: AsyncOpenAIBackend) -> str:
    assert messages[-1]["role"] == "assistant"
    
    new_system_prompt = """You are a curious chatbot designed to ask insightful questions.\
Your task is to ask a new question or provide a new instruction based on the conversation so far.\
You should not provide any feedbacks to the user messages, just ask a new question or provide a new instruction.\
The new question or instruction must be reasonable, coherent, and must be understood and responded by humans."""
    # reverse the role user -> assistant; assistant -> user
    role_reversed_messages = [
        {"role": "system", "content": new_system_prompt}
    ]
    for msg in messages:
        if msg["role"] == "user":
            role_reversed_messages.append({"role": "assistant", "content": msg["content"]})
        elif msg["role"] == "assistant":
            role_reversed_messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "system":
            continue
        else:
            logger.error(f"Invalid role: {msg['role']}")
            raise ValueError(f"Invalid role: {msg['role']}")
    
    assert role_reversed_messages[-1]["role"] == "user"
    
    response_content = await synthesis_model.chat_generate(role_reversed_messages)
    
    return {"role": "user", "content": response_content}
    

async def synthesize_based_on_reference(reference_messages: List[Dict[str,str]], tokenizer: AutoTokenizer, synthesis_model: AsyncOpenAIBackend, verification_model: AsyncOpenAIBackend, dump_to: str = None) -> List[Dict[str, str]]:
    assert reference_messages[0]["role"] in {"system", "user"}
    if reference_messages[0]["role"] == "system":
        assert reference_messages[1]["role"] == "user"
        synthetic_messages = reference_messages[:2].copy()
    else:
        synthetic_messages = reference_messages[:1].copy()
    
    while True:
        try:
            bot_message = await generate_bot_message(synthetic_messages, synthesis_model)
            synthetic_messages.append(bot_message)
            
            if len(synthetic_messages) == len(reference_messages):
                break
            
            ref_user_message = reference_messages[len(synthetic_messages)]
            if await is_related(synthetic_messages, ref_user_message, tokenizer, verification_model):
                synthetic_messages.append(ref_user_message)
            else:
                logger.info(f"Next message is not related to the conversation. Stopping synthesis. Length of synthesized messages: {len(synthetic_messages)} - Length of reference messages: {len(reference_messages)}")
                break
            
        except TimeoutError:
            logger.error("TimeoutError")
            return
    
    if dump_to:
        with open(dump_to, "w") as f:
            json.dump(synthetic_messages, f, indent=4)
    
    return synthetic_messages


async def synthesize_expansion(reference_messages: List[Dict[str,str]], synthesis_model: AsyncOpenAIBackend, n: int = 2, dump_to: str = None) -> List[Dict[str, str]]:
    synthetic_messages = reference_messages.copy()
    
    for _ in range(n):
        try:
            if synthetic_messages[-1]["role"] == "user":
                bot_message = await generate_bot_message(synthetic_messages, synthesis_model)
                synthetic_messages.append(bot_message)
            elif synthetic_messages[-1]["role"] == "assistant":
                user_message = await generate_user_message(synthetic_messages, synthesis_model)
                synthetic_messages.append(user_message)
            else:
                logger.error(f"Invalid role: {synthetic_messages[-1]['role']}")
                return
            
        except TimeoutError:
            logger.error("TimeoutError")
            return

    if dump_to:
        with open(dump_to, "w") as f:
            json.dump(synthetic_messages, f, indent=4)
            
    return synthetic_messages

async def main():
    config = load_config(file_path="config.yaml")
    logger.info(f"**CONFIG**: {config}")
    
    data = get_dataset(config.dataset)
    tokenizer = AutoTokenizer.from_pretrained(config.verification_model.model)
    
    save_dir = config.save_dir
    os.makedirs(save_dir, exist_ok=True)
    cache_dir = os.path.join(save_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    synthesis_model = AsyncOpenAIBackend(model_config=config.synthesis_model)
    verification_model = AsyncOpenAIBackend(model_config=config.verification_model)
    
    # lst_instructions = data["instruction"]
    lst_reference_messages = data["messages"]
    
    
    tasks = set()
    num_concurrents = config.batch_size
    
    for i in range(0, len(lst_reference_messages)):
        if os.path.exists(os.path.join(cache_dir, f"{i}.json")):
            continue
        
        brefeference_messages = lst_reference_messages[i]
        if len(tasks) >= num_concurrents:
            _done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        
        print(f"starting task {i}")
        task = asyncio.create_task(synthesize_based_on_reference(brefeference_messages, tokenizer, synthesis_model, verification_model, dump_to=os.path.join(cache_dir, f"{i}.json")))
        # task = asyncio.create_task(synthesize_expansion(brefeference_messages, synthesis_model, n=2, dump_to=os.path.join(cache_dir, f"{i}.json")))
        tasks.add(task)
        
    await asyncio.wait(tasks)
        
    
    
    lst_cache_files = glob(f"{cache_dir}/*.json")
    lst_cache_files = sorted(lst_cache_files, key=lambda x: int(x.split("/")[-1].split(".")[0]))
    results = []
    for cache_file in tqdm(lst_cache_files):
        with open(cache_file, "r") as f:
            results.append(json.load(f))
            
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
        
if __name__ == "__main__":
    import time
    
    start = time.time()
    asyncio.run(main())
    
    print(f"Time taken: {time.time() - start} seconds")
import argparse
import json
import os


from loguru import logger
from transformers import AutoTokenizer
from typing import List, Dict, Callable
from src.config import load_config
from src.data import get_dataset
from src.llm import OpenAIBackend
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

logger.add("logs/main.log", rotation="10 MB", backtrace=True, diagnose=True)

def is_related(current_messages: List[Dict[str, str]], next_message: Dict[str, str], tokenizer: AutoTokenizer, verification_model: OpenAIBackend) -> bool:
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
```
{{
    "explanation": "Your explanation here.",
    "output": "YES" or "NO"
}}
```
'''
    fmt_prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) + "{"
    
    response = verification_model.generate(fmt_prompt, sampling_params={"temperature": 0.0, "max_tokens": 1024, "stop": ["}"]})
    
    response = "{" + response + "}"
    
    try:
        response = json.loads(response)
        
        if response["output"] == "YES":
            return True
        return False
    
    except json.JSONDecodeError:
        logger.error(f"Failed to decode response: {response}")
        return False
    

def generate_bot_message(messages: List[Dict[str, str]], synthesis_model: OpenAIBackend) -> str:
    assert messages[-1]["role"] == "user"
    
    return {"role": "assistant", "content": synthesis_model.chat_generate(messages)}
    

def generate_user_message(messages: List[Dict[str, str]], synthesis_model: OpenAIBackend) -> str:
    assert messages[-1]["role"] == "assistant"
    
    new_system_prompt = """You are a curious chatbot designed to ask insightful questions or provide high-quality instructions based on the user's message. Ensure that the conversation with the user flows naturally and smoothly."""
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
    
    return {"role": "user", "content": synthesis_model.chat_generate(role_reversed_messages)}


def generate_user_message_until_pass(messages: List[Dict[str, str]], synthesis_model: OpenAIBackend, tokenizer: AutoTokenizer, verification_model: OpenAIBackend, retries=5) -> str:
    assert messages[-1]["role"] == "assistant"
    
    for _ in range(retries):
        user_message = generate_user_message(messages, synthesis_model)
        
        if is_related(messages, user_message, tokenizer, verification_model):
            return user_message
        
    logger.info(f"Failed to generate a related user message after {retries} retries.")
    return None
    

def synthesize_from_reference(reference_messages: List[Dict[str,str]],tokenizer: AutoTokenizer, synthesis_model: OpenAIBackend, verification_model: OpenAIBackend) -> List[Dict[str, str]]:
    assert reference_messages[0]["role"] in {"system", "user"}
    if reference_messages[0]["role"] == "system":
        assert reference_messages[1]["role"] == "user"
        synthesized_messages = reference_messages[:2].copy()
    else:
        synthesized_messages = reference_messages[:1].copy()
    
    while True:
        bot_message = generate_bot_message(synthesized_messages, synthesis_model)
        synthesized_messages.append(bot_message)
        
        if len(synthesized_messages) == len(reference_messages):
            break
        
        ref_user_message = reference_messages[len(synthesized_messages)]
        if is_related(synthesized_messages, ref_user_message, tokenizer, verification_model):
            synthesized_messages.append(ref_user_message)
        else:
            logger.info(f"Next message is not related to the conversation. Stopping synthesis. Length of synthesized messages: {len(synthesized_messages)} - Length of reference messages: {len(reference_messages)}")
        
    return synthesized_messages


def synthesize_continuous(reference_messages: List[Dict[str,str]],tokenizer: AutoTokenizer, synthesis_model: OpenAIBackend, verification_model: OpenAIBackend, n=2) -> List[Dict[str, str]]:
    assert reference_messages[0]["role"] in {"system", "user"}
    
    synthesized_messages = reference_messages.copy()
    
    for _ in range(n):
        if synthesized_messages[-1]["role"] == "user":
            bot_message = generate_bot_message(synthesized_messages, synthesis_model)
            synthesized_messages.append(bot_message)
            
        elif synthesized_messages[-1]["role"] == "assistant":
            user_message = generate_user_message_until_pass(synthesized_messages, synthesis_model, tokenizer, verification_model)
            if not user_message:
                break
            synthesized_messages.append(user_message)
            
        else:
            raise ValueError(f"Invalid role: {synthesized_messages[-1]['role']}")

    return synthesized_messages

def main():
    config = load_config(file_path="config.yaml")
    
    data = get_dataset(config.dataset)
    tokenizer = AutoTokenizer.from_pretrained(config.verification_model.model)
    
    save_dir = config.save_dir
    os.makedirs(save_dir, exist_ok=True)
    cache_dir = os.path.join(save_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    synthesis_model = OpenAIBackend(model_config=config.synthesis_model)
    verification_model = OpenAIBackend(model_config=config.verification_model)
    
    lst_instructions = data["instruction"]
    lst_reference_messages = data["messages"]
    
    batch_size = 100
    
    for i in tqdm(range(0, len(lst_instructions), batch_size)):
        if os.path.exists(os.path.join(cache_dir, f"{i}.json")):
            continue
        
        batch_reference_messages = lst_reference_messages[i:i+batch_size]
        
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [
                executor.submit(synthesize_from_reference, reference_messages, tokenizer, synthesis_model, verification_model) for reference_messages in batch_reference_messages
            ]
            
            batch_synthesized_messages = [future.result() for future in futures]
    
            # save to cache
            with open(os.path.join(cache_dir, f"{i}.json"), "w") as file:
                json.dump(batch_synthesized_messages, file, indent=4)
                
        break
        
if __name__ == "__main__":
    main()
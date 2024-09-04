from openai import OpenAI, AsyncOpenAI
from src.config import ModelConfig
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

def is_valid_messages(messages: List[Dict[str, str]]) -> bool:
    if len(messages) == 0:
        return False
    
    for i in range(1, len(messages)):
        if messages[i]["content"].strip() == "":
            return False
        
        if messages[i]["role"] == messages[i - 1]["role"]:
            return False
        
        if messages[i]["role"] not in {"user", "assistant"}:
            return False
        
    return True


class OpenAIBackend:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.client = OpenAI(api_key=model_config.api_key, base_url=model_config.base_url)
        self.sampling_params = {
            "temperature": self.model_config.temperature,
            "top_p": self.model_config.top_p,
            "max_tokens": self.model_config.max_tokens,
            "stop": self.model_config.stop,
        }

    def generate(self, text: str, sampling_params: Dict = None) -> str:
        if not sampling_params:
            sampling_params = self.sampling_params
            
        response = self.client.completions.create(
            model=self.model_config.served_model_name,
            prompt=text,
            **sampling_params
        )
        
        return response.choices[0].text
    
    def chat_generate(self, messages: List[Dict[str, str]], sampling_params: Dict = None) -> str:
        if not sampling_params:
            sampling_params = self.sampling_params
            
        response = self.client.chat.completions.create(
            model=self.model_config.served_model_name,
            messages=messages,
            **sampling_params
        )
        
        return response.choices[0].message.content
        
    def generate_batch(self, lst_texts: List[str], sampling_params: Dict = None) -> List[str]:
        if not sampling_params:
            sampling_params = self.sampling_params
            
        with ThreadPoolExecutor(max_workers=len(lst_texts)) as executor:
            futures =[
                executor.submit(self.generate, text, sampling_params) for text in lst_texts
            ]
            
            results = [future.result() for future in futures]
            
        return results
    
    
class AsyncOpenAIBackend:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.client = AsyncOpenAI(api_key=model_config.api_key, base_url=model_config.base_url)
        self.sampling_params = {
            "temperature": self.model_config.temperature,
            "top_p": self.model_config.top_p,
            "max_tokens": self.model_config.max_tokens,
            "stop": self.model_config.stop,
        }

    async def generate(self, text: str, sampling_params: Dict = None) -> str:
        if not sampling_params:
            sampling_params = self.sampling_params
            
        response = await self.client.completions.create(
            model=self.model_config.served_model_name,
            prompt=text,
            timeout=30000,
            **sampling_params
        )
        
        return response.choices[0].text
    
    async def chat_generate(self, messages: List[Dict[str, str]], sampling_params: Dict = None) -> str:
        if not sampling_params:
            sampling_params = self.sampling_params
            
        response = await self.client.chat.completions.create(
            model=self.model_config.served_model_name,
            messages=messages,
            timeout=30000,
            **sampling_params
        )
        
        return response.choices[0].message.content
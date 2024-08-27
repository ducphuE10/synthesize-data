from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import yaml


class ModelConfig(BaseModel):
    backend: str
    model: str
    served_model_name: str
    chat_template: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    
    temperature: float
    max_tokens: int
    top_p: float
    stop: List[str]

    @field_validator('backend')
    def validate_backend(cls, v):
        if v not in {'vllm', 'openai'}:
            raise ValueError('backend must be either "vllm" or "openai"')
        return v


class DatasetConfig(BaseModel):
    path: str
    instruction_column: str
    messages_column: str


class Config(BaseModel):
    synthesis_model: ModelConfig
    verification_model: ModelConfig | None = None
    dataset: DatasetConfig
    save_dir: str


def load_config(file_path: str) -> Config:
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)        
    config = Config(**config_dict)

    if not config.verification_model:
        config.verification_model = config.synthesis_model.model_copy()

    return config
from datasets import load_dataset, load_from_disk, DatasetDict, concatenate_datasets
from src.config import DatasetConfig

def get_dataset(dataset_config: DatasetConfig):
    try:
        dataset = load_from_disk(dataset_config.path)
    except FileNotFoundError:
        dataset = load_dataset(dataset_config.path, split="train")
        
    return dataset
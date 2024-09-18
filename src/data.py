from datasets import load_dataset, load_from_disk, DatasetDict, concatenate_datasets
from src.config import DatasetConfig
from src import utils

def get_dataset(dataset_config: DatasetConfig):
    try:
        dataset = load_from_disk(dataset_config.path)
    except FileNotFoundError:
        dataset = load_dataset(dataset_config.path, split="train")
        
    if dataset_config.messages_type == "openAI":
        for example in dataset:
            utils.validate_openAI(example["messages"])
        
    elif dataset_config.messages_type == "shareGPT":
        for example in dataset:
            utils.validate_shareGPT(example["conversations"])
        
        def prepare_fn(example):
            example["messages"] = utils.fmt_shareGPT_to_openAI(example["conversations"])
            return example
        
        dataset = dataset.map(prepare_fn)
    
    if dataset_config.num_test:
        dataset = dataset.select(range(dataset_config.num_test))
    
    print("Example:")
    print("----------------------------------------")
    print(dataset[0])
    print("----------------------------------------")
    print(dataset[-1])
    print("----------------------------------------")
    
    return dataset
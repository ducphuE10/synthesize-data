import pandas as pd
import random
import torch
import os
import argparse
import json
import pandas as pd

from loguru import logger
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset, load_from_disk, DatasetDict
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import PyTorchModelHubMixin


class QualityModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super(QualityModel, self).__init__()
        self.model = AutoModel.from_pretrained(config["base_model"])
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))

    def forward(self, input_ids, attention_mask):
        features = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        dropped = self.dropout(features)
        outputs = self.fc(dropped)
        return torch.softmax(outputs[:, 0, :], dim=1)


device = "cuda"
# device = "cuda" if torch.cuda.is_available() else "cpu"

# Setup configuration and model
config = AutoConfig.from_pretrained("nvidia/quality-classifier-deberta")
tokenizer = AutoTokenizer.from_pretrained("nvidia/quality-classifier-deberta")
model = QualityModel.from_pretrained("nvidia/quality-classifier-deberta").to(device)
model.eval()
print("Model loaded")


def predict(text):
    inputs = tokenizer(
        text, return_tensors="pt", padding="longest", truncation=True
    ).to(device)
    outputs = model(inputs["input_ids"], inputs["attention_mask"])
    predicted_classes = torch.argmax(outputs, dim=1)
    predicted_domains = [
        config.id2label[class_idx.item()]
        for class_idx in predicted_classes.cpu().numpy()
    ]
    return predicted_domains[0].lower()

def batch_predict(lst_texts):
    inputs = tokenizer(
        lst_texts, return_tensors="pt", padding="longest", truncation=True
    ).to(device)
    
    outputs = model(inputs["input_ids"], inputs["attention_mask"])
    predicted_classes = torch.argmax(outputs, dim=1)
    predicted_domains = [
        config.id2label[class_idx.item()]
        for class_idx in predicted_classes.cpu().numpy()
    ]
    return [domain.lower() for domain in predicted_domains]

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--instruction_column", type=str, default="instruction")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--offset", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="unitag_output")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()


    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get the instructions
    lst_instructions = get_instructions(
        args.data_path, args.instruction_column, args.start, args.offset
    )
    logger.info(f"Processing {len(lst_instructions)} instructions")
    
    output_qualities = []
    for i in tqdm(range(0, len(lst_instructions), args.batch_size)):
        batch = lst_instructions[i:i + args.batch_size]
        output_qualities.extend(batch_predict(batch))
        
    
    # Save the output
    df = pd.DataFrame({"instruction": lst_instructions, "quality_NeMo": output_qualities})
    df.to_parquet(os.path.join(args.output_dir, f'quality-NeMo_start-{args.start}_offset-{args.offset}.parquet'), index=False)    
    
if __name__ == "__main__":
    main()
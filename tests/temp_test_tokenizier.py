from checker.utils.transformer_tokenizer import tokenizer_base
import os
import json
from datasets import load_dataset


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(base_dir, "checker/config/config.json")) as f:
    config = json.load(f)

RAW_PATH = os.path.join(base_dir, config["raw_data_path"])
TRAIN_PATH = os.path.join(base_dir, config["train_data_path"])
VAL_PATH = os.path.join(base_dir, config["val_data_path"])
TEST_PATH = os.path.join(base_dir, config["test_data_path"])


data = {"train": TRAIN_PATH, "val": VAL_PATH, "test": TEST_PATH}
# Read data
data_raw = load_dataset("csv", data_files=data)

dataset_raw = data_raw.map(tokenizer_base, batched=True)

dataset_cleaned = dataset_raw.remove_columns(["title","url", "person", 'statementdate', 'source', 'factcheckdate', 'factchecker', 'sources', 'long_text', 'short_text', 'text','Unnamed: 0','_id'])
tokenized_datasets = dataset_raw.rename_column("label", "labels")

print("Finished")
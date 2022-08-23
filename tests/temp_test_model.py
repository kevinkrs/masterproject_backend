import json
import logging
import os
import mlflow
from datasets import load_dataset

from datetime import datetime
from checker.model.transformer import TransformerModel
from checker.utils.transformer_tokenizer import tokenizer_base

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s",
    level=logging.DEBUG,
)
LOGGER = logging.getLogger(__name__)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZER_PARALLELISM"] = "true"


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(base_dir, "checker/config/config.json")) as f:
    config = json.load(f)

# set_random_seed(42)
mlflow.set_experiment(config["model"])

model_output_path = os.path.join(base_dir, config["model_output_path"])
# Update full model output path
config["model_output_path"] = model_output_path
os.makedirs(model_output_path, exist_ok=True)

TRAIN_PATH = os.path.join(base_dir, config["train_data_path"])
VAL_PATH = os.path.join(base_dir, config["val_data_path"])
TEST_PATH = os.path.join(base_dir, config["test_data_path"])

data = {"train": TRAIN_PATH, "val": VAL_PATH, "test": TEST_PATH}
# Read data
data_raw = load_dataset("csv", data_files=data)


dataset_raw = data_raw.map(tokenizer_base, batched=True)

dataset_cleaned = dataset_raw.remove_columns(["title","url", "person", 'statementdate', 'source', 'factcheckdate', 'factchecker', 'sources', 'long_text', 'short_text', 'text','Unnamed: 0','_id'])
tokenized_datasets = dataset_cleaned.rename_column("label", "labels")

if config["from_ckp"]:
    model = TransformerModel(config, load_from_ckpt=True)
else:
    model = TransformerModel(config)


model.train(tokenized_datasets)
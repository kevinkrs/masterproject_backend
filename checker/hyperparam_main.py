import os
import json

from model.transformer import tune_bert
from utils.dataset_module import TransformerDataModule

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(base_dir, "config/config.json")) as f:
    config = json.load(f)

data_dir = os.path.join(base_dir, config["full_data_path"])

datamodule = TransformerDataModule(config)
analysis = tune_bert(data_dir, datamodule)

print(analysis.best_config)

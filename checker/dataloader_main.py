import json
import os
import logging

from checker.utils.dataloader import Dataloader
from config import config_secrets


logger = logging.getLogger("Test")
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(base_dir, "config/config.json")) as f:
    config = json.load(f)

RAW_PATH = os.path.join(base_dir, config["raw_data_path"])
FULL_PATH = os.path.join(base_dir, config["full_data_path"])
TRAIN_PATH = os.path.join(base_dir, config["train_data_path"])
VAL_PATH = os.path.join(base_dir, config["val_data_path"])
TEST_PATH = os.path.join(base_dir, config["test_data_path"])

# Only if required
loader = Dataloader(config, config_secrets)
res = loader.load_data_from_db(RAW_PATH)
loader.create_model_data(RAW_PATH, FULL_PATH, TRAIN_PATH, VAL_PATH, TEST_PATH)

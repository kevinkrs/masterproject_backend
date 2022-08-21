from checker.utils.dataloader import load_data_from_db, create_model_data
import os
import json


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(base_dir, "checker/config/roberta_v1.json")) as f:
    config = json.load(f)


RAW_PATH = os.path.join(base_dir, config["raw_data_path"])
TRAIN_PATH = os.path.join(base_dir, config["train_data_path"])
VAL_PATH = os.path.join(base_dir, config["val_data_path"])
TEST_PATH = os.path.join(base_dir, config["test_data_path"])


load_data_from_db(RAW_PATH)
create_model_data(RAW_PATH, TRAIN_PATH, VAL_PATH, TEST_PATH, config)
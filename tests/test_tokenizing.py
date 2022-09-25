from checker.model.utils.transformer_tokenizer import tokenizer_base
import os
import pandas as pd
import json


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_tokenizer():
    with open(os.path.join(base_dir, "config/config.json")) as f:
        config = json.load(f)

    TRAIN_PATH = os.path.join(base_dir, config["train_data_path"])
    VAL_PATH = os.path.join(base_dir, config["val_data_path"])
    TEST_PATH = os.path.join(base_dir, config["test_data_path"])

    # Read data
    train_raw = pd.read_csv(TRAIN_PATH)
    val_raw = pd.read_csv(VAL_PATH)
    test_raw = pd.read_csv(TEST_PATH)

    # Tokenize data
    train_datapoints = tokenizer_base(train_raw)
    val_datapoints = tokenizer_base(val_raw)
    test_datapoints = tokenizer_base(test_raw)

    return (train_datapoints, val_datapoints, test_datapoints)

import argparse
import json
import logging
import os
import random
from shutil import copy
from datetime import datetime

import mlflow
import numpy as np
import torch

from utils.dataloader import load_data_from_db, create_model_data
from checker.model.roberta_based import RobertaModel
from checker.utils.reader import read_csv_data

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s",
    level=logging.DEBUG,
)
LOGGER = logging.getLogger(__name__)


# def read_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser()
#     # Defining command line input "--config-file"
#     parser.add_argument("--config-file", type=str)
#     return parser.parse_args()


def set_random_seed(val: int = 1) -> None:
    random.seed(val)
    np.random.seed(val)
    # Torch-specific random-seeds
    torch.manual_seed(val)
    torch.cuda.manual_seed_all(val)


if __name__ == "__main__":
    # Triggered via "python checker/train.py --config-file config/distilbert.json -> Config is read and ran by the code"
    # args = read_args()
    # with open(args.config_file) as f:
    #     config = json.load(f)

    with open("config/roberta_config_v1.json") as f:
        config = json.load(f)

    set_random_seed(42)
    mlflow.set_experiment(config["model"])

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_output_path = os.path.join(base_dir, config["model_output_path"])
    # Update full model output path
    config["model_output_path"] = model_output_path
    os.makedirs(model_output_path, exist_ok=True)

    # Copy config to model directory
    # copy(args.config_file, model_output_path)
    with mlflow.start_run() as run:
        with open(os.path.join(model_output_path, "meta.json"), "w") as f:
            json.dump({"mlflow_run_id": run.info.run_id}, f)
        mlflow.set_tags({"evaluate": config["evaluate"]})

        RAW_PATH = os.path.join(base_dir, config["raw_data_path"])
        TRAIN_PATH = os.path.join(base_dir, config["train_data_path"])
        VAL_PATH = os.path.join(base_dir, config["val_data_path"])
        TEST_PATH = os.path.join(base_dir, config["test_data_path"])

        if config["update_data"]:
            ## Main
            load_data_from_db(RAW_PATH)
            create_model_data(RAW_PATH, TRAIN_PATH, VAL_PATH, TEST_PATH, config)

        # Read data
        train_datapoints = read_csv_data(TRAIN_PATH)
        val_datapoints = read_csv_data(VAL_PATH)
        test_datapoints = read_csv_data(TEST_PATH)

        if config["model"] == "roberta":
            model = RobertaModel(config)
        else:
            raise ValueError(f"Invalid model type {config['model']} provided")

        if not config["evaluate"]:
            LOGGER.info("Training model...")
            model.train(train_datapoints, val_datapoints, cache_featurizer=True)

        mlflow.log_params(model.get_params())
        LOGGER.info("Evaluating model...")
        val_metrics = model.compute_metrics(val_datapoints, split="val")
        LOGGER.info(f"Val metrics: {val_metrics}")
        test_metrics = model.compute_metrics(test_datapoints, split="test")
        LOGGER.info(f"Test metrics: {test_metrics}")
        mlflow.log_metrics(val_metrics)
        mlflow.log_metrics(test_metrics)

        metrics = ["val_metrics", "test_metrics"]
        metric_objects = [val_metrics, test_metrics]

        for metric in metrics:
            with open(
                f"logs/{metric}_{config['model']}_{datetime.now()}.json", "w"
            ) as i:
                json.dump([str(x) for x in metric_objects], i)

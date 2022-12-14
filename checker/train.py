import json
import logging
import os
import mlflow
import torch

from model.transformer import TransformerModel
from utils.dataloader import Dataloader
from utils.dataset_module import TransformerDataModule
from config import config_secrets

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s",
    level=logging.DEBUG,
)
LOGGER = logging.getLogger(__name__)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZER_PARALLELISM"] = "true"

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":

    with open(os.path.join(base_dir, "config/config.json")) as f:
        config = json.load(f)

    mlflow.set_experiment(config["model"])

    model_output_path = os.path.join(base_dir, config["model_output_path"])

    os.makedirs(os.path.join(base_dir, config["model_output_path"]), exist_ok=True)

    with mlflow.start_run() as run:
        with open(os.path.join(model_output_path, "meta.json"), "w") as f:
            json.dump({"mlflow_run_id": run.info.run_id}, f)
        mlflow.set_tags({"evaluate": config["evaluate"]})

        RAW_PATH = os.path.join(base_dir, config["raw_data_path"])
        FULL_PATH = os.path.join(base_dir, config["full_data_path"])
        TRAIN_PATH = os.path.join(base_dir, config["train_data_path"])
        VAL_PATH = os.path.join(base_dir, config["val_data_path"])
        TEST_PATH = os.path.join(base_dir, config["test_data_path"])

        if config["update_data"]:
            LOGGER.info("Data preprocessing started")
            loader = Dataloader(config, secrets=config_secrets)
            loader.load_data_from_db(RAW_PATH)
            loader.create_model_data(
                RAW_PATH, FULL_PATH, TRAIN_PATH, VAL_PATH, TEST_PATH
            )

        # Datamodule handles loading the data and tokenizing it
        datamodule = TransformerDataModule(config)
        datamodule.setup("fit")

        if config["from_ckp"]:
            model = TransformerModel(config, load_from_ckpt=True)
        else:
            model = TransformerModel(config)

        if config["train"]:
            LOGGER.info("Training model...")
            model.train(datamodule)

        validation = model.validate_model(datamodule)
        dataloader_val = datamodule.val_dataloader()
        dataloader_test = datamodule.test_dataloader()
        LOGGER.info("Evaluating model...")
        val_metrics = model.compute_metrics(dataloader_val, split="val")
        LOGGER.info(f"Val metrics: {val_metrics}")
        test_metrics = model.compute_metrics(dataloader_test, split="test")
        LOGGER.info(f"Test metrics: {test_metrics}")
        mlflow.log_metrics(val_metrics)
        mlflow.log_metrics(test_metrics)

        metrics = ["val_metrics", "test_metrics"]
        metric_objects = [val_metrics, test_metrics]

        # Additional Note: Logs caused git pull to fail on windows machine
        # for metric in metrics:
        #     with open(
        #             f"{base_dir}/logs/{metric}_{config['model']}_{datetime.now()}.json", "w"
        #     ) as i:
        #         json.dump([str(x) for x in metric_objects], i)

        mlflow.end_run()

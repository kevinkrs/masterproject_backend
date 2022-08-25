import json
import logging
import os
import mlflow
from datasets import load_dataset

from datetime import datetime
from model.transformer import TransformerModel
from utils.dataloader import Dataloader
from modules.dataset_module import TransformerDataModule
from utils.transformer_tokenizer import tokenizer_base

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s",
    level=logging.DEBUG,
)
LOGGER = logging.getLogger(__name__)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZER_PARALLELISM"] = "true"

# def read_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser()
#     # Defining command line input "--config-file"
#     parser.add_argument("--config-file", type=str)
#     return parser.parse_args()


# def set_random_seed(val: int = 1) -> None:
#     random.seed(val)
#     np.random.seed(val)
#     # Torch-specific random-seeds
#     torch.manual_seed(val)
#     torch.cuda.manual_seed_all(val)


if __name__ == "__main__":
    # Triggered via "python checker/train.py --config-file config/distilbert.json -> Config is read and ran by the code"
    # args = read_args()
    # with open(args.config_file) as f:
    #     config = json.load(f)
    # set some environment variables
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with open(os.path.join(base_dir, "checker/config/config.json")) as f:
        config = json.load(f)

    # set_random_seed(42)
    mlflow.set_experiment(config["model"])

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
            loader = Dataloader()
            loader.load_data_from_db(RAW_PATH)
            loader.create_model_data(RAW_PATH, TRAIN_PATH, VAL_PATH, TEST_PATH)

        # Datamodule handles loading the data and tokenizing it
        datamodule = TransformerDataModule(config["type"])
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
        dataloader_train = datamodule.test_dataloader()
        LOGGER.info("Evaluating model...")
        val_metrics = model.compute_metrics(dataloader_val, split="val")
        LOGGER.info(f"Val metrics: {val_metrics}")
        test_metrics = model.compute_metrics(dataloader_train, split="test")
        LOGGER.info(f"Test metrics: {test_metrics}")
        mlflow.log_metrics(val_metrics)
        mlflow.log_metrics(test_metrics)

        metrics = ["val_metrics", "test_metrics"]
        metric_objects = [val_metrics, test_metrics]

        for metric in metrics:
            with open(
                    f"{base_dir}/logs/{metric}_{config['model']}_{datetime.now()}.json", "w"
            ) as i:
                json.dump([str(x) for x in metric_objects], i)

        mlflow.end_run()
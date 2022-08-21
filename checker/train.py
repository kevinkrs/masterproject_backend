import json
import logging
import os
import mlflow
import pandas as pd


from datetime import datetime
from model.roberta_based import RobertaModel
# from utils.dataloader import load_data_from_db, create_model_data TODO: Can't fix module not found error...
from utils.roberta_tokenizer import tokenizer_base

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

    with open(os.path.join(base_dir, "checker/config/roberta_v1.json")) as f:
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
            load_data_from_db(RAW_PATH)
            create_model_data(RAW_PATH, TRAIN_PATH, VAL_PATH, TEST_PATH, config)

        # Read data
        train_raw = pd.read_csv(TRAIN_PATH)
        val_raw = pd.read_csv(VAL_PATH)
        test_raw = pd.read_csv(TEST_PATH)

        # Tokenize data
        train_datapoints = tokenizer_base(train_raw)
        val_datapoints = tokenizer_base(val_raw)
        test_datapoints = tokenizer_base(test_raw)

        if config["model"] == "roberta":
            if config["from_ckp"]:
                model = RobertaModel(config, load_from_ckpt=True)
            else:
                model = RobertaModel(config)
        else:
            raise ValueError(f"Invalid model type {config['model']} provided")

        if config["train"]:
            LOGGER.info("Training model...")
            model.train(train_datapoints, val_datapoints)


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
                    f"{base_dir}/logs/{metric}_{config['model']}_{datetime.now()}.json", "w"
            ) as i:
                json.dump([str(x) for x in metric_objects], i)

        mlflow.end_run()
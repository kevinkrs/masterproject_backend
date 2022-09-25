import json
import logging
import os
from datasets import load_dataset
from checker.model.utils.dataset_module import TransformerDataModule
from datetime import datetime
from checker.model.transformer import TransformerModel


def test_model():
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

    RAW_PATH = os.path.join(base_dir, config["raw_data_path"])
    TRAIN_PATH = os.path.join(base_dir, config["train_data_path"])
    VAL_PATH = os.path.join(base_dir, config["val_data_path"])
    TEST_PATH = os.path.join(base_dir, config["test_data_path"])

    datamodule = TransformerDataModule(config["type"])
    datamodule.setup("fit")
    # next(iter(dataloader.train_dataloader()))

    if config["from_ckp"]:
        model = TransformerModel(config, load_from_ckpt=True)
    else:
        model = TransformerModel(config)

    model.train(datamodule)

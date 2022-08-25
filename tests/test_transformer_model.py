import json
import logging
import os
from datasets import load_dataset
from checker.modules.dataset_module import TransformerDataModule
from datetime import datetime
from checker.model.transformer import TransformerModel
from checker.utils.transformer_tokenizer import tokenizer_base
from checker.utils.dataloader import Dataloader


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

    # Only if required
    # loader = Dataloader()
    # loader.load_data_from_db(RAW_PATH)
    # loader.create_model_data(RAW_PATH, TRAIN_PATH, VAL_PATH, TEST_PATH)

    # set_random_seed(42)

    model_output_path = os.path.join(base_dir, config["model_output_path"])
    # Update full model output path

    datamodule = TransformerDataModule(config["type"])
    datamodule.setup("fit")
    #next(iter(dataloader.train_dataloader()))

    if config["from_ckp"]:
        model = TransformerModel(config, load_from_ckpt=True)
    else:
        model = TransformerModel(config)


    model.train(datamodule)
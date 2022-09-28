import os
import json
import logging

from checker.config import config_secrets

from checker.model.transformer import TransformerModel
from checker.utils.dataloader import Dataloader
from checker.utils.dataset_module import TransformerDataModule
from checker.utils.hyperparam_tuning import HyperParamTuning

LOGGER = logging.getLogger("Module Info")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(base_dir, "config/config.json")) as f:
    config = json.load(f)


def test_all_modules():
    model = TransformerModel(config)
    LOGGER.info("TransformerModel initialized successfully")
    dataloader = Dataloader(config, config_secrets)
    LOGGER.info("Dataloader initialized successfully")
    dm = TransformerDataModule(config)
    LOGGER.info("TransformerDataModule initialized successfully")
    tuner = HyperParamTuning(config)
    LOGGER.info("HyperParamTuning initialized successfully")

    print("finish")

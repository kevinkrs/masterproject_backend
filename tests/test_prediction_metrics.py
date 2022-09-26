import json
import os
import logging

from checker.model.transformer import TransformerModel
from checker.utils.dataset_module import TransformerDataModule


def test_run_metric_calculation():
    logging.basicConfig(
        format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s",
        level=logging.DEBUG,
    )
    LOGGER = logging.getLogger(__name__)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with open(os.path.join(base_dir, "checker/config/config.json")) as f:
        config = json.load(f)
    datamodule = TransformerDataModule()
    datamodule.setup("fit")
    # next(iter(dataloader.train_dataloader()))

    model = TransformerModel(config, load_from_ckpt=True)

    dataloader_val = datamodule.val_dataloader()
    dataloader_test = datamodule.test_dataloader()
    LOGGER.info("Evaluating model...")
    val_metrics = model.compute_metrics(dataloader_val, split="val")
    LOGGER.info(f"Val metrics: {val_metrics}")
    test_metrics = model.compute_metrics(dataloader_test, split="test")
    LOGGER.info(f"Test metrics: {test_metrics}")

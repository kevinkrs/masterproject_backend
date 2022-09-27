import os
import json
from model.transformer import TransformerModel
from utils.dataset_module import TransformerDataModule
from ray import air, tune
from ray.tune import CLIReporter
from ray_lightning.tune import TuneReportCallback, get_tune_resources
import pytorch_lightning as pl
from ray_lightning import RayStrategy


class HyperParamTuning:
    def __init__(self, config):
        self.config = config
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # self.dm = datamodule

    def run(self):
        def train_ray(data_dir=None, num_epochs=10, num_gpus=1):
            metrics = {"loss": "avg_val_loss"}

            callbacks = [TuneReportCallback(metrics, on="validation_end")]
            trainer = pl.Trainer(
                max_epochs=num_epochs,
                callbacks=callbacks,
                # devices=num_gpus, accelerator="auto",
                strategy=RayStrategy(num_workers=1, use_gpu=True),
            )
            model = TransformerModel(self.config).model
            dm = TransformerDataMdoule(self.config)

            trainer.fit(model, dm)

        num_samples = 10

        data = os.path.join(self.base_dir, self.config["full_data_path"])

        config = {
            "layer_1": tune.choice([32, 64, 128]),
            "layer_2": tune.choice([64, 128, 256]),
            "lr": tune.loguniform(1e-5, 1e-1),
            "batch_size": tune.choice([32, 64, 128]),
        }

        analysis = tune.run(
            train_ray,
            metric="loss",
            mode="min",
            config=config,
            num_samples=num_samples,
            resources_per_trial=get_tune_resources(
                num_workers=1, num_cpus_per_worker=3, use_gpu=True
            ),
            name="tune_bert",
        )

        print("Best hyperparameters found were: ", analysis.best_config)
        return analysis

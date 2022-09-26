import os
import json

from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from pytorch_lightning import Trainer


class HyperParamTuning:
    def __init__(self, config, model, datamodule):
        self.config = config
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model = model
        self.dm = datamodule

    def train_ray(self, data_dir=None, num_epochs=10, num_gpus=1):
        metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}
        trainer = Trainer(
            max_epochs=num_epochs,
            gpus=num_gpus,
            progress_bar_refresh_rate=0,
            callbacks=[TuneReportCallback(metrics, on="validation_end")],
        )
        trainer.fit(self.model, self.dm)

    def run(self):
        num_samples = 10
        num_epochs = 10
        gpus_per_trial = 1

        data = os.path.join(self.base_dir, self.config["full_data_path"])
        # Download data

        config = {
            "layer_1": tune.choice([32, 64, 128]),
            "layer_2": tune.choice([64, 128, 256]),
            "lr": tune.loguniform(1e-5, 1e-5),
            "batch_size": tune.choice([32, 64, 128]),
        }

        trainable = tune.with_parameters(
            self.train_ray,
            data_dir=data,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial,
        )

        analysis = tune.run(
            trainable,
            resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
            metric="loss",
            mode="min",
            config=config,
            num_samples=num_samples,
            name="tune_mnist",
        )

        return analysis

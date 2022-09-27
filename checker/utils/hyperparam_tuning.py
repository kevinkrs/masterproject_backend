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

    def run(self):
        def train_ray(
            model, dm, data_dir=None, num_epochs=10, num_gpus=0, checkpoint_dir=None
        ):
            metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}
            callbacks = [TuneReportCallback(metrics, on="validation_end")]
            trainer = Trainer(
                max_epochs=num_epochs,
                callbacks=callbacks,
                devices=num_gpus,
                accelerator="auto",
                # strategy=RayStrategy(num_workers=4, use_gpu=True)
            )
            trainer.fit(model, dm)

        num_samples = 10
        num_epochs = 10
        gpus_per_trial = 1

        data_dir = os.path.join(self.base_dir, self.config["full_data_path"])

        config = {
            "layer_1": tune.choice([32, 64, 128]),
            "layer_2": tune.choice([64, 128, 256]),
            "lr": tune.loguniform(1e-5, 1e-5),
            "batch_size": tune.choice([32, 64, 128]),
        }

        trainable = tune.with_parameters(
            train_ray,
            model=self.model,
            dm=self.dm,
            data_dir=data_dir,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial,
            checkpoint_dir=None,
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

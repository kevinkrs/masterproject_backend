import os
import json
from model.transformer import TransformerModel
from ray import air, tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from pytorch_lightning import Trainer


class HyperParamTuning:
    def __init__(self, config, datamodule):
        self.config = config
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dm = datamodule

    def run(self):
        def train_ray(data_dir=None, num_epochs=10, num_gpus=1):
            metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}
            trainer = Trainer(
                max_epochs=num_epochs,
                gpus=num_gpus,
                #progress_bar_refresh_rate=0,
                callbacks=[TuneReportCallback(metrics, on="validation_end")],
            )
            model = TransformerModel(config).model
            trainer.fit(self.model, self.dm)

        num_samples = 10
        num_epochs = 10
        gpus_per_trial = 1

        data = os.path.join(self.base_dir, self.config["full_data_path"])
        # Download data

        config = {
            "layer_1": tune.choice([32, 64, 128]),
            "layer_2": tune.choice([64, 128, 256]),
            "lr": tune.loguniform(1e-5, 1e-1),
            "batch_size": tune.choice([32, 64, 128]),
        }


        reporter = CLIReporter(
        parameter_columns=["layer_1_size", "layer_2_size", "lr", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])

        """
        trainable = tune.with_parameters(
            train_ray(self.model,self.dm),
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
        """
        tuner = tune.Tuner(
          train_ray,
          tune_config=tune.TuneConfig(
              metric="loss",
              mode="min",
              num_samples=num_samples
          ),
           run_config=air.RunConfig(
            name="tune_test",
            progress_reporter=reporter,
          ),
          param_space=config
        )
         
        results = tuner.fit()
        analysis = results.get_best_result().config
        return analysis

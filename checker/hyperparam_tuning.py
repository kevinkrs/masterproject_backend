from ray import tune
import os
import json
from model.transformer import TransformerModel
from model.utils.dataset_module import TransformerDataModule
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from pytorch_lightning import Trainer


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(base_dir, "config/config.json")) as f:
    config = json.load(f)


def train_ray(config, data_dir=None, num_epochs=10, num_gpus=1):
    model = TransformerModel(config)
    dm = TransformerDataModule()
    metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}
    trainer = Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        progress_bar_refresh_rate=0,
        callbacks=[TuneReportCallback(metrics, on="validation_end")],
    )
    trainer.fit(model, dm)


num_samples = 10
num_epochs = 10
gpus_per_trial = 2

data = os.path.join(base_dir, config["full_data_path"])
# Download data

config = {
    "layer_1": tune.choice([32, 64, 128]),
    "layer_2": tune.choice([64, 128, 256]),
    "lr": tune.loguniform(1e-5, 1e-5),
    "batch_size": tune.choice([32, 64, 128]),
}

trainable = tune.with_parameters(
    train_ray, data_dir=data, num_epochs=num_epochs, num_gpus=gpus_per_trial
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

print(analysis.best_config)

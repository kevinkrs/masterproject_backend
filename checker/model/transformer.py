import pytorch_lightning as pl
import torch
import numpy as np
import mlflow
import os

from typing import Optional, Dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoConfig, AutoModelForSequenceClassification

from .base import BaseModel

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune


# import sys, os
#
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class LModule(pl.LightningModule):
    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            "bert-base-uncased",
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
        )
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, config=self.config
        )
        self.save_hyperparameters()

    def forward(self, **inputs):
        outputs = self.classifier(**inputs)
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)

        self.log("train_loss", outputs[0])
        # outputs includes: loss[0], logits[1], hidden states[2] and attentions [3]
        return outputs[0]

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log("val_loss", outputs[0])

        return outputs[0]

    def validation_epoch_end(self, outputs) -> None:
        avg_val_loss = float(sum(outputs) / len(outputs))
        self.log("avg_val_loss", avg_val_loss)
        mlflow.log_metric("avg_val_loss", avg_val_loss, self.current_epoch)
        print(f"Avg val loss: {avg_val_loss}")
        # ray tune specific
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log("test_loss", outputs[0])
        return outputs[0]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-5)

        return optimizer

    def predict_step(self, batch, batch_idx):
        return self(batch)


# Model Definition: Add BaseModel once module import error fixed
class TransformerModel(BaseModel):
    def __init__(self, config, load_from_ckpt=False):
        self.config = config
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        if load_from_ckpt:
            self.model = LModule.load_from_checkpoint(
                os.path.join(
                    base_dir,
                    config["model_output_path"],
                    f"trained_model_{config['type']}-{config['version']}.ckpt",
                ),
            )

        else:
            self.model = LModule(config["type"])
            model_output_path = os.path.join(base_dir, config["model_output_path"])
            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                dirpath=model_output_path,
                filename=f"trained_model_{config['type']}-{config['version']}",
                save_weights_only=True,
            )
            self.trainer = Trainer(
                max_epochs=config["num_epochs"],
                logger=False,
                accelerator="auto",
                devices=1,
                callbacks=[
                    checkpoint_callback,
                ],
            )

    def train(self, datamodule):
        self.trainer.fit(self.model, datamodule)

    def validate_model(self, datamodule):
        self.trainer.validate(self.model, datamodule)

    def compute_metrics(self, dataloader, split: Optional[str] = None) -> Dict:
        logits = self.predict(dataloader)
        probs = torch.cat(logits, axis=0).cpu().detach().numpy()
        labels = dataloader.dataset.data.columns[7]
        preds = np.argmax(probs, axis=1)
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        auc = roc_auc_score(labels, preds)
        conf_mat = confusion_matrix(labels, preds)
        tn, fp, tp, fn = conf_mat.ravel()
        print(f"Accuracy: {accuracy}, F1: {f1}, AUC: {auc}")
        split_prefix = "" if split is None else split
        return {
            f"{split_prefix} f1": f1,
            f"{split_prefix} accuracy": accuracy,
            f"{split_prefix} auc": auc,
            f"{split_prefix} true negative": tn,
            f"{split_prefix} false negative": fn,
            f"{split_prefix} false positive": fp,
            f"{split_prefix} true positive": tp,
        }

    def predict(self, dataloader):
        logits = []
        self.model.eval()
        # detaching of tensors from current computantional graph
        self.model.cuda()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                output = self.model(
                    input_ids=batch["input_ids"].cuda(),
                    attention_mask=batch["attention_mask"].cuda(),
                    labels=batch["labels"].cuda(),
                )
                logits.append(output[1])  # we only return the logits tensor

        return logits

    # Only required for mlflow to work
    def get_params(self) -> Dict:
        return {}


def train_ray(
    config, datamodule, data_dir=None, num_epochs=10, num_gpus=0, checkpoint_dir=None
):
    model = LModule(config["type"])
    metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}
    callbacks = [TuneReportCallback(metrics, on="validation_end")]
    trainer = Trainer(
        max_epochs=num_epochs,
        callbacks=callbacks,
        devices=num_gpus,
        accelerator="auto",
        # strategy=RayStrategy(num_workers=4, use_gpu=True)
    )
    trainer.fit(model, datamodule)


def tune_bert(data_dir, datamodule):

    num_samples = 10
    num_epochs = 10
    gpus_per_trial = 1

    config = {
        "layer_1": tune.choice([32, 64, 128]),
        "layer_2": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-5, 1e-5),
        "batch_size": tune.choice([32, 64, 128]),
    }

    trainable = tune.with_parameters(
        train_ray,
        datamodule=datamodule,
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

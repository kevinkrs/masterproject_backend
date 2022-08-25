import pytorch_lightning as pl
import torch
import numpy as np
import mlflow
import os

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoConfig, AutoModelForSequenceClassification
from typing import Optional, Dict

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

#from checker.model.base import BaseModel


class LModule(pl.LightningModule):
    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
        )
        self.classifier = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
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
        val_loss, logits = outputs[:2]
        # if self.hparams.num_labels > 1:
        #     preds = torch.argmax(logits, axis=1)
        # elif self.hparams.num_labels == 1:
        #     preds = logits.squeeze()
        #
        # labels = batch["labels"]
        self.log("val_loss", val_loss)
        return outputs[0]

    def validation_epoch_end(self, outputs) -> None:
        avg_val_loss = float(sum(outputs) / len(outputs))
        self.log("avg_val_loss", avg_val_loss)
        mlflow.log_metric("avg_val_loss", avg_val_loss, self.current_epoch)
        print(f"Avg val loss: {avg_val_loss}")

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)

        self.log("test_loss", outputs[0])
        return outputs[0]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-5)

        return optimizer


# Model Definition
class TransformerModel:
    def __init__(self, config, load_from_ckpt=False):
        self.config = config
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        if load_from_ckpt:
            self.model = LModule.load_from_checkpoint(os.path.join(base_dir,
                                                                   config["model_output_path"],
                                                                   "epoch=epoch=0-val_loss=val_loss=0.5034.ckpt"),
                                                      )

        else:
            self.model = LModule(config["type"])
            model_output_path = os.path.join(base_dir, config["model_output_path"])
            checkpoint_callback = ModelCheckpoint(
                            monitor="val_loss",
                            mode="min",
                            dirpath=model_output_path,
                            filename="epoch={epoch}-val_loss={val_loss:.4f}",
                            save_weights_only=True,
                        )
            self.trainer = Trainer(
                max_epochs=config["num_epochs"],
                logger=False,
                accelerator="auto",
                devices=1,
                callbacks=[checkpoint_callback],
            )



    def train(self, datamodule):
        self.trainer.fit(self.model, datamodule)

    def validate_model(self, datamodule):
        self.trainer.validate(self.model, datamodule)

    def compute_metrics(self, datamodule, split: Optional[str] = None) -> Dict:
        logits = self.predict(datamodule)
        probs = torch.cat(logits, axis=0).cpu().detach().numpy()
        labels = datamodule.dataset["val"].features["labels"]
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

    def predict(self, datamodule):
        self.model.eval()
        logits = []
        device = torch.device("mps")
        self.model.to(device)
        # detaching of tensors from current computantional graph
        with torch.no_grad():
            for batch in datamodule:
                output = self.model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    # token_type_ids=batch['token_type_ids'].to(device),
                    labels=batch["labels"].to(device),
                )
                logits.append(output[1])  # we only return the logits tensor

        return logits

    # Only required for mlflow to work
    def get_params(self) -> Dict:
        return {}
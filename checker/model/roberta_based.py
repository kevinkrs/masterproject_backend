import pytorch_lightning as pl
import torch
import numpy as np
import mlflow
import os

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import RobertaConfig, RobertaForSequenceClassification
from typing import Optional, Dict

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

#from checker.model.base import BaseModel


class LModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = RobertaConfig.from_pretrained(
            config["type"],
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
        )
        self.classifier = RobertaForSequenceClassification(self.config)


    def forward(self, input_ids, attention_mask, labels):
        output = self.classifier(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # special_tokens_mask=special_tokens_mask,
            labels=labels,
        )

        return output

    def training_step(self, batch, batch_idx):
        output = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            # token_type_ids=batch['token_type_ids'],
            # special_tokens_mask=batch['special_tokens_mask'],
            labels=batch["label"],
        )

        self.log("train_loss", output[0])
        # output includes: loss[0], logits[1], hidden states[2] and attentions [3]
        return output[0]

    def validation_step(self, batch, batch_idx):
        output = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            # token_type_ids=batch['token_type_ids'],
            # special_tokens_mask=batch['special_tokens_mask'],
            labels=batch["label"],
        )

        self.log("val_loss", output[0])
        return output[0]

    def validation_epoch_end(self, outputs) -> None:
        avg_val_loss = float(sum(outputs) / len(outputs))
        self.log("avg_val_loss", avg_val_loss)
        mlflow.log_metric("avg_val_loss", avg_val_loss, self.current_epoch)
        print(f"Avg val loss: {avg_val_loss}")

    def test_step(self, batch, batch_idx):
        output = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            # token_type_ids=batch['token_type_ids'],
            # special_tokens_mask=batch['special_tokens_mask'],
            labels=batch["label"],
        )

        self.log("test_loss", output[0])
        return output[0]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-5)

        return optimizer


# Model Definition
class RobertaModel:
    def __init__(self, config, load_from_ckpt=False):
        self.config = config
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        if load_from_ckpt:
            self.model = LModule.load_from_checkpoint(os.path.join(base_dir,
                                                                   config["model_output_path"],
                                                                   "epoch=epoch=4-val_loss=val_loss=0.3836.ckpt"),
                                                      config=config)

        else:
            self.model = LModule(config)
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
                accelerator=config["accelerator"],
                devices=1,
                callbacks=[checkpoint_callback],
            )



    def train(self, train_data, val_data):
        train_dataloader = DataLoader(
            train_data,
            batch_size=self.config["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=6,
        )

        val_dataloader = DataLoader(
            val_data, batch_size=16, shuffle=False, pin_memory=True, num_workers=6
        )

        self.trainer.fit(self.model, train_dataloader, val_dataloader)

    def compute_metrics(self, data, split: Optional[str] = None) -> Dict:
        expected_labels = [datapoint["label"] for datapoint in data]
        logits = self.predict(data)
        probs = torch.cat(logits, axis=0).cpu().detach().numpy()
        preds = np.argmax(probs, axis=1)
        accuracy = accuracy_score(expected_labels, preds)
        f1 = f1_score(expected_labels, preds)
        auc = roc_auc_score(expected_labels, preds)
        conf_mat = confusion_matrix(expected_labels, preds)
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

    def predict(self, data):
        dataloader = DataLoader(
            data, batch_size=self.config["batch_size"], pin_memory=True
        )
        self.model.eval()
        logits = []
        device = torch.device("mps")
        self.model.to(device)
        # detaching of tensors from current computantional graph
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                output = self.model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    # token_type_ids=batch['token_type_ids'].to(device),
                    labels=batch["label"].to(device),
                )
                logits.append(output[1])  # we only return the logits tensor

        return logits

    # Only required for mlflow to work
    def get_params(self) -> Dict:
        return {}
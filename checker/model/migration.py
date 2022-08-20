import json
from typing import Optional, Dict, List
import numpy as np
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    RobertaConfig,
)
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from torch.utils.data import DataLoader
import mlflow

from base import BaseModel

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

## Tokenizer function


# with open("config/roberta_v1.json") as f:
#     config = json.load(f)
#
# train_raw = pd.read_csv('data/preprocessed/train_news.csv')
# val_raw = pd.read_csv('data/preprocessed/valid_news.csv')
# test_raw = pd.read_csv('data/preprocessed/test_news.csv')
# train_raw.groupby(["label"]).count()

# Module Definition

with open("config/roberta_v1.json") as f:
    config = json.load(f)


class LModule(pl.LightningModule):
    def __init__(self):
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
class RobertaModel(BaseModel):
    def __init__(self, load_from_ckpt=False):

        if load_from_ckpt:
            self.model = LModule.load_from_checkpoint("saved_models/roberta-base")
        else:
            self.model = LModule()
            checkpoint_callback = ModelCheckpoint(
                monitor="avg_val_loss",
                mode="min",
                dirpath=config["model_output_path"],
                filename="epoch={epoch}-val_loss={val_loss:.4f}",
                save_weights_only=True,
            )

            self.trainer = Trainer(
                max_epochs=config["num_epochs"],
                logger=False,
                # accelerator='gpu',
                devices=1,
                callbacks=[checkpoint_callback],
            )

    def train(self, train_data, val_data):
        train_dataloader = DataLoader(
            train_data,
            batch_size=config["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=2,
        )

        val_dataloader = DataLoader(
            val_data, batch_size=16, shuffle=False, pin_memory=True, num_workers=2
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
        dataloader = DataLoader(data, batch_size=config["batch_size"], pin_memory=True)
        self.model.eval()
        logits = []
        self.model.cuda()
        # detaching of tensors from current computantional graph
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                output = self.model(
                    input_ids=batch["input_ids"].cuda(),
                    attention_mask=batch["attention_mask"].cuda(),
                    # token_type_ids=batch['token_type_ids'].cuda(),
                    labels=batch["label"].cuda(),
                )
                logits.append(output[1])  # we only return the logits tensor

        return logits

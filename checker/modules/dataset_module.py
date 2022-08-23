import pytorch_lightning as pl
import json
import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

class TransformerDataModule(pl.LightningDataModule):

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(base_dir, "checker/config/config.json")) as f:
            self.config = json.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["type"], padding_side="right")

    def setup(self, dataset, stage: str):
        self.dataset = dataset
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.tokenizer_base,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.config["batch_size"], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset["val"], batch_size=16, shuffle=False)

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def tokenizer_base(self, dataset):

        features = self.tokenizer(
            dataset["title"],
            dataset["statementdate"],
            # return_tensors="pt",
            padding="max_length",
            add_special_tokens=True,
            max_length=self.config["max_seq_length"],
            return_token_type_ids=True,
            return_attention_mask=True,
            # return_special_tokens_mask=True,
            truncation=True,
        )

        features["labels"] = dataset["label"]

        return features

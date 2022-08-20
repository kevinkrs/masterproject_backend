import json
import os
from transformers import RobertaTokenizerFast

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(base_dir, "../config/roberta_v1.json")) as f:
    config = json.load(f)


def tokenizer_base(data):
    tokenizer = RobertaTokenizerFast.from_pretrained(config["type"], padding_side="right")
    tokenized_data = []

    for idx, statement in data.iterrows():
        tokenized = tokenizer(
            statement[2],
            return_tensors="pt",
            # this one is optional. If nothing set, a python list of integers will be returned
            padding="max_length",
            add_special_tokens=True,
            max_length=config["max_seq_length"],
            return_token_type_ids=True,
            return_attention_mask=True,
            return_special_tokens_mask=True,
            truncation=True,
        )

        tokenized_data.append(
            {
                "input_ids": tokenized.data["input_ids"].squeeze(),
                "attention_mask": tokenized.data["attention_mask"].squeeze(),
                "token_type_ids": tokenized.get("token_type_ids"),
                "special_tokens_mask": tokenized.get("special_tokens_mask"),
                "label": int(statement["label"]),
            }
        )  # TODO: Check if cast to int is enough

    return tokenized_data

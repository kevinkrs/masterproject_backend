import json
import os
from transformers import AutoTokenizer

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(base_dir, "checker/config/config.json")) as f:
    config = json.load(f)


def tokenizer_base(data):
    tokenizer = AutoTokenizer.from_pretrained(config["type"], padding_side="right")

    features = tokenizer(
        data["title"],
        data["short_text"],
        # return_tensors="pt",
        padding="max_length",
        add_special_tokens=True,
        max_length=config["max_seq_length"],
        return_token_type_ids=True,
        return_attention_mask=True,
        #return_special_tokens_mask=True,
        truncation=True,
    )

        # tokenized_data.append(
        #     {
        #         "input_ids": tokenized.data["input_ids"].squeeze(),
        #         "attention_mask": tokenized.data["attention_mask"].squeeze(),
        #         "token_type_ids": tokenized.get("token_type_ids").squeeze(),
        #        # "special_tokens_mask": tokenized.get("special_tokens_mask"),
        #         "labels": int(statement["label"]),
        #     }
        #)

    return features

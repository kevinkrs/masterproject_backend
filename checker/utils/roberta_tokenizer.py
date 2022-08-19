import json
from transformers import RobertaTokenizerFast

with open("config/roberta_config_v1.json") as f:
    config = json.load(f)

tokenizer = RobertaTokenizerFast.from_pretrained(config["type"], padding_side="right")


class RobertaTokenizer:
    def tokenizer_bert(data):
        tokenized_data = []

        for idx, statement in data.iterrows():
            tokenized = tokenizer.encode_plus(
                statement.text,
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

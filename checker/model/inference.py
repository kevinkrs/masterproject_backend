import os
import json
import torch

from transformers import BertTokenizerFast, BertForSequenceClassification
from api.DataModel import DataModel
from transformer import LModule


base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(base_dir, "checker/config/config.json")) as f:
    config = json.load(f)

class Inference:
    def __init__(self):
        self.model = LModule('bert-base-uncased')
        checkpoint = torch.load(
            os.path.join(base_dir, config['model_output_path'],
                         f"trained_model_{config['type']}-{config['version']}.ckpt"),
            map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint["state_dict"])
    def get_prediction(self, data):
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        tokenized_data = tokenizer(data.text, data.statementdate, return_attention_mask=True, return_tensors="pt",
                                   padding="max_length", )

        # Load model from checkpoint
        self.model.eval()
        # 1. Get prediction as list
        logits = []
        with torch.no_grad():
            output = self.model(**tokenized_data)
            logits = logits.append(output.logits)

        print(logits)
        return logits
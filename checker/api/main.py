import os
import json
import torch
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import BertTokenizerFast, BertForSequenceClassification
from checker.model.transformer import LModule
from pydantic import BaseModel
from typing import Optional

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append("/checker/model")

app = FastAPI()

origins = [
    "http://localhost:8080",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(base_dir, "checker/config/config.json")) as f:
    config = json.load(f)


class DataModel(BaseModel):
    text: str
    statementdate: str
    author: Optional[str]
    source: Optional[str]

class Inference:
    def __init__(self):
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        checkpoint = torch.load(
            os.path.join(base_dir, config['model_output_path'],
                         f"trained_model_{config['type']}-{config['version']}.ckpt"),
            map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint["state_dict"])
    def get_prediction(self, data: DataModel):
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        tokenized_data = tokenizer(data['text'], data['statementdate'], return_attention_mask=True, return_tensors="pt",
                                   padding="max_length", )

        # Load model from checkpoint
        self.model.eval()
        # 1. Get prediction as list
        logits = []
        with torch.no_grad():
            output = self.model(**tokenized_data)
            logits = logits.append(output.logits)

        return logits


@app.post("/api/predict")
# SCHEMA: text - statementdate
def inference( data: DataModel):
    # TODO: Could get quite slow, since model is intialized every request
    predict = Inference()
    logits = predict.get_prediction(data)
    # 2. Transform list to torch tensor
    preds_tp = torch.cat(logits, dim=0)
    # 3. Run Softmax to get max values = Probabilities
    probs = torch.nn.functional.softmax(preds_tp, dim=-1).squeeze()
    probs_np = torch.cat(logits, axis=0).cpu().detach().numpy()
    preds = np.argmax(probs_np, axis=1)
    predicted_class_id = preds.max().item()
    pred_label = predict.model.config.id2label[predicted_class_id]
    if pred_label == 'LABEL_0':
        label = 'FAKE'
    else:
        label = 'TRUE'
    response = f"{label}: {probs.max}"
    print(f"{label}: {probs.max}")
    print(probs)

    return response


# SCHEMA of json body
# {
#   title: "",
#   date: "",
#   source: "", [OPTIONAL]
#   person: "", [OPTIONAL]
# }




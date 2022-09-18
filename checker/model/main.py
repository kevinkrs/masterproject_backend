
import torch
import numpy as np
import logging
import os
import json

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from api.DataModel import DataModel
from inference import Inference
from transformer import LModule
from transformers import BertTokenizerFast


logger = logging.getLogger('inference')
app = FastAPI()

origins = [
    "http://localhost:8080",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(base_dir, "checker/config/config.json")) as f:
    config = json.load(f)



def get_prediction(data):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokenized_data = tokenizer(data.text, data.statementdate, return_attention_mask=True, return_tensors='pt',
                               padding="max_length", )

    # Load model from checkpoint
    model = LModule('bert-base-uncased')
    checkpoint = torch.load(
        os.path.join(base_dir, config['model_output_path'], f"trained_model_{config['type']}-{config['version']}.ckpt"),
        map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    # 1. Get prediction as list
    logits = []
    with torch.no_grad():
        output = model(**tokenized_data)
        logits.append(output.logits)
    # # 2. Transform list to torch tensor
    preds_tp = torch.cat(logits, dim=0)
    # # 3. Run Softmax to get max values = Probabilities
    probs = torch.nn.functional.softmax(preds_tp, dim=-1).squeeze()
    preds = probs.argmax()
    predicted_class_id = preds.max().item()
    pred_label = model.config.id2label[predicted_class_id]
    if pred_label == 'LABEL_0':
        label = 'FAKE'
    else:
        label = 'TRUE'

    return label, probs.numpy().tolist(), probs.numpy().max().tolist()



@app.post("/api/predict", response_class=ORJSONResponse)
# SCHEMA: text - statementdate
def inference(data: DataModel):
    # TODO: Could get quite slow, since model is intialized every request
    label, probs, prob_max = get_prediction(data)
    response = {
        'label': label,
        'probs': probs,
        'prob_max': prob_max
    }

    return ORJSONResponse(response)

import torch
import numpy as np
import logging
import os
import json

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from utils.datamodel import DataModel
from model.transformer import LModule
from transformers import BertTokenizerFast

from api.inference import Inference


logger = logging.getLogger("inference")
app = FastAPI()

origins = [
    "http://localhost:8080",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(base_dir, "config/config.json")) as f:
    config = json.load(f)

model = LModule("bert-base-uncased")
inference = Inference(config=config, model=model)


@app.post("/api/predict", response_class=ORJSONResponse)
# SCHEMA: text - statementdate
def inference(data: DataModel):
    # TODO: Could get quite slow, since model is intialized every request
    label, probs, prob_max = inference.get_prediction(data)
    response = {"label": label, "probs": probs, "prob_max": prob_max}

    return ORJSONResponse(response)


@app.get("api/attentions")
def attentions():
    pass

import torch
import numpy as np
import logging
import os
import json

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from utils.datamodels import DataModel

# from model.transformer import LModule
from transformers import AutoModelForSequenceClassification

from api.inference import Inference
from api.search import SemanticSearch

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

# model = LModule("bert-base-uncased")
# inference = Inference(config=config, model=model)
model = AutoModelForSequenceClassification.from_pretrained(config["type"])
inference = Inference(config=config, model=model)
search = SemanticSearch()


@app.post("/api/predict", response_class=ORJSONResponse)
def inference(data: DataModel):
    # model = LModule(config['type'])
    inference = Inference(config=config, model=model)
    # TODO: Could get quite slow, since model is intialized every request
    label, probs, prob_max = inference.get_prediction(data)
    response = {"label": label, "probs": probs, "prob_max": prob_max}

    return ORJSONResponse(response)


@app.post("api/search", response_class=ORJSONResponse)
def search(data: DataModel):
    response = search.get_similar(data)
    return ORJSONResponse(response)

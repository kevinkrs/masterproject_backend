import pandas as pd
import torch
import numpy as np
import logging
import os
import json

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from utils.datamodels import DataModel, TrainModel
from model.transformer import LModule, TransformerModel

from api.inference import Inference
from api.search import SemanticSearch
from transformers import AutoModelForSequenceClassification

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


model = TransformerModel(config, load_from_ckpt=True).model
inference_api = Inference(config=config, model=model)
semantic_search = SemanticSearch()


@app.post("/api/predict", response_class=ORJSONResponse)
def inference(data: DataModel):
    label, probs, prob_max = inference_api.get_prediction(data)
    response = {"label": label, "probs": probs, "prob_max": prob_max}

    return ORJSONResponse(response)


@app.post("/api/search", response_class=ORJSONResponse)
def search(data: DataModel):
    response = semantic_search.get_similar(data)
    return ORJSONResponse(response)


@app.post("/api/training")
def getUserData(data: TrainModel):
    data_dir = os.path.join(base_dir, "data", "userHistory", "userHistory.csv")
    if os.path.exists(os.path.join(base_dir, "data", "userHistory")):
        old = pd.read_csv(data_dir, sep=",")
        new = pd.DataFrame(jsonable_encoder(data))
        save = pd.concat([old, new])
        save.to_csv(data_dir)
    else:
        os.makedirs(os.path.join(base_dir, "data", "userHistory"), exist_ok=True)
        new = pd.DataFrame(jsonable_encoder(data))
        new.to_csv(data_dir)
    return True

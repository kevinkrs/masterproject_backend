import os
import json
import torch

from torch.utils.data import DataLoader
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from checker.model.transformer import TransformerModel
from checker.modules.dataset_module import TransformerDataModule
from datasets import load_dataset

app = FastAPI()

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

def get_prediction(features):
    model = TransformerModel(config, load_from_ckpt=True)
    dataloader = DataLoader(features, batch_size=64, shuffle=False)
    logits = model.predict(dataloader)
    predictions = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class_id = logits.argmax().item()
    pred_label = model.config.id2label[predicted_class_id]
    print(f"{pred_label}: {predictions[0][predicted_class_id]}")
    return (pred_label, predictions[0][predicted_class_id])


@app.post("/api/predict")
# TODO: Not sure if this is the right way to load and tokenize data, since load_dataset is quite complex
async def inference(data: json):
    data_files = {"inference": data}
    datamodule = TransformerDataModule()
    dataset = load_dataset("json", data_files=data_files, split="inference")
    features = datamodule.tokenizer_base(dataset)
    resp = get_prediction(features)

    return resp


# SCHEMA of json body
# {
#   title: "",
#   date: "",
#   source: "", [OPTIONAL]
#   person: "", [OPTIONAL]
# }




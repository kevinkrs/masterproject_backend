import pandas as pd
import torch
import os
import json
import numpy as np
import requests

from checker.model.transformer import LModule
from checker.utils.dataset_module import TransformerDataModule
from pytorch_lightning.trainer import Trainer
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizerFast
from checker.utils.datamodel import DataModel


def test_inference_mode():

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    with open(os.path.join(base_dir, "checker/config/config.json")) as f:
        config = json.load(f)

    raw = {
        "text": "Texas public high school graduation rate is at 90% overall.",
        "statementdate": "2022-07-29",
    }

    data = DataModel(**raw)

    # Get tokenizer and tokenize input data
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokenized_data = tokenizer(
        data.text,
        data.statementdate,
        return_attention_mask=True,
        return_tensors="pt",
        padding="max_length",
    )

    # Load model from checkpoint
    model = LModule("bert-base-uncased")
    checkpoint = torch.load(
        os.path.join(
            base_dir,
            config["model_output_path"],
            f"trained_model_{config['type']}-{config['version']}.ckpt",
        ),
        map_location=torch.device("cpu"),
    )
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
    probs_np = torch.cat(logits, axis=0).cpu().detach().numpy()
    preds = np.argmax(probs_np, axis=1)
    predicted_class_id = preds.max().item()
    pred_label = model.config.id2label[predicted_class_id]
    if pred_label == "LABEL_0":
        label = "FAKE"
    else:
        label = "TRUE"
    print(f"{label}: {probs.max()}")
    print(probs.numpy())


def test_inference_api():
    headers = {"Content-Type": "application/json"}
    data = {
        "text": "When the New York State Senate voted to legalize abortion in 1970, 12 Republican senators voted in favor of it.",
        "statementdate": "30-06-2022",
    }
    prediction = requests.post(
        "http://127.0.0.1:8000/api/predict", headers=headers, json=data
    )
    response = json.loads(prediction.content)
    print(response)

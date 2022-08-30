import pandas as pd
import torch
import os
import json

from checker.model.transformer import TransformerModel
from checker.modules.dataset_module import TransformerDataModule
from torch.utils.data import DataLoader

def test_inference_mode():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    with open(os.path.join(base_dir, "checker/config/config.json")) as f:
        config = json.load(f)

    data = {
            'title': 'Donald trump states that he still is the president of the united states',
            'statementdate': '2022-07-29'}

    with open('test_json.json', 'w') as f:
        json.dump(data, f)




    def predict(config, features):
        dataloader = DataLoader(features, batch_size=16)
        logits = []
        model = TransformerModel(config, load_from_ckpt=True)
        # detaching of tensors from current computantional graph
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                output = model(
                    input_ids=batch["input_ids"].to_device('auto'),
                    attention_mask=batch["attention_mask"].to_device('auto'),
                    labels=batch["labels"].to_device('auto'),
                )
                logits.append(output[1])  # we only return the logits tensor

        return logits


    datamodule = TransformerDataModule()
    features = datamodule.tokenizer_base(data)


    # 1. Get prediction as list
    logits = predict(config, features)
    # 2. Transform list to torch tensor
    preds = torch.cat(logits, dim=0)
    # 3. Run Softmax to get max values = Probabilities
    probs = torch.nn.functional.softmax(preds, dim=-1)

    print(probs)
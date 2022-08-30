import pandas as pd
import torch
import os
import json

from checker.model.transformer import LModule
from checker.modules.dataset_module import TransformerDataModule
from pytorch_lightning.trainer import Trainer
from torch.utils.data import DataLoader

def test_inference_mode():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    with open(os.path.join(base_dir, "checker/config/config.json")) as f:
        config = json.load(f)

    data = {
            'title': 'Donald trump states that he still is the president of the united states',
            'statementdate': '2022-07-29'}



    datamodule = TransformerDataModule()
    features = datamodule.tokenizer_base(data)
    dataloader = DataLoader(features, batch_size=1)
    model = LModule(os.path.join(base_dir, config['model_output_path'], f"trained_model_{config['type']}-{config['version']}.ckpt"))
    trainer = Trainer()

    # 1. Get prediction as list
    logits = trainer.predict(model, dataloader)
    # 2. Transform list to torch tensor
    preds = torch.cat(logits, dim=0)
    # 3. Run Softmax to get max values = Probabilities
    probs = torch.nn.functional.softmax(preds, dim=-1)

    print(probs)


    # Load model directly via torch
    # model = LModule('bert-base-uncased')
    # checkpoint = torch.load(os.path.join(base_dir, config['model_output_path'], f"trained_model_{config['type']}-{config['version']}.ckpt"),
    #                         map_location=torch.device('cpu'))
    # model.load_state_dict(checkpoint["state_dict"])
    # model.eval()
    # logits = []
    # with torch.no_grad():
    #     for batch_idx, batch in enumerate(dataloader):
    #         output = model(
    #             input_ids=batch["input_ids"],
    #             attention_mask=batch["attention_mask"],
    #             labels=batch["labels"],
    #         )
    #         logits.append(output[1])  # we only return the logits tensor

import pandas as pd
import torch
import os
import json

from checker.model.roberta_based import RobertaModel
from checker.utils.roberta_tokenizer import tokenizer_base

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
with open(os.path.join(base_dir, "checker/config/roberta_v1.json")) as f:
    config = json.load(f)


model = RobertaModel(config, load_from_ckpt=True)

data = [['The corona virus pandemic is a lie', False],
        [
            'It costs us about $33,000 a year (on average nationally) to lock somebody up. In California it costs about $75,000 a year.',
            False],
        [
            'The Biden administration’s American Jobs Plan will be “the biggest non-defense investment in research and development in the history of our country.',
            False],
        ['Hillary replaces Kamala', False]]

df = pd.DataFrame(data, columns=['text', 'label'])

inputs = tokenizer_base(df)

outputs = model.predict(inputs)


# 1. Get prediction as list
logits = model.predict(inputs)
# 2. Transform list to torch tensor
preds = torch.cat(logits, dim=0)
# 3. Run Softmax to get max values = Probabilities
probs = torch.nn.functional.softmax(preds, dim=-1)

print(probs)
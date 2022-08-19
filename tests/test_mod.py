import torch
import numpy as np

from checker.model.roberta_based import RobertaModel


model = RobertaModel()
# 1. Get prediction as list
logits = model.predict(inputs)

# 2. Transform list to torch tensor
preds = torch.cat(logits, dim=0)

# 3. Run Softmax to get max values = Probabilities
probs = torch.nn.functional.softmax(preds, dim=-1)
# 4. Get argmax (label) as numpy array or torch tensor []
probs_max_pre = torch.cat(logits, axis=0).cpu().detach().numpy()
probs_max = np.argmax(probs_max_pre, axis=1)
# -> Same with torch possible
test = torch.argmax(preds, axis=1)

import os
import json
import torch

from transformers import AutoTokenizer

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Inference:
    """Inference class loading the model based on trained model as well as the required tokenizer for inference task.
    :arg config
    :arg model"""

    def __init__(self, config, model):
        # Generic pytorch model load from state_dict
        self.model = model
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["type"])

    def get_prediction(self, data):
        """Prediction function tokenizing the input data, setting the model into eval mode and running inference.
        Function transforms logits into probabilitites and returns the labels, the probabilities and the max probobility.
        @:arg data
        """
        tokenized_data = self.tokenizer(
            data.text,
            data.statementdate,
            return_attention_mask=True,
            return_tensors="pt",
            padding="max_length",
        )

        # Load model from checkpoint
        self.model.eval()
        # 1. Get prediction as list
        logits = []
        with torch.no_grad():
            output = self.model(**tokenized_data)
            logits.append(output.logits)
        # # 2. Transform list to torch tensor
        preds_tp = torch.cat(logits, dim=0)
        # # 3. Run Softmax to get max values = Probabilities
        probs = torch.nn.functional.softmax(preds_tp, dim=-1).squeeze()
        preds = probs.argmax()
        predicted_class_id = preds.max().item()
        pred_label = self.model.config.id2label[predicted_class_id]
        if pred_label == "LABEL_0":
            label = "FAKE"
        else:
            label = "TRUE"

        return label, probs.numpy().tolist(), probs.numpy().max().tolist()

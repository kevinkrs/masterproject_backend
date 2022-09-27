import os
import json
import torch

from transformers import BertTokenizerFast


base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Inference:
    def __init__(self, config, model):
        self.model = model
        self.config = config
        self.checkpoint = torch.load(
            os.path.join(
                base_dir,
                config["model_output_path"],
                f"trained_model_{config['type']}-{config['version']}.ckpt",
            ),
            map_location=torch.device("cpu"),
        )
        self.model.load_state_dict(self.checkpoint["state_dict"])
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def get_prediction(self, data):
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

    def get_output(self, data):
        tokenized_data = self.tokenizer(
            data.text,
            data.statementdate,
            return_attention_mask=True,
            return_tensors="pt",
            padding="max_length",
        )

        # Load model from checkpoint
        self.model.eval()
        with torch.no_grad():
            output = self.model(**tokenized_data)

        return output

import pandas as pd
import os
import json
import torch
from datasets import Dataset

from transformers import AutoModel, AutoTokenizer


class SemanticSearch:

    def __init__(self):
        self.base_dir = base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(self.base_dir, "config/config.json")) as f:
            self.config = json.load(f)
        # check if embeddings csv exists
        self.model, self.tokenizer = SemanticSearch.load_model()

        embedding_path = os.path.join(
            self.base_dir, self.config["embedding_path"])
        if not os.path.exists(embedding_path) or self.config["update_data"]:
            # if not, create it
            self.create_embeddings()
        self.embeddings = pd.read_csv(embedding_path)

    def create_embeddings(self):
        # create embeddings
        raw_path = os.path.join(self.base_dir, self.config["raw_data_path"])
        facts_dataset_df = pd.read_csv(raw_path)
        facts_dataset = Dataset.from_pandas(facts_dataset_df)
        facts_dataset = facts_dataset.map(SemanticSearch.concatenate_text)
        embeddings_dataset = facts_dataset.map(
            lambda x: {"embeddings": self.get_embeddings(
                x["text"]).detach().cpu().numpy()[0]}
        )
        embeddings_dataset.add_faiss_index(column="embeddings")
        embeddings_dataset.to_csv(os.path.join(
            self.base_dir, self.config["embedding_path"]))

    def get_similar(self, data):
        num_results = data.get("num_results", 5)
        embeddings = self.get_embeddings(data["text"]).cpu().detach().numpy()
        scores, samples = self.embeddings_dataset.get_nearest_examples(
            "embeddings", embeddings, k=num_results
        )
        samples_df = pd.DataFrame.from_dict(samples)
        samples_df["scores"] = scores
        samples_df.sort_values("scores", ascending=False, inplace=True)
        return samples_df.to_dict(orient="records")

    @staticmethod
    def concatenate_text(fatcs):
        return {
            "text": fatcs["title"] + " from the " + fatcs["factcheckdate"] #.strftime('%Y-%m-%d')
            + " \n "
            + str(fatcs["long_text"] or '')
            + " \n "
            + str(fatcs["short_text"] or '')
        }

    @staticmethod
    def load_model():
        # load model
        model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        model = AutoModel.from_pretrained(model_ckpt)
        device = torch.device("cuda")
        model.to(device)
        return model, tokenizer

    def get_embeddings(self, text_list):
        encoded_input = self.tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input = {k: v.to(torch.device)
                         for k, v in encoded_input.items()}
        model_output = self.model(**encoded_input)
        return SemanticSearch.cls_pooling(model_output)

    @staticmethod
    def cls_pooling(model_output):
        return model_output.last_hidden_state[:, 0]

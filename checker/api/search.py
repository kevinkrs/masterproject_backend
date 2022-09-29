import pandas as pd
import os
import json
import torch
from datasets import Dataset

from transformers import AutoModel, AutoTokenizer


class SemanticSearch:
    def __init__(self):
        self.base_dir = base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        with open(os.path.join(self.base_dir, "config/config.json")) as f:
            self.config = json.load(f)
        # check if embeddings csv exists
        self.model, self.tokenizer = SemanticSearch.load_model()

        embedding_path = os.path.join(self.base_dir, self.config["embedding_path"])
        if not os.path.exists(embedding_path) or self.config["update_data"]:
            # if not, create it
            self.create_embeddings()

        self.embeddings_dataset = Dataset.load_from_disk(embedding_path)
        self.embeddings_dataset.add_faiss_index(column="embeddings")

    def create_embeddings(self):
        # create embeddings
        raw_path = os.path.join(self.base_dir, self.config["full_data_path"])
        facts_dataset_df = pd.read_csv(raw_path)
        facts_dataset = Dataset.from_pandas(facts_dataset_df)
        facts_dataset = facts_dataset.map(SemanticSearch.concatenate_text)
        embeddings_dataset = facts_dataset.map(
            lambda x: {
                "embeddings": self.get_embeddings(x["text"]).detach().cpu().numpy()[0]
            }
        )

        # remove file name from path
        embedding_dir = os.path.dirname(self.config["embedding_path"])
        os.makedirs(os.path.join(self.base_dir, embedding_dir), exist_ok=True)
        embeddings_dataset.save_to_disk(
            os.path.join(self.base_dir, self.config["embedding_path"])
        )

    def get_similar(self, data):
        num_results = 5
        search_data = {}
        #check if source is filled
        if data.source:
            search_data["sources"] = data.source
            search_data["url"] = data.source
        else:
            # if not, empty
            search_data["sources"] = ""
            search_data["url"] = ""

        #check if author is filled
        if data.author:
            search_data["source"] = data.author
        else:
            # if not, empty
            search_data["source"] = ""

        search_data["title"] = data.text

        search_data["statementdate"] = data.statementdate
        search_data["factchecker"] = ""
        search_data["factcheckdate"] = ""
        search_data = SemanticSearch.concatenate_text(search_data)
        print(search_data["text"])
        embeddings = self.get_embeddings(search_data["text"]).cpu().detach().numpy()
        scores, samples = self.embeddings_dataset.get_nearest_examples(
            "embeddings", embeddings, k=num_results
        )
        samples_df = pd.DataFrame.from_dict(samples)
        samples_df["scores"] = scores
        samples_df.sort_values("scores", ascending=True, inplace=True)
        return samples_df.to_dict(orient="records")

    @staticmethod
    def concatenate_text(fatcs):
        return {
            "text": fatcs["title"]
                    + " from the "
                    + fatcs["statementdate"] + " " + fatcs["factcheckdate"]
                    + " by "
                    + fatcs["factchecker"] + " " + fatcs["url"]
                    + " \n "
                    + fatcs["source"]
                    + " \n "
                    + fatcs["sources"]
        }

    @staticmethod
    def load_model():
        # load model
        model_ckpt = "sentence-transformers/all-mpnet-base-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        model = AutoModel.from_pretrained(model_ckpt)
        device = torch.device("cpu")
        model.to(device)
        return model, tokenizer

    def get_embeddings(self, text_list):
        encoded_input = self.tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        )
        device = torch.device("cpu")
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        model_output = self.model(**encoded_input)
        return SemanticSearch.cls_pooling(model_output)

    @staticmethod
    def cls_pooling(model_output):
        return model_output.last_hidden_state[:, 0]

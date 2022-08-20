import logging
import pymongo
from pytunneling import TunnelNetwork
import pandas as pd
import json
import os
import nltk
import secrets

nltk.download("stopwords")
from nltk.corpus import stopwords
import re

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


with open(os.path.join(base_dir, "../config/roberta_v1.json")) as f:
    config = json.load(f)
logger = logging.getLogger()


def load_data_from_db(path: str):
    username = secrets.USERNAME
    passwort = secrets.PASSWORD
    host = secrets.HOST
    host2 = secrets.HOST2
    ssh_username = secrets.SSH_USERNAME
    ssh_password = secrets.SSH_PASSWORD

    tunnel_info = [
        {
            "ssh_address_or_host": host,
            "ssh_username": username,
            "ssh_password": passwort,
        },
        {
            "ssh_address_or_host": host2,
            "ssh_username": ssh_username,
            "ssh_password": ssh_password,
        },
    ]

    with TunnelNetwork(
        tunnel_info=tunnel_info, target_ip="127.0.0.1", target_port=27017
    ) as tn:
        logger.info(f"Tunnel available at localhost:{tn.local_bind_port}")
        client = pymongo.MongoClient(
            host="localhost",
            port=tn.local_bind_port,
            username="mongoadmin",
            password="mongoadmin",
        )
        logger.info(client.list_database_names())
        facts_db = client["facts"]
        politifact = facts_db.politifact
        df = pd.DataFrame(list(politifact.find()))
        # Write raw dataframe
        df.to_csv(path)

        logger.info(f"Dataframe successfully written as csv to {path}")


def text_preprocessing(s: str):
    """
    - Lowercase the sentence
    - Change "'t" to "not"
    - Remove "@name"
    - Remove other special characters
    - Remove stop words except "not" and "can"
    - Remove trailing whitespace
    """
    s = s.lower()

    # Remove some special characters
    s = re.sub(r"([\;\:\|•«\n])", " ", s)

    # Remove quotes
    s = re.sub(r'([\'"`])\s*\1\s*', "", s)

    # Remove stopwords except 'not' and 'can'
    s = " ".join(
        [
            word
            for word in s.split()
            if word not in stopwords.words("english") or word in ["not", "can"]
        ]
    )
    # Remove trailing whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s


def drop_empty(df):
    df_cleaned = df.dropna()
    return df_cleaned


def get_binary_label(label: str) -> bool:
    if label in {"pants-fire", "barely-true", "false"}:
        return False
    elif label in {"true", "half-true", "mostly-true"}:
        return True


def preprocess_summarizer(text):
    from transformers import pipeline

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    s = summarizer(text, max_length=512, min_length=20)

    return s


def concat_columns(df, config):
    cols_title = ["title", "statementdate", "person", "source"]  # Current data
    cols_short_text = ["short_text", "statementdate", "person", "source"]

    if config["train_target"] == "title":
        cols = cols_title
    elif config["train_target"] == "short_text":
        cols = cols_short_text
    else:
        cols = cols_long_text

    df["text"] = (
        df[cols].copy(True).apply(lambda row: " ".join(row.values.astype(str)), axis=1)
    )

    return df


def preprocess_cleaning(raw_path, config):
    names = [
        "_id",
        "title",
        "url",
        "person",
        "statementdate",
        "source",
        "label",
        "factcheckdate",
        "factchecker",
        "sources",
        "long_text",
        "short_text",
        "text",
    ]

    df_raw = pd.read_csv(raw_path, sep=",")
    df_raw = df_raw.drop(index=0)
    # df_raw = df_raw[:100] # Only for debug
    df_cleaned = drop_empty(df_raw)
    df_with_target = concat_columns(df_cleaned, config)

    # Apply binary label
    df_with_target["label"] = df_with_target.label.apply(get_binary_label)
    df = df_with_target[names]

    df["text"] = df.text.astype(str)

    # [OPTIONAL]
    if config["long_text_summarization"]:
        df["text"] = df.text.apply(preprocess_summarizer)

    df["text"] = df.text.apply(text_preprocessing)

    ## Save cleaned full dataset
    df.to_csv("data/full/df_cleaned.csv")

    return df


def create_model_data(raw_path, train_path, valid_path, test_path, config):
    df = preprocess_cleaning(raw_path, config)

    train_valid = df.sample(frac=0.9, random_state=12)
    train = train_valid.sample(frac=(0.8 / 0.9), random_state=12)  # 80%
    valid = train_valid.drop(train.index)  # 10%
    test = df.drop(train_valid.index)  # 10%

    train.to_csv(train_path)
    valid.to_csv(valid_path)
    test.to_csv(test_path)

    logger.info("Successfully created test, valid and train dataset")

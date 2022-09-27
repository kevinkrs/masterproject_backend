import pymongo
import pandas as pd
import os, json

from pytunneling import TunnelNetwork

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_news(self, config_secrets):
    username = config_secrets.USERNAME
    passwort = config_secrets.PASSWORD
    host = config_secrets.HOST
    host2 = config_secrets.HOST2
    ssh_username = config_secrets.SSH_USERNAME
    ssh_password = config_secrets.SSH_PASSWORD

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
        client = pymongo.MongoClient(
            host="localhost",
            port=tn.local_bind_port,
            username="mongoadmin",
            password="mongoadmin",
        )
        facts_db = client["facts"]
        politifact = facts_db.politifact
        df = pd.DataFrame(list(politifact.find()))
        df["statementdate"] = pd.to_datetime(df["statementdate"])
        df = df.sort_values(by="statementdate")

    return df.to_json(default_handler=str)


def get_news_from_csv(config):

    RAW_PATH = config["raw_data_path"]

    df = pd.read_csv(os.path.join(base_dir, RAW_PATH), header=0, sep=",")
    df["statementdate"] = pd.to_datetime(df["statementdate"])
    df = df.sort_values(by="statementdate")

    return df[:31]

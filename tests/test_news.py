import os
import json
import requests
import pandas as pd
import pymongo

from bson.json_util import dumps
from checker.config import config_secrets
from pytunneling import TunnelNetwork

from checker.api.news import get_news_from_csv


def test_news_fetching():
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
        politifact = facts_db.politifact.find().sort("statementdate", -1).limit(20)
        response = dumps(politifact)

    print("finished")

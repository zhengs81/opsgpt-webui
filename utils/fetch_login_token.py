import requests
import json
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.joinpath(".env"))

def fetch_login_token():
    local_token = os.environ["BIZSEER_TOKEN"]
    if local_token.startswith("Bearer"):
        return local_token
    url = os.environ["BIZSEER_LOGIN_URL"]
    username = os.environ["BIZSEER_LOGIN_USERNAME"]
    password = os.environ["BIZSEER_LOGIN_PASSWORD"]

    payload = json.dumps({
        "username": username,
        "password": password
    })

    headers = {
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json',
    'Accept': '*/*',
    'Host': '10.0.80.239:8088',
    'Connection': 'keep-alive'
    }

    response = requests.request("POST", url, headers=headers, data=payload, timeout=5)
    
    token = "Bearer " + response.json()['data']['token']

    return token 

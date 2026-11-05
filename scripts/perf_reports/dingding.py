
from pprint import pprint
import hmac
import hashlib
import base64
import time
import requests
import json
import os
import sys
from tabulate import tabulate
from datetime import date


access_token = os.environ['PERF_DINGDING_ACCESS_TOKEN']
secret = os.environ['PERF_DINGDING_SERCRET']

def generate_signature(timestamp, secret):
    secret_enc = secret.encode('utf-8')
    string_to_sign = f'{timestamp}\n{secret}'
    string_to_sign_enc = string_to_sign.encode('utf-8')
    hmac_code = hmac.new(secret_enc, string_to_sign_enc, hashlib.sha256).digest()
    sign = base64.b64encode(hmac_code)
    return sign.decode('utf-8')


def send_message_to_dingtalk(access_token, message):
    timestamp = str(round(time.time() * 1000))
    signature = generate_signature(timestamp, secret)
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        "msgtype": "markdown",
        "markdown": {
            "title": "perf report from vsag",
            "text": message,
        }
    }
    url = "https://oapi.dingtalk.com/robot/send?"
    url = url + f"&access_token={access_token}" + f"&timestamp={timestamp}" + f"&sign={signature}"
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(response.text)


if len(sys.argv) < 2:
    print("Usage: python dingding.py <filename>")
    sys.exit(1)

filenames = sys.argv[1:]

try:
    rows = list()
    for filename in filenames:
        with open(filename, 'r') as file:
            origin_results = json.loads(file.read())
            # pprint(origin_results)
            for name, result in origin_results.items():
                tps = int(result["tps"])
                qps = int(result["qps"])
                rt = round(result["latency_avg(ms)"], 3)
                recall = round(result["recall_avg"], 3)
                score = tps + qps
                rows.append([name, tps, qps, rt, recall, score])
            rows.append(["", "", "", "", "", ""])
    rows = rows[:-1]
    table = tabulate(rows,
                     headers=["**Name**","**TPS**", "**QPS**", "**RT(ms)**", "**Recall**", "**Score**"],
                     tablefmt="github")
    table = f"## VSAG Perf Report {date.today()}\n\n" + table
    print(table)
    send_message_to_dingtalk(access_token, table)
except Exception as e:
    print("Error: ", str(e))


import hmac
import hashlib
import base64
import time
import requests
import json
import os
import sys

access_token = os.environ['PERF_DINGDING_ACCESS_TOKEN']
secret = os.environ['PERF_DINGDING_SERCRET']

def generate_signature(timestamp, secret):
    secret_enc = secret.encode('utf-8')
    string_to_sign = f'{timestamp}\n{secret}'
    string_to_sign_enc = string_to_sign.encode('utf-8')
    hmac_code = hmac.new(secret_enc, string_to_sign_enc, hashlib.sha256).digest()
    sign = base64.b64encode(hmac_code)
    return sign.decode('utf-8')

def send_message_to_dingtalk(content):
    timestamp = str(round(time.time() * 1000))
    signature = generate_signature(timestamp, secret)
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        "msgtype": "text",
        "text": {
            "content": content
        },
    }
    url = "https://oapi.dingtalk.com/robot/send?"
    url = url + f"&access_token={access_token}" + f"&timestamp={timestamp}" + f"&sign={signature}"
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(response.text)


if len(sys.argv) < 2:
    print("Usage: python script.py <filename>")
    sys.exit(1)

filename = sys.argv[1]

try:
    with open(filename, 'r') as file:
        file_contents = file.read()
        file_contents = file_contents + "\nfrom vsag"
        if len(file_contents) > 1000:
            print(f"report is too long({len(file_contents)})")
        send_message_to_dingtalk(file_contents)
except FileNotFoundError:
    print(f"File '{filename}' not found.")

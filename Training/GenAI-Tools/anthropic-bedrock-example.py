import anthropic
import json
import urllib.request
import os
import getpass

BASE = "https://go.apis.huit.harvard.edu/ais-bedrock-llm/v2"
MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

huit_bedrock_api_key = getpass.getpass('Enter your HUIT API key for AWS Bedrock: ')
os.environ['HARVARD_API_PORTAL_KEY'] = huit_bedrock_api_key

def ask_ai(prompt):

    payload = {
        "messages": [
            {"role": "user", "content": [{"text": prompt}]}
        ]
    }

    req = urllib.request.Request(
        url=f"{BASE}/model/{MODEL}/converse",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "x-api-key": os.environ["HARVARD_API_PORTAL_KEY"],
            "User-Agent": "" # HTTP 403 occurs if no User-Agent header
        },
        method="POST",
    )

    with urllib.request.urlopen(req) as response:
        result = json.load(response)

    return result["output"]["message"]["content"][0]["text"]

print(ask_ai("Write a Python function that adds two numbers"))

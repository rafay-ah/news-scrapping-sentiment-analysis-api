api_key = "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"

import requests

API_URL = "https://api-inference.huggingface.co/models/hassan4830/xlm-roberta-base-finetuned-urdu"
headers = {"Authorization": f"Bearer {api_key}"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


output = query({
    "inputs": "ملک میں گزشتہ ایک ہفتے کے دوران مرکزی بینک کے زرمبادلہ کے ذخائر میں 30 کروڑ سے زائد کی کمی ہوئی ہے۔",
})

print(output)
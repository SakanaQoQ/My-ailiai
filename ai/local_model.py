# ai/local_model.py
import requests
from config import OLLAMA_API_URL, OLLAMA_MODEL

def ollama_chat(prompt):
    url = f"{OLLAMA_API_URL}/api/generate"
    data = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    resp = requests.post(url, json=data)
    resp.raise_for_status()
    return resp.json()['response']

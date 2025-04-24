import openai
from config import OPENAI_API_KEY, OPENAI_PROXY_URL

client = openai.OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_PROXY_URL if OPENAI_PROXY_URL and OPENAI_PROXY_URL.strip() else None
)

def openai_chat(prompt):
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content
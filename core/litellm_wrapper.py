import os
import requests
from dotenv import load_dotenv

load_dotenv()



class LiteLLMChat:
    def __init__(self, model=None):
        base_url = os.getenv("LITELLM_BASE_URL", "https://litellm.foundry.ubisoft.org")
        # Remove /openai suffix if it exists
        if base_url.endswith("/openai"):
            base_url = base_url[:-7]  # Remove "/openai"
        self.base = base_url
        self.model = model or os.getenv("LITELLM_MODEL", "gpt-4o-mini")
        self.key = os.getenv("LITELLM_API_KEY")

    def invoke(self, prompt: str) -> str:
        url = f"{self.base}/engines/{self.model}/chat/completions"
        headers = {
            "x-litellm-api-key": self.key,
            "accept": "application/json",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        }

        print(f"  → Making API call to: {url}")
        print(f"  → API Key present: {'Yes' if self.key else 'No'}")
        
        r = requests.post(url, headers=headers, json=body, timeout=60)
        print(f"  → Response status: {r.status_code}")
        
        if r.ok:
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            print(f"  → API response content: {repr(content)}")
            return content

        print(f"❌ {url}")
        print("Status:", r.status_code)
        print("Body:", r.text[:500])
        r.raise_for_status()


class LiteLLMEmbeddings:
    def __init__(self, model=None):
        base_url = os.getenv("LITELLM_BASE_URL", "https://litellm.foundry.ubisoft.org")
        # Remove /openai suffix if it exists
        if base_url.endswith("/openai"):
            base_url = base_url[:-7]  # Remove "/openai"
        self.base = base_url
        self.model = model or os.getenv("LITELLM_EMBED_MODEL", "text-embedding-3-large")
        self.key = os.getenv("LITELLM_API_KEY")

    def embed_query(self, text: str):
        url = f"{self.base}/engines/{self.model}/embeddings"
        headers = {
            "x-litellm-api-key": self.key,
            "accept": "application/json",
            "Content-Type": "application/json",
        }
        body = {"input": text}

        r = requests.post(url, headers=headers, json=body, timeout=60)
        if r.ok:
            data = r.json()
            return data["data"][0]["embedding"]

        print(f"❌ {url}")
        print("Status:", r.status_code)
        print("Body:", r.text[:500])
        r.raise_for_status()

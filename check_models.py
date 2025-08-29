# check_models.py
import os, requests
from dotenv import load_dotenv
load_dotenv()

base = "https://litellm.foundry.ubisoft.org"
key = os.getenv("LITELLM_API_KEY")

url = f"{base}/models?return_wildcard_routes=false&include_model_access_groups=false&only_model_access_groups=false"
resp = requests.get(url, headers={"x-litellm-api-key": key, "accept": "application/json"})

print("Status:", resp.status_code)
print(resp.text[:50000])  # dump first 5000 chars so we see the model list

#!/usr/bin/env python3
"""
Get all models from LiteLLM API.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

def get_models():
    """Get all models from LiteLLM"""
    base_url = "https://litellm.foundry.ubisoft.org"
    api_key = os.getenv("LITELLM_API_KEY")
    
    url = f"{base_url}/v1/models"
    headers = {
        "x-litellm-api-key": api_key,
        "accept": "application/json",
        "Content-Type": "application/json",
    }
    
    params = {
        "include_metadata": "true",
        "fallback_type": "general"
    }
    
    response = requests.get(url, headers=headers, params=params, timeout=30)
    
    if response.status_code == 200:
        data = response.json()
        
        if isinstance(data, dict) and 'data' in data:
            models = data['data']
        elif isinstance(data, list):
            models = data
        else:
            models = [data]
        
        print(f"Available Models ({len(models)}):")
        print("-" * 30)
        
        for i, model in enumerate(models, 1):
            if isinstance(model, dict):
                model_id = model.get('id', 'Unknown')
                print(f"{i:2d}. {model_id}")
            else:
                print(f"{i:2d}. {model}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    get_models()

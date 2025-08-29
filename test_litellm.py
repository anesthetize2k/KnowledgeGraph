#!/usr/bin/env python3

from litellm_wrapper import LiteLLMChat

def test_litellm():
    print("Testing LiteLLM wrapper...")
    
    # Test without API key
    llm = LiteLLMChat()
    print(f"API Key present: {'Yes' if llm.key else 'No'}")
    print(f"Base URL: {llm.base}")
    print(f"Model: {llm.model}")
    
    try:
        print("Attempting to call LLM...")
        response = llm.invoke("Hello, world!")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
        print(f"Error type: {type(e)}")
        print(f"Error repr: {repr(e)}")

if __name__ == "__main__":
    test_litellm()

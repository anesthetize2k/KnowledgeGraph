# smoke_test.py
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ensure .env is loaded
load_dotenv()

chat_model = os.getenv("LLM_MODEL_CHAT", "gpt-4o")
embed_model = os.getenv("LLM_MODEL_EMBED", "text-embedding-3-small")

print("🔎 Testing ChatOpenAI...")
llm = ChatOpenAI(model=chat_model, temperature=0)
print("Chat response:", llm.invoke("Say 'LiteLLM OK'").content)

print("\n🔎 Testing OpenAIEmbeddings...")
vec = OpenAIEmbeddings(model=embed_model).embed_query("embedding via LiteLLM")
print("Embedding length:", len(vec))

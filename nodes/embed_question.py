# nodes/embed_question.py
from query_state import QueryState
from langchain_openai import OpenAIEmbeddings

embedder = OpenAIEmbeddings(model="text-embedding-3-small")


def embed_question(state: QueryState) -> QueryState:
    embedding = embedder.embed_query(state["question"])
    state["embedding"] = embedding
    return state

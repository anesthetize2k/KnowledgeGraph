# nodes/generate_answer.py
from query_state import QueryState
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)


def generate_answer(state: QueryState) -> QueryState:
    context = state["context"]
    question = state["question"]

    prompt = f"""Answer the following question using only the information in the context.

Question: {question}

Context:
{context}

Answer:"""

    response = llm.invoke(prompt)
    state["final_answer"] = response.content
    return state

# query_state.py
from typing import TypedDict, List, Tuple


class QueryState(TypedDict, total=False):
    question: str
    embedding: List[float]
    chunk_ids: List[str]
    chunks: List[str]
    mentions: List[Tuple[str, str]]
    expansion_triples: List[str]
    final_answer: str
    context: str

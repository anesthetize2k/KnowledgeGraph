# ingestion_state.py
from typing import TypedDict, List, Any
from langchain_core.documents import Document


class IngestionState(TypedDict):
    pdf_path: str
    doc_id: str
    chunks: List[Document]
    embeddings: List[List[float]]
    triplets: List[tuple]
    new_entities: List[str]
    new_relations: List[str]

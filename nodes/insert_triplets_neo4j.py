# nodes/insert_triplets_neo4j.py
from ingestion_state import IngestionState
from triplet_extractor import TripletExtractor

extractor = TripletExtractor()


def insert_triplets_neo4j(state: IngestionState) -> IngestionState:
    chunks = state["chunks"]
    triplets = state["triplets"]

    for i, chunk in enumerate(chunks):
        chunk_id = chunk.metadata.get("chunk_id", f"unknown_chunk_{i}")
        extractor.insert_into_neo4j(triplets, chunk_id)

    return state

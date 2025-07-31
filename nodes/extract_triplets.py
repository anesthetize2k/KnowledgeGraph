# nodes/extract_triplets.py
from ingestion_state import IngestionState
from triplet_extractor import TripletExtractor

extractor = TripletExtractor()


def extract_triplets(state: IngestionState) -> IngestionState:
    triplets = []
    new_entities = set()
    new_relations = set()

    for chunk in state["chunks"]:
        response = extractor.chain.invoke({"text": chunk.page_content})
        raw = response.content
        parsed, ents, rels = extractor.parse_triplets(raw)
        triplets.extend(parsed)
        new_entities |= ents
        new_relations |= rels

    state["triplets"] = triplets
    state["new_entities"] = list(new_entities)
    state["new_relations"] = list(new_relations)
    return state

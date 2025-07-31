# nodes/update_ontology.py
from ingestion_state import IngestionState
from triplet_extractor import TripletExtractor

extractor = TripletExtractor()  # re-use same class


def update_ontology(state: IngestionState) -> IngestionState:
    new_ents = state.get("new_entities", [])
    new_rels = state.get("new_relations", [])
    extractor.update_ontology(new_ents, new_rels)
    return state

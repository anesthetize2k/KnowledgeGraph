# ingestion_graph.py

from langgraph.graph import StateGraph
from ingestion_state import IngestionState

# 5 pipeline steps (all your LangGraph nodes)
from nodes.load_and_embed_chunks import load_and_embed_chunks
from nodes.store_chunks_in_neo4j import store_chunks_in_neo4j
from nodes.extract_triplets import extract_triplets
from nodes.update_ontology import update_ontology
from nodes.insert_triplets_neo4j import insert_triplets_neo4j

# Build the LangGraph
graph = StateGraph(IngestionState)

# Add nodes
graph.add_node("load_and_embed_chunks", load_and_embed_chunks)
graph.add_node("store_chunks_in_neo4j", store_chunks_in_neo4j)
graph.add_node("extract_triplets", extract_triplets)
graph.add_node("update_ontology", update_ontology)
graph.add_node("insert_triplets_neo4j", insert_triplets_neo4j)

# Set execution order
graph.set_entry_point("load_and_embed_chunks")
graph.add_edge("load_and_embed_chunks", "store_chunks_in_neo4j")
graph.add_edge("store_chunks_in_neo4j", "extract_triplets")
graph.add_edge("extract_triplets", "update_ontology")
graph.add_edge("update_ontology", "insert_triplets_neo4j")

# Compile the app
app = graph.compile()

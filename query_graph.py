# query_graph.py
from langgraph.graph import StateGraph
from query_state import QueryState
from nodes.embed_question import embed_question
from nodes.vector_search_chunks import vector_search_chunks
from nodes.collect_mentions import collect_mentions
from nodes.expand_neighbors import expand_neighbors
from nodes.compose_context import compose_context
from nodes.generate_answer import generate_answer

graph = StateGraph(QueryState)

graph.add_node("embed_question", embed_question)
graph.add_node("vector_search_chunks", vector_search_chunks)
graph.add_node("collect_mentions", collect_mentions)
graph.add_node("expand_neighbors", expand_neighbors)
graph.add_node("compose_context", compose_context)
graph.add_node("generate_answer", generate_answer)

graph.set_entry_point("embed_question")
graph.add_edge("embed_question", "vector_search_chunks")
graph.add_edge("vector_search_chunks", "collect_mentions")
graph.add_edge("collect_mentions", "expand_neighbors")
graph.add_edge("expand_neighbors", "compose_context")
graph.add_edge("compose_context", "generate_answer")

app = graph.compile()

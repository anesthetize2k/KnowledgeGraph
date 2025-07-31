# nodes/vector_search_chunks.py
from query_state import QueryState
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()


def vector_search_chunks(state: QueryState) -> QueryState:
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )

    embedding = state["embedding"]
    chunks = []
    chunk_ids = []

    with driver.session() as session:
        result = session.run(
            """
            CALL db.index.vector.queryNodes('vector', 10, $embedding)
            YIELD node, score
            RETURN node.chunk_id AS cid, node.text AS chunk
            ORDER BY score DESC
            """,
            embedding=embedding,
        )
        for row in result:
            chunks.append(row["chunk"])
            chunk_ids.append(row["cid"])

    state["chunks"] = chunks
    state["chunk_ids"] = chunk_ids
    return state

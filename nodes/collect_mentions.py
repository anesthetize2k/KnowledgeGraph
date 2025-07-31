# nodes/collect_mentions.py
from query_state import QueryState
from neo4j import GraphDatabase
import os

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
)


def collect_mentions(state: QueryState) -> QueryState:
    chunk_ids = state["chunk_ids"]
    mentions = []

    with driver.session() as session:
        for cid in chunk_ids:
            result = session.run(
                """
                MATCH (c:Chunk {chunk_id: $cid})-[:MENTIONS]->(e)
                WHERE NOT 'Chunk' IN labels(e)
                RETURN e.name AS name, labels(e) AS labels
                """,
                cid=cid,
            )
            for row in result:
                if row["name"]:
                    label = row["labels"][0] if row["labels"] else ""
                    mentions.append((row["name"], label))

    state["mentions"] = mentions
    return state

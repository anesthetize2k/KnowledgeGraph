# nodes/expand_neighbors.py
from query_state import QueryState
from neo4j import GraphDatabase
import os

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
)


def expand_neighbors(state: QueryState) -> QueryState:
    mentions = state.get("mentions", [])
    expansion_triples = []

    with driver.session() as session:
        for name, typ in mentions:
            result = session.run(
                f"""
                MATCH (e:{typ} {{name: $name}})-[rel]->(n)
                WHERE NOT 'Chunk' IN labels(n)
                RETURN e.name AS entity_name, labels(e) AS entity_type,
                       type(rel) AS rel_type, n.name AS neighbor_name, labels(n) AS neighbor_type
                """,
                name=name,
            )
            for row in result:
                if row["rel_type"] and row["neighbor_name"]:
                    expansion_triples.append(
                        f"{row['entity_name']} ({row['entity_type'][0]}) --[{row['rel_type']}]--> {row['neighbor_name']} ({row['neighbor_type'][0]})"
                    )

    state["expansion_triples"] = list(dict.fromkeys(expansion_triples))  # dedupe
    return state

# nodes/store_chunks_in_neo4j.py
import os
from neo4j import GraphDatabase
from ingestion_state import IngestionState
from dotenv import load_dotenv

load_dotenv()


def store_chunks_in_neo4j(state: IngestionState) -> IngestionState:
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )

    doc_id = state["doc_id"]
    chunks = state["chunks"]
    embeddings = state["embeddings"]

    with driver.session() as session:
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            text = chunk.page_content
            chunk_id = chunk.metadata["chunk_id"]
            session.run(
                """
                MERGE (d:Document {source_id: $doc_id})
                MERGE (c:Chunk {chunk_id: $chunk_id})
                    ON CREATE SET c.text = $text, c.embedding = $embedding
                MERGE (d)-[:HAS_CHUNK]->(c)
                """,
                doc_id=doc_id,
                chunk_id=chunk_id,
                text=text,
                embedding=embedding,
            )

    return state

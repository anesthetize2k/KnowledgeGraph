import os
import logging
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO)


class DocumentIngestor:
    """
    Loads a PDF, splits into overlapping chunks, computes embeddings, and writes nodes to Neo4j.
    """

    def __init__(
        self, pdf_path, doc_id, neo4j_uri=None, neo4j_user=None, neo4j_pwd=None
    ):
        self.pdf_path = pdf_path
        self.doc_id = doc_id
        self.neo4j_uri = neo4j_uri or os.environ.get("NEO4J_URI")
        self.neo4j_user = os.environ.get("NEO4J_USERNAME")
        self.neo4j_pwd = os.environ.get("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(
            self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_pwd)
        )
        self.embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    def load_chunks(self, chunk_size=500, chunk_overlap=100):
        loader = PyMuPDFLoader(str(self.pdf_path))
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(docs)
        return [chunk.page_content for chunk in chunks]

    def ingest(self):
        chunks = self.load_chunks()
        logging.info(f"Loaded {len(chunks)} chunks from {self.pdf_path}")
        with self.driver.session() as session:
            for i, text in enumerate(chunks):
                chunk_id = f"{self.doc_id}:{i}"
                emb = self.embedder.embed_query(text)
                session.run(
                    """
                    MERGE (d:Document {source_id: $doc_id})
                    MERGE (c:Chunk {chunk_id: $chunk_id})
                        ON CREATE SET c.text = $text, c.embedding = $embedding
                    MERGE (d)-[:HAS_CHUNK]->(c)
                    """,
                    doc_id=self.doc_id,
                    chunk_id=chunk_id,
                    text=text,
                    embedding=emb,
                )
                if i % 10 == 0 or i == len(chunks) - 1:
                    logging.info(f"Ingested chunk {i+1}/{len(chunks)}")
        logging.info(f"Successfully ingested all {len(chunks)} chunks into Neo4j.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest a PDF into Neo4j as Document and Chunk nodes with embeddings."
    )
    parser.add_argument("pdf_path", type=str, help="Path to PDF file")
    parser.add_argument(
        "--doc_id", type=str, required=True, help="Unique document ID for this PDF"
    )
    args = parser.parse_args()

    ingestor = DocumentIngestor(
        pdf_path=args.pdf_path,
        doc_id=args.doc_id,
        # Optionally: neo4j_uri=..., neo4j_user=..., neo4j_pwd=...
    )
    ingestor.ingest()

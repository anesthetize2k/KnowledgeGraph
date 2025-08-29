import os
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from neo4j import GraphDatabase
from litellm_wrapper import LiteLLMEmbeddings
from langchain.schema import Document

load_dotenv()


class DocumentIngestor:
    def __init__(self):
        self.embedder = LiteLLMEmbeddings(model=os.getenv("LITELLM_EMBED_MODEL", "text-embedding-3-large"))

        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, length_function=len, is_separator_regex=False
        )

    def _get_text_preview(self, file_path: str, max_chars: int = 6000) -> str:
        docs = self._load_document(file_path)
        text = " ".join([d.page_content for d in docs])
        return text[:max_chars]

    def _hash_file(self, file_path: str) -> str:
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _load_document(self, file_path: str):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return PyMuPDFLoader(file_path).load()
        elif ext in [".md", ".txt"]:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                return [Document(page_content=text, metadata={"source": file_path})]
            except Exception as e:
                raise ValueError(f"Failed to load markdown/txt file {file_path}: {e}")
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def process(self, file_path: str):
        docs = self._load_document(file_path)
        chunks = self.text_splitter.split_documents(docs)

        file_hash = self._hash_file(file_path)
        file_name = os.path.basename(file_path)

        with self.driver.session() as session:
            session.run(
                """
                MERGE (d:Document {source_id: $source_id})
                SET d.name = $name
                """,
                source_id=file_hash,
                name=file_name,
            )

            for i, chunk in enumerate(chunks):
                chunk_id = f"{file_hash}_{i}"
                chunk_text = chunk.page_content

                try:
                    embedding = self.embedder.embed_query(chunk_text)
                except Exception as e:
                    print(f"❌ Failed to embed chunk {i}: {e}")
                    continue

                session.run(
                    """
                    MERGE (c:Chunk {chunk_id: $chunk_id})
                    SET c.text = $text,
                        c.embedding = $embedding
                    MERGE (d:Document {source_id: $source_id})
                    MERGE (d)-[:HAS_CHUNK]->(c)
                    """,
                    chunk_id=chunk_id,
                    text=chunk_text,
                    embedding=embedding,
                    source_id=file_hash,
                )

                chunk.metadata["chunk_id"] = chunk_id

        return chunks

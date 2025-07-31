# nodes/load_and_embed_chunks.py
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from ingestion_state import IngestionState

embedder = OpenAIEmbeddings(model="text-embedding-3-small")


def load_and_embed_chunks(state: IngestionState) -> IngestionState:
    loader = PyMuPDFLoader(state["pdf_path"])
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"{state['doc_id']}:{i}"
        chunk.metadata["chunk_id"] = chunk_id
        emb = embedder.embed_query(chunk.page_content)
        embeddings.append(emb)

    state["chunks"] = chunks
    state["embeddings"] = embeddings
    return state

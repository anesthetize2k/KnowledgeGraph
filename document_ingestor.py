from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path


class DocumentIngestor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def load_chunks(self):
        loader = PyMuPDFLoader(self.pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        return splitter.split_documents(docs)

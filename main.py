from pathlib import Path
import json

from document_ingestor import DocumentIngestor
from triplet_extractor import TripletExtractor
from semantic_agent import SemanticAgent

PROCESSED_LOG = Path("processed.json")


def load_processed():
    if PROCESSED_LOG.exists():
        return set(json.loads(PROCESSED_LOG.read_text()))
    return set()


def save_processed(files):
    with open(PROCESSED_LOG, "w") as f:
        json.dump(sorted(list(files)), f)


def process_new_files():
    print("\n🔍 Checking for new PDFs in /data")
    data_dir = Path("data")
    all_pdfs = set(p.name for p in data_dir.glob("*.pdf"))
    processed = load_processed()
    new_files = all_pdfs - processed

    if not new_files:
        print("✅ No new files to process.")
        return

    extractor = TripletExtractor()
    for pdf_name in new_files:
        pdf_path = data_dir / pdf_name
        print(f"\n📄 Processing: {pdf_name}")
        chunks = DocumentIngestor(str(pdf_path)).load_chunks()
        extractor.process_chunks(chunks)
        processed.add(pdf_name)

    save_processed(processed)
    print("\n✅ All new files processed.")


def query_graph():
    agent = SemanticAgent()
    while True:
        question = input("\n💬 Enter your question (or 'exit' to quit): ")
        if question.lower() in {"exit", "quit"}:
            break
        agent.ask(question)


if __name__ == "__main__":
    print("\n🧠 KnowledgeGraph CLI")
    print("1. Process new PDF files")
    print("2. Query the knowledge graph")
    choice = input("\nEnter choice (1 or 2): ")

    if choice == "1":
        process_new_files()
    elif choice == "2":
        query_graph()
    else:
        print("❌ Invalid choice. Please run the script again.")

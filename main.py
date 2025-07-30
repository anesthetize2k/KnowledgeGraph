from pathlib import Path
import json

from document_ingestor import DocumentIngestor
from semantic_agent import SemanticAgent
from dotenv import load_dotenv

PROCESSED_LOG = Path("processed.json")

load_dotenv()


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

    for pdf_name in new_files:
        pdf_path = data_dir / pdf_name
        doc_id = pdf_path.stem
        print(f"\n📄 Processing: {pdf_name}")

        # Ingest the document, creating nodes with embeddings
        ingestor = DocumentIngestor(str(pdf_path), doc_id)
        ingestor.ingest()

        # Optionally process for triplets/entities (if implemented)
        # You may want to implement extractor.process_chunks(chunks) within ingestor.ingest()
        # For now, we'll leave it as a placeholder

        processed.add(pdf_name)
        save_processed(processed)  # Save after each file, safer for crash recovery

    print("\n✅ All new files processed.")


def query_graph():
    agent = SemanticAgent()
    while True:
        question = input("\n💬 Enter your question (or 'exit' to quit): ").strip()
        if question.lower() in {"exit", "quit"}:
            break

        answer = agent.run_query(question)
        print(f"\n🤖 {answer}")


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

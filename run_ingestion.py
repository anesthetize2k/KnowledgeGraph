# run_ingestion.py
from pathlib import Path
import json
from ingestion_graph import app

PROCESSED_LOG = Path("processed.json")


def load_processed():
    if PROCESSED_LOG.exists():
        return set(json.loads(PROCESSED_LOG.read_text()))
    return set()


def save_processed(files):
    with open(PROCESSED_LOG, "w") as f:
        json.dump(sorted(list(files)), f)


# Load processed files
processed = load_processed()

# Ingest only new PDFs
pdf_dir = Path("data")
new_files = [p for p in pdf_dir.glob("*.pdf") if p.name not in processed]

if not new_files:
    print("✅ No new PDFs to ingest.")
else:
    for pdf_path in new_files:
        doc_id = pdf_path.stem
        print(f"\n📄 Ingesting: {pdf_path.name}")
        state = {"pdf_path": str(pdf_path), "doc_id": doc_id}
        app.invoke(state)
        # Add to processed list
        processed.add(pdf_path.name)
        save_processed(processed)

    print("✅ All new files processed.")

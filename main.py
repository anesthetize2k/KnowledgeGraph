import os
import json
from dotenv import load_dotenv

from document_ingestor import DocumentIngestor
from triplet_extractor import TripletExtractor
from semantic_agent import SemanticAgent
from wiki_creator import WikiCreator

from entity_harvest import EntityHarvest

load_dotenv()

PROCESSED_FILE = "processed.json"


def load_processed_files():
    if os.path.exists(PROCESSED_FILE):
        with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_processed_files(processed):
    with open(PROCESSED_FILE, "w", encoding="utf-8") as f:
        json.dump(list(processed), f, indent=2)




def process_new_files():
    ingestor = DocumentIngestor()
    extractor = TripletExtractor()
    wiki = WikiCreator()
    harvester = EntityHarvest()

    processed = load_processed_files()
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    files = [f for f in os.listdir(data_dir)
             if f.lower().endswith((".pdf",".md",".txt")) and not f.startswith("Wikis")]
    new_files = [f for f in files if f not in processed]
    if not new_files:
        print("✅ No new files to process.")
        return

    held_files = []

    for f in new_files:
        path = os.path.join(data_dir, f)
        name_no_ext = os.path.splitext(f)[0]
        print(f"\n📄 Processing {f}...")

        try:
            # preview text
            try:
                doc_text = ingestor._get_text_preview(path, max_chars=6000)
            except Exception:
                with open(path, "r", encoding="utf-8", errors="ignore") as rf:
                    doc_text = rf.read()[:6000]

            # A) ENTITY WIKIS FIRST (always)
            ents = harvester.harvest(doc_text)
            if ents:
                print(f"👥 Entities detected: {', '.join([e['name'] for e in ents])}")
            else:
                print("👥 No clear entities detected.")

            # Optional: consult graph for canonical names (if you want)
            # (pseudo) For each entity name, you can query Neo4j for an existing node and reuse that 'name'

            for e in ents:
                try:
                    print(f"  → Processing entity: {e}")
                    wiki.upsert_entity_wiki(e, doc_text, f)
                except Exception as entity_error:
                    print(f"  ❌ Error processing entity {e}: {entity_error}")
                    continue

            # B) STUDY/TYPE WIKI CLASSIFICATION (gated)
            print("🔎 Running wiki classification with LLM...")
            classification = wiki.classify_document(doc_text)
            matches = classification.get("matches", []) or []
            new_type = classification.get("new_type")
            print(f"  → Matches: {json.dumps(matches, indent=2)}")
            if new_type:
                print(f"  → LLM also suggested NEW type: {new_type}")

            good_matches = [m for m in matches if float(m.get("confidence", 0)) >= 0.60]
            if not good_matches:
                print(f"⚠️  No confident study wiki match for {f}. Proposing NEW type + holding file.")
                wiki.propose_new_type(doc_text, f)
                held_files.append(f)
                # DO NOT build graph yet for held study files
                continue

            # If matched, write the study wiki(s)
            print(f"📝 Generating study wiki(s) for {f}...")
            written_wikis = wiki.create_or_update_wikis_from_matches(good_matches, name_no_ext, f, doc_text)

            # Graph ingest AFTER wiki(s)
            print("📥 Ingesting chunks + embeddings...")
            chunks = ingestor.process(path)
            print(f"  → {len(chunks)} chunks created and embedded.")

            # Link Document → entity wikis too (optional)
            if ents:
                from neo4j import GraphDatabase
                import hashlib, os as _os
                def _hash_file(fp):
                    import hashlib
                    hasher = hashlib.md5()
                    with open(fp, "rb") as h:
                        while block := h.read(8192):
                            hasher.update(block)
                    return hasher.hexdigest()
                file_hash = _hash_file(path)
                driver = GraphDatabase.driver(_os.getenv("NEO4J_URI"),
                                              auth=(_os.getenv("NEO4J_USERNAME"), _os.getenv("NEO4J_PASSWORD")))
                with driver.session() as session:
                    for e in ents:
                        wiki_type = wiki._map_entity_to_wiki_type(e["type"])
                        wpath = wiki._wiki_path(wiki_type, e["name"])
                        title = f"{wiki_type}_{e['name'].replace(' ','_')}"
                        session.run(
                            """
                            MERGE (w:WikiPage {path: $path})
                            SET w.title = $title
                            WITH w
                            MATCH (d:Document {source_id: $source_id})
                            MERGE (d)-[:MENTIONS_ENTITY_WIKI]->(w)
                            """,
                            path=wpath, title=title, source_id=file_hash
                        )
                print("  → Linked Document to entity wiki pages.")

            print("🧩 Extracting triplets & updating ontology...")
            extractor.process_chunks(chunks)
            print("  → Triplets extracted & ontology updated.")

        except Exception as e:
            print(f"❌ Error processing {f}: {e}")
            continue

        processed.add(f)

    save_processed_files(processed)

    if held_files:
        print("\n⏸️  HELD FILES (awaiting schema approval):")
        for hf in held_files: print("   -", hf)
        print("👉 Edit schema_update_suggestions.json (set approved:'ok'), apply schemas, then re-run processing.")

    print("\n✅ All eligible files processed.")


def query_graph():
    agent = SemanticAgent()
    while True:
        q = input("\nAsk a question (or 'exit' to quit): ")
        if q.strip().lower() == "exit":
            break
        try:
            answer = agent.run_query(q)
            print("\n🧠 Answer:", answer)
        except Exception as e:
            print(f"❌ Query failed: {e}")

def apply_approved_schemas():
    wiki = WikiCreator()
    added = wiki.apply_approved_suggestions()
    if added:
        print("✅ Added new wiki types to schema:", ", ".join(added))
    else:
        print("ℹ️ No approved suggestions found.")


def main():
    while True:
        print("1. Process new files")
        print("2. Query the knowledge graph")
        print("3. Apply approved schema changes")
        print("4. Exit")

        choice = input("Choose an option: ").strip()
        if choice == "1":
            process_new_files()
        elif choice == "2":
            query_graph()
        elif choice == "3":
            apply_approved_schemas()
        elif choice == "4":
            break
        else:
            print("Invalid choice, try again.")


if __name__ == "__main__":
    main()

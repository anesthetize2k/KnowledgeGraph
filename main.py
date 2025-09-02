import os
import json
from dotenv import load_dotenv

from document_ingestor import DocumentIngestor
from triplet_extractor import TripletExtractor
from semantic_agent import SemanticAgent
from wiki_creator import WikiCreator, CREATE_EMPTY_STUBS

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


def _get_text_preview(text: str, max_chars: int = 200) -> str:
    """Extract a meaningful preview of text content."""
    if not text or not text.strip():
        return ""
    
    # Remove common binary/encoding artifacts
    cleaned = text.strip()
    if cleaned.startswith("PK\x03\x04") or cleaned.startswith("[Document"):
        return ""
    
    # Take first meaningful chunk
    preview = cleaned[:max_chars]
    if len(cleaned) > max_chars:
        preview += "..."
    
    return preview


def detect_research_study(doc_text: str, file_name: str, llm) -> dict:
    """
    Use LLM to detect if a document is a research study/analysis/report/post mortem.
    Returns a dict with classification info.
    """
    prompt = f"""
You are a document classifier specializing in identifying research studies, analyses, reports, and post mortems.

Analyze the following document and determine its type.

Document name: {file_name}

Document content (first 3000 characters):
\"\"\"{doc_text[:3000]}\"\"\"

Answer ONLY with JSON in this exact format:
{{
  "is_research_document": true/false,
  "confidence": 0.0-1.0,
  "document_type": "post_mortem|analysis|research_study|report|other",
  "subtype": "brief description (e.g., 'Game Performance Analysis', 'Player Perception Study', 'Market Research Report')",
  "reasoning": "brief explanation of the classification"
}}

 Document types:
 - "post_mortem": Game performance analysis, post-release evaluation, sales analysis, player engagement review, post-mortem reports
 - "analysis": Data analysis, trend analysis, market analysis, competitive analysis, performance analysis, buyer/non-buyer analysis, customer analysis, player analysis, audience analysis, behavioral analysis
 - "research_study": Perception studies, brand awareness research, user feedback studies, survey results, focus group studies, qualitative research
 - "report": General reports, summaries, findings reports, executive summaries, business reports
 - "other": Marketing materials, press releases, product descriptions, manuals, articles, general documents

 IMPORTANT: If the document content is corrupted or unreadable, classify based on the filename and context clues. File names often contain valuable information about the document type.

 Examples:
 - Post-mortem analyses of game performance → "post_mortem"
 - Player perception studies → "research_study" 
 - Market analysis reports → "analysis"
 - Sales performance reviews → "post_mortem"
 - Competitive analysis → "analysis"
 - Brand awareness studies → "research_study"
 - Buyer/non-buyer analysis → "analysis"
 - Customer analysis → "analysis"
 - Player behavior analysis → "analysis"
 - Audience analysis → "analysis"
 - General reports → "report"
"""

    try:
        response = llm.invoke(prompt)
        # Clean up response
        cleaned = response.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        data = json.loads(cleaned)
        return {
            "is_research_document": bool(data.get("is_research_document", False)),
            "confidence": float(data.get("confidence", 0.0)),
            "document_type": data.get("document_type", "other"),
            "subtype": data.get("subtype", ""),
            "reasoning": data.get("reasoning", "")
        }
    except Exception as e:
        print(f"      ⚠️ Research study detection failed: {e}")
        return {
            "is_research_document": False,
            "confidence": 0.0,
            "document_type": "other",
            "subtype": "",
            "reasoning": "Detection failed"
        }


def process_new_files():
    """Process new files with atomic operations - one file at a time with checkpointing."""
    print("📁 Scanning for new files...")
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory '{data_dir}' not found!")
        return

    # Get all files in data directory
    all_files = [f for f in os.listdir(data_dir) 
                 if f.endswith(('.md', '.txt', '.pdf', '.docx'))]
    
    if not all_files:
        print("ℹ️ No files found in data directory.")
        return

    # Load already processed files
    processed = load_processed_files()
    new_files = [f for f in all_files if f not in processed]
    
    if not new_files:
        print("ℹ️ All files already processed.")
        return

    print(f"📊 Found {len(new_files)} new files to process:")
    for f in new_files:
        print(f"   📄 {f}")

    # Initialize components
    print("\n🔧 Initializing processing components...")
    ingestor = DocumentIngestor()
    extractor = TripletExtractor()
    wiki = WikiCreator()
    harvester = EntityHarvest()
    
    # Initialize LLM for research study detection
    from litellm_wrapper import LiteLLMChat
    llm = LiteLLMChat()

    print(f"\n🚀 Starting atomic processing of {len(new_files)} files...")
    print("=" * 60)

    # Process files one at a time with checkpointing
    for file_idx, f in enumerate(new_files, 1):
        print(f"\n📄 Processing file {file_idx}/{len(new_files)}: {f}")
        print("-" * 50)
        
        try:
            path = os.path.join(data_dir, f)
            
            # Step 1: Generate file hash for unique identification
            import hashlib
            with open(path, 'rb') as file:
                file_hash = hashlib.md5(file.read()).hexdigest()
            print(f"🔐 File hash: {file_hash}")

            # Step 2: Load and preview document
            print(f"📖 Loading document...")
            with open(path, 'r', encoding='utf-8', errors='ignore') as file:
                doc_text = file.read()
            
            # Check if document has meaningful content
            meaningful_content = _get_text_preview(doc_text)
            if not meaningful_content or meaningful_content.strip() == "":
                print(f"   ⚠️ Document appears to be empty or corrupted")
                # Still process it but mark as corrupted
                meaningful_content = "corrupted or unreadable document"
            
            print(f"   📊 Document size: {len(doc_text)} characters")
            print(f"   📝 Preview: {meaningful_content[:100]}...")

            # Step 3: Create document node in graph (single source of truth)
            print(f"📋 Creating document node in graph...")
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))
            )
            
            with driver.session() as session:
                # Create document node with all metadata
                session.run(
                    """
                    MERGE (d:Document {source_id: $source_id})
                    SET d.name = $name,
                        d.path = $path,
                        d.processed_at = datetime(),
                        d.file_size = $file_size,
                        d.content_preview = $content_preview
                    """,
                    source_id=file_hash,
                    name=f,
                    path=path,
                    file_size=len(doc_text),
                    content_preview=meaningful_content[:200]
                )
            driver.close()
            print(f"   ✅ Document node created/updated")

            # Step 4: Entity extraction and wiki creation
            print(f"👥 Extracting entities for wiki creation...")
            ents = harvester.harvest(doc_text)  # returns [{"name":..., "type":...}, ...]
            if ents:
                print(f"   ✅ Entities detected: {', '.join([e['name'] for e in ents])}")
            else:
                print("   ℹ️ No clear entities detected.")

            for e_idx, e in enumerate(ents, 1):
                try:
                    print(f"   🔍 Assessing entity {e_idx}/{len(ents)}: {e['name']} ({e['type']})")
                    decision = wiki.assess_entity_worthiness(e, doc_text, f)
                    msg = f"      merit={decision['merit']} | facts={decision['facts_count']} | ubisoft={decision['ubisoft_linked']} | reason={decision['reason']}"
                    print(f"      {msg}")

                    if decision["merit"]:
                        print(f"      ✅ Creating/updating entity wiki...")
                        wiki.upsert_entity_wiki(e, doc_text, f, file_hash)
                    else:
                        if CREATE_EMPTY_STUBS:
                            print(f"      📝 Creating empty stub (config enabled)...")
                            wiki.upsert_entity_wiki(e, "", f, file_hash)
                        else:
                            print(f"      ⏸️ Skipped: insufficient merit")
                except Exception as entity_error:
                    print(f"      ❌ Error processing entity {e}: {entity_error}")
                    continue

            # Step 5: Research Study Detection (LLM-based)
            print(f"🔬 Detecting research study type...")
            research_detection = detect_research_study(doc_text, f, llm)
            print(f"   📊 Detection result: {json.dumps(research_detection, indent=2)}")
            
            # Check if this document name matches any extracted entities (indicating it's a source file)
            name_no_ext = os.path.splitext(f)[0]
            document_is_source_file = any(
                name_no_ext.lower() in entity['name'].lower() or 
                entity['name'].lower() in name_no_ext.lower()
                for entity in ents
            )
            
            if document_is_source_file:
                print(f"   ⏸️ Skipping study wiki creation - document appears to be a source file")
                print(f"   📝 Document name: '{name_no_ext}' matches extracted entities")
            elif research_detection["is_research_document"] and research_detection["confidence"] >= 0.70:
                # Auto-create appropriate wiki based on document type
                doc_type = research_detection["document_type"]
                if doc_type in ["post_mortem", "analysis"]:
                    wiki_type = "Post Mortem"
                elif doc_type == "research_study":
                    wiki_type = "Research Study"
                else:
                    wiki_type = "Research Study"  # fallback
                
                print(f"   ✅ Auto-creating {wiki_type} wiki...")
                auto_match = {
                    "wiki_type": wiki_type,
                    "subtype": research_detection["subtype"],
                    "confidence": research_detection["confidence"]
                }
                written_wikis = wiki.create_or_update_wikis_from_matches(
                    [auto_match], name_no_ext, f, doc_text, file_hash
                )
                print(f"   ✅ Generated {wiki_type} wiki")
            else:
                # Fall back to regular classification - but always create something
                print(f"   🔎 Running standard document classification...")
                classification = wiki.classify_document(doc_text)
                matches = classification.get("matches", []) or []
                print(f"   📊 Raw classification: {json.dumps(classification, indent=2)}")
                print(f"   📊 Matches: {json.dumps(matches, indent=2)}")

                good_matches = [m for m in matches if float(m.get("confidence", 0)) >= 0.60]
                print(f"   ✅ Good matches (confidence >= 0.60): {json.dumps(good_matches, indent=2)}")
                
                if not good_matches:
                    # Create a default "Other" wiki type for any document that doesn't match
                    print(f"   📝 No confident matches - creating 'Other' wiki type...")
                    default_match = {
                        "wiki_type": "Other",
                        "subtype": "General Document",
                        "confidence": 0.50
                    }
                    written_wikis = wiki.create_or_update_wikis_from_matches(
                        [default_match], name_no_ext, f, doc_text, file_hash
                    )
                    print(f"   ✅ Generated 'Other' wiki")
                else:
                    # Generate study wikis
                    print(f"📝 Generating study wiki(s)...")
                    written_wikis = wiki.create_or_update_wikis_from_matches(
                        good_matches, name_no_ext, f, doc_text, file_hash
                    )
                    print(f"   ✅ Generated {len(written_wikis)} wiki(s)")

            # Step 6: Knowledge graph creation
            print(f"🧠 Creating knowledge graph...")
            print(f"   📥 Ingesting document chunks...")
            chunks = ingestor.process(path)
            print(f"   ✅ Created {len(chunks)} chunks with embeddings")

            print(f"   🧩 Extracting knowledge triplets...")
            extractor.process_chunks(chunks, source_document_id=file_hash)
            print(f"   ✅ Knowledge graph updated")

            # ATOMIC CHECKPOINT: Save progress after successful file processing
            processed.add(f)
            save_processed_files(processed)
            print(f"💾 Checkpoint saved - {f} marked as processed")

            print(f"✅ File {f} processed successfully!")

        except Exception as e:
            print(f"❌ Error processing {f}: {e}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            print(f"   ⚠️ File {f} will be retried on next run")
            continue

    print(f"\n🎉 Processing complete! {len([f for f in new_files if f in processed])} files processed successfully.")
    newly_processed = len([f for f in new_files if f in processed])
    already_processed = len([f for f in new_files if f not in processed])
    print(f"📊 Summary: {newly_processed} new files processed, {already_processed} files already processed, {len(new_files)} total files scanned")


def query_graph():
    print("🧠 Starting knowledge graph query interface...")
    agent = SemanticAgent()
    while True:
        q = input("\nAsk a question (or 'exit' to quit): ")
        if q.strip().lower() == "exit":
            break
        try:
            print("🔍 Searching knowledge graph...")
            response = agent.run_query(q)
            print("\n🧠 Answer:")
            print(response)
        except Exception as e:
            print(f"❌ Query failed: {e}")

def apply_approved_schemas():
    print("🔧 Applying approved schema changes...")
    wiki = WikiCreator()
    added = wiki.apply_approved_suggestions()
    if added:
        print(f"✅ Added new wiki types to schema: {', '.join(added)}")
    else:
        print("ℹ️ No approved suggestions found.")


def manage_ontology():
    """Manage ontology suggestions and updates."""
    from ontology_expander import OntologyExpander
    
    print("🧬 Ontology Management Interface")
    print("=" * 40)
    
    expander = OntologyExpander()
    summary = expander.get_suggestion_summary()
    
    print(f"📊 Current status:")
    print(f"   • Pending suggestions: {summary['pending']}")
    print(f"   • Approved suggestions: {summary['approved']}")
    print(f"   • Rejected suggestions: {summary['rejected']}")
    
    if summary['pending'] == 0:
        print("\n✅ No pending suggestions to review.")
        return
    
    print(f"\n📋 Pending suggestions:")
    pending = expander.get_pending_suggestions()
    for suggestion in pending:
        print(f"\n   ID: {suggestion['id']}")
        print(f"   Type: {suggestion['type']}")
        print(f"   Original: {suggestion['original_type']}")
        print(f"   Canonical: {suggestion['canonical_name']}")
        print(f"   Description: {suggestion['description']}")
        print(f"   Confidence: {suggestion['confidence']:.2f}")
        print(f"   Reasoning: {suggestion['reasoning']}")
    
    while True:
        print(f"\nOptions:")
        print(f"1. Approve suggestion (enter ID)")
        print(f"2. Reject suggestion (enter ID)")
        print(f"3. Exit")
        
        choice = input("\nChoose an option: ").strip()
        
        if choice == "1":
            try:
                suggestion_id = int(input("Enter suggestion ID to approve: "))
                if expander.approve_suggestion(suggestion_id):
                    print(f"✅ Suggestion {suggestion_id} approved and added to ontology!")
                else:
                    print(f"❌ Suggestion {suggestion_id} not found.")
            except ValueError:
                print("❌ Invalid ID. Please enter a number.")
        
        elif choice == "2":
            try:
                suggestion_id = int(input("Enter suggestion ID to reject: "))
                if expander.reject_suggestion(suggestion_id):
                    print(f"✅ Suggestion {suggestion_id} rejected.")
                else:
                    print(f"❌ Suggestion {suggestion_id} not found.")
            except ValueError:
                print("❌ Invalid ID. Please enter a number.")
        
        elif choice == "3":
            break
        
        else:
            print("❌ Invalid choice, try again.")


def main():
    print("🎯 Knowledge Graph Processing System")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Process new files")
        print("2. Query the knowledge graph")
        print("3. Apply approved schema changes")
        print("4. Manage ontology suggestions")
        print("5. Exit")

        choice = input("\nChoose an option: ").strip()
        if choice == "1":
            process_new_files()
        elif choice == "2":
            query_graph()
        elif choice == "3":
            apply_approved_schemas()
        elif choice == "4":
            manage_ontology()
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice, try again.")


if __name__ == "__main__":
    main()

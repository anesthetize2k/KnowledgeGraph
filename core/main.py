import os
import json
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

from .document_ingestor import DocumentIngestor
from .triplet_extractor import TripletExtractor
from .semantic_agent import SemanticAgent
from .wiki_creator import WikiCreator, CREATE_EMPTY_STUBS

from .entity_harvest import EntityHarvest
from .db_utils import processed_docs_db

load_dotenv()


def recreate_wiki_index():
    """Recreate the wiki index.md file with all current files in the Wikis folder."""
    print("🔄 Recreating wiki index...")
    
    wikis_dir = Path("Wikis")
    if not wikis_dir.exists():
        print("❌ Wikis directory not found!")
        return
    
    # Get all markdown files in the Wikis directory (excluding index.md)
    md_files = [f for f in wikis_dir.glob("*.md") if f.name != "index.md"]
    
    if not md_files:
        print("❌ No markdown files found in Wikis directory!")
        return
    
    # Sort files alphabetically
    md_files.sort(key=lambda x: x.name)
    
    # Group files by category
    categories = {
        "Entity Wikis": [],
        "Research Wikis": [],
        "Source Documents": [],
        "Other": []
    }
    
    for file_path in md_files:
        filename = file_path.name
        stem = file_path.stem
        
        # Determine category based on filename
        if filename.startswith("Source_"):
            categories["Source Documents"].append((stem, filename))
        elif filename.startswith("Research_"):
            categories["Research Wikis"].append((stem, filename))
        elif any(keyword in filename.lower() for keyword in ["assassin", "ac_", "ac-"]):
            categories["Entity Wikis"].append((stem, filename))
        else:
            categories["Other"].append((stem, filename))
    
    # Generate index content
    index_content = "# Wiki Index\n\n"
    
    # Add "All Documents" section that links to every wiki
    index_content += "## All Documents\n\n"
    for stem, filename in sorted([(f.stem, f.name) for f in md_files]):
        index_content += f"- [[{stem}]] ({filename})\n"
    index_content += "\n"
    
    # Add categorized sections
    for category, files in categories.items():
        if files:  # Only show categories that have files
            index_content += f"## {category}\n\n"
            for stem, filename in files:
                # Use the actual filename (stem) for the wiki link to match Quartz URL generation
                # Quartz converts underscores to hyphens in URLs, so we need to use the actual filename
                index_content += f"- [[{stem}]] ({filename})\n"
            index_content += "\n"
    
    # Write to index.md
    index_path = wikis_dir / "index.md"
    try:
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(index_content)
        print(f"✅ Wiki index recreated successfully with {len(md_files)} files!")
        print(f"📁 Index file: {index_path}")
    except Exception as e:
        print(f"❌ Error writing index file: {e}")


def load_processed_files():
    """Load processed files from Neo4j database."""
    return processed_docs_db.get_processed_files()


def save_processed_files(processed):
    """Save processed files to Neo4j database (no-op as files are saved individually)."""
    # Files are now saved individually when processed, so this is a no-op
    pass


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



def _process_chunks_parallel(chunks, extractor, thread_id, max_chunk_workers=3):
    """Process chunks in parallel streams for embeddings and triplet extraction."""
    print(f"[Thread {thread_id}] 🔄 Processing {len(chunks)} chunks in {max_chunk_workers} parallel streams...")
    
    # Progress tracking
    processed_count = 0
    total_chunks = len(chunks)
    
    def process_chunk_batch(chunk_batch, stream_id):
        """Process a batch of chunks in a single stream."""
        stream_triples = []
        batch_processed = 0
        
        for chunk in chunk_batch:
            try:
                txt = chunk.page_content
                result = extractor.extract_structured(txt)  # Use structured extraction
                entities = result["entities"]
                relations = result["relations"]
                batch_processed += 1
                
                if entities or relations:
                    # Get chunk_id from chunk metadata
                    chunk_id = chunk.metadata.get("chunk_id") if hasattr(chunk, 'metadata') else None
                    print(f"      🔍 DEBUG: chunk_id = {chunk_id}")
                    
                    # First, upsert all entities and relations to build the graph
                    extractor.upsert(result, None, chunk_id)  # No source_document_id in parallel processing
                    
                    # Then, identify primary entities using hierarchical clustering
                    print(f"      🔍 DEBUG: Starting hierarchical clustering for {len(entities)} entities")
                    primary_entities = extractor._identify_primary_entities(entities, relations)
                    print(f"      🔍 DEBUG: Identified {len(primary_entities)} primary entities: {primary_entities}")
                    
                    # Link only primary entities to chunks
                    if primary_entities and chunk_id:
                        print(f"      🔍 DEBUG: Linking {len(primary_entities)} entities to chunk {chunk_id}")
                        extractor._link_primary_entities_to_chunk(primary_entities, chunk_id, None)
                        print(f"      🔍 DEBUG: Linking completed")
                    else:
                        print(f"      🔍 DEBUG: Skipping link - primary_entities: {len(primary_entities) if primary_entities else 0}, chunk_id: {chunk_id}")
                    
                    # Convert to old format for compatibility
                    for relation in relations:
                        stream_triples.append((chunk_id, {
                            "head": relation["head"], "head_type": relation["head_type"],
                            "relation": relation["relation"], "tail": relation["tail"], "tail_type": relation["tail_type"]
                        }))
            except Exception as e:
                print(f"[Thread {thread_id}] [Stream {stream_id}] ⚠️ Error processing chunk: {e}")
                continue
        
        print(f"[Thread {thread_id}] [Stream {stream_id}] ✅ Processed {batch_processed} chunks, extracted {len(stream_triples)} triples")
        return stream_triples, batch_processed
    
    # Split chunks into batches for parallel processing
    chunk_batches = []
    batch_size = max(1, len(chunks) // max_chunk_workers)
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        chunk_batches.append(batch)
    
    # Process chunk batches in parallel
    all_triples = []
    total_processed = 0
    
    with ThreadPoolExecutor(max_workers=max_chunk_workers) as chunk_executor:
        future_to_batch = {
            chunk_executor.submit(process_chunk_batch, batch, stream_id): (batch, stream_id)
            for stream_id, batch in enumerate(chunk_batches, 1)
        }
        
        for future in as_completed(future_to_batch):
            batch, stream_id = future_to_batch[future]
            try:
                stream_triples, batch_processed = future.result()
                all_triples.extend(stream_triples)
                total_processed += batch_processed
                print(f"[Thread {thread_id}] 📊 Progress: {total_processed}/{total_chunks} chunks processed in parallel")
            except Exception as e:
                print(f"[Thread {thread_id}] [Stream {stream_id}] ❌ Error: {e}")
    
    print(f"[Thread {thread_id}] 🎯 Total triples extracted: {len(all_triples)} from {total_processed} chunks")
    return all_triples

def _create_embeddings_parallel(chunks, ingestor, thread_id, max_embedding_workers=3):
    """Create embeddings for chunks in parallel streams."""
    print(f"[Thread {thread_id}] 🔗 Creating embeddings for {len(chunks)} chunks in {max_embedding_workers} parallel streams...")
    
    def embed_chunk_batch(chunk_batch, stream_id):
        """Create embeddings for a batch of chunks."""
        embedded_chunks = []
        for chunk in chunk_batch:
            try:
                # Check if chunk already has embedding in metadata
                if chunk.metadata.get("embedding") is not None:
                    embedded_chunks.append(chunk)
                    continue
                
                # Create embedding for this chunk
                embedding = ingestor.embed_text(chunk.page_content)
                if embedding:
                    # Store embedding in metadata
                    chunk.metadata["embedding"] = embedding
                    embedded_chunks.append(chunk)
                else:
                    print(f"[Thread {thread_id}] [Embed Stream {stream_id}] ⚠️ Failed to create embedding for chunk")
            except Exception as e:
                print(f"[Thread {thread_id}] [Embed Stream {stream_id}] ⚠️ Error embedding chunk: {e}")
                continue
        return embedded_chunks
    
    # Split chunks into batches for parallel embedding
    chunk_batches = []
    batch_size = max(1, len(chunks) // max_embedding_workers)
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        chunk_batches.append(batch)
    
    # Process embedding batches in parallel
    all_embedded_chunks = []
    with ThreadPoolExecutor(max_workers=max_embedding_workers) as embed_executor:
        future_to_batch = {
            embed_executor.submit(embed_chunk_batch, batch, stream_id): (batch, stream_id)
            for stream_id, batch in enumerate(chunk_batches, 1)
        }
        
        for future in as_completed(future_to_batch):
            batch, stream_id = future_to_batch[future]
            try:
                embedded_batch = future.result()
                all_embedded_chunks.extend(embedded_batch)
                print(f"[Thread {thread_id}] [Embed Stream {stream_id}] ✅ Embedded {len(embedded_batch)} chunks")
            except Exception as e:
                print(f"[Thread {thread_id}] [Embed Stream {stream_id}] ❌ Error: {e}")
    
    print(f"[Thread {thread_id}] 🎯 Total embedded chunks: {len(all_embedded_chunks)}")
    return all_embedded_chunks

def _process_single_file(file_info, data_dir, components, thread_id):
    """Process a single file with all the processing steps in correct order."""
    f, file_idx, total_files = file_info
    ingestor, extractor, wiki, harvester, llm = components
    
    print(f"\n[Thread {thread_id}] 📄 Processing file {file_idx}/{total_files}: {f}")
    print(f"[Thread {thread_id}] " + "-" * 50)
    
    try:
        path = os.path.join(data_dir, f)
        
        # Step 1: Generate file hash for unique identification
        import hashlib
        with open(path, 'rb') as file:
            file_hash = hashlib.md5(file.read()).hexdigest()
        print(f"[Thread {thread_id}] 🔐 File hash: {file_hash}")

        # Step 2: Load and preview document
        print(f"[Thread {thread_id}] 📖 Loading document...")
        
        # Handle different file types
        if f.endswith('.docx'):
            try:
                # Convert .docx to text using docx2txt (already in requirements.txt)
                import docx2txt
                doc_text = docx2txt.process(path)
                print(f"[Thread {thread_id}] 📄 Converted .docx to text using docx2txt")
            except Exception as e:
                print(f"[Thread {thread_id}] ⚠️ docx2txt failed: {e}")
                # Try alternative method with python-docx
                try:
                    import docx
                    doc = docx.Document(path)
                    doc_text = ""
                    for paragraph in doc.paragraphs:
                        doc_text += paragraph.text + "\n"
                    print(f"[Thread {thread_id}] 📄 Converted .docx to text using python-docx")
                except Exception as e2:
                    print(f"[Thread {thread_id}] ⚠️ python-docx also failed: {e2}")
                    # Try to read as plain text (for corrupted ZIP files)
                    try:
                        with open(path, 'r', encoding='utf-8', errors='ignore') as file:
                            doc_text = file.read()
                        print(f"[Thread {thread_id}] 📄 Read as plain text (corrupted ZIP file)")
                    except Exception as e3:
                        print(f"[Thread {thread_id}] ❌ All methods failed: {e3}")
                        print(f"[Thread {thread_id}] ⚠️ Skipping .docx file: {f}")
                        return False, f"All .docx reading methods failed: {str(e)}"
        elif f.endswith('.pdf'):
            try:
                # Convert .pdf to text using PyMuPDF (already in requirements.txt)
                import fitz  # PyMuPDF
                doc = fitz.open(path)
                doc_text = ''
                for page in doc:
                    doc_text += page.get_text() + '\n'
                doc.close()
                print(f"[Thread {thread_id}] 📄 Converted .pdf to text using PyMuPDF")
            except ImportError:
                print(f"[Thread {thread_id}] ❌ PyMuPDF not installed. Install with: pip install PyMuPDF")
                print(f"[Thread {thread_id}] ⚠️ Skipping .pdf file: {f}")
                return False, "PyMuPDF not installed"
            except Exception as e:
                print(f"[Thread {thread_id}] ❌ Error reading .pdf file: {e}")
                print(f"[Thread {thread_id}] ⚠️ Skipping .pdf file: {f}")
                return False, str(e)
        else:
            # Handle .md and .txt files
            with open(path, 'r', encoding='utf-8', errors='ignore') as file:
                doc_text = file.read()
            print(f"[Thread {thread_id}] 📄 Read text file")
        
        # Check if document has meaningful content
        meaningful_content = _get_text_preview(doc_text)
        if not meaningful_content or meaningful_content.strip() == "":
            print(f"[Thread {thread_id}] ⚠️ Document appears to be empty or corrupted")
            # Still process it but mark as corrupted
            meaningful_content = "corrupted or unreadable document"
        
        print(f"[Thread {thread_id}] 📊 Document size: {len(doc_text)} characters")
        print(f"[Thread {thread_id}] 📝 Preview: {meaningful_content[:100]}...")

        # Step 3: Create document node in graph (single source of truth)
        print(f"[Thread {thread_id}] 📋 Creating document node in graph...")
        success = processed_docs_db.mark_file_processed(
            filename=f,
            file_hash=file_hash,
            file_path=path,
            file_size=len(doc_text),
            content_preview=meaningful_content[:200]
        )
        
        if success:
            print(f"[Thread {thread_id}] ✅ Document node created/updated")
        else:
            print(f"[Thread {thread_id}] ❌ Failed to create document node")
            return False, "Failed to create document node"

        # Step 4: Entity extraction and storage (but don't create wikis yet)
        print(f"[Thread {thread_id}] 👥 Extracting entities...")
        ents = harvester.harvest(doc_text)  # returns [{"name":..., "type":...}, ...]
        if ents:
            print(f"[Thread {thread_id}] ✅ Entities detected: {', '.join([e['name'] for e in ents])}")
            
            # Store entities in the database with MENTIONED_IN relationships
            print(f"[Thread {thread_id}] 💾 Storing entities in database...")
            try:
                with extractor.driver.session() as session:
                    for entity in ents:
                        # Create entity node
                        session.run(f"""
                        MERGE (e:{entity['type'].capitalize()} {{name: $name}})
                        MERGE (d:Document {{source_id: $source_id}})
                        MERGE (e)-[:MENTIONED_IN]->(d)
                        """, name=entity['name'], source_id=file_hash)
                print(f"[Thread {thread_id}] ✅ Stored {len(ents)} entities in database")
            except Exception as e:
                print(f"[Thread {thread_id}] ❌ Error storing entities: {e}")
        else:
            print(f"[Thread {thread_id}] ℹ️ No clear entities detected.")

        # Step 5: Research Study Detection (LLM-based)
        print(f"[Thread {thread_id}] 🔬 Detecting research study type...")
        research_detection = detect_research_study(doc_text, f, llm)
        print(f"[Thread {thread_id}] 📊 Detection result: {json.dumps(research_detection, indent=2)}")
        
        # Check if this document name matches any extracted entities (indicating it's a source file)
        name_no_ext = os.path.splitext(f)[0]
        document_is_source_file = any(
            name_no_ext.lower() in entity['name'].lower() or 
            entity['name'].lower() in name_no_ext.lower()
            for entity in ents
        )
        
        if document_is_source_file:
            print(f"[Thread {thread_id}] ⏸️ Skipping study wiki creation - document appears to be a source file")
            print(f"[Thread {thread_id}] 📝 Document name: '{name_no_ext}' matches extracted entities")
        # === Deferred wiki strategy ===
        # Detect + record, but do NOT emit wikis here.
        # Wikis are now generated using the advanced multi-step agent system.
        print(f"[Thread {thread_id}] 🧭 Deferred wiki creation: recorded detection and candidates; use option 3 to generate high-quality wikis.")

        # Step 6: Knowledge graph creation with nested parallelism
        print(f"[Thread {thread_id}] 🧠 Creating knowledge graph with nested parallelism...")
        print(f"[Thread {thread_id}] 📥 Ingesting document chunks...")
        chunks = ingestor.process(path)
        print(f"[Thread {thread_id}] ✅ Created {len(chunks)} chunks")

        # Create embeddings in parallel streams
        print(f"[Thread {thread_id}] 🔗 Creating embeddings in parallel streams...")
        try:
            embedded_chunks = _create_embeddings_parallel(chunks, ingestor, thread_id)
            print(f"[Thread {thread_id}] ✅ Created embeddings for {len(embedded_chunks)} chunks")
        except Exception as embed_error:
            print(f"[Thread {thread_id}] ⚠️ Parallel embedding failed: {embed_error}")
            print(f"[Thread {thread_id}] 🔄 Falling back to sequential embedding...")
            embedded_chunks = chunks

        # Extract triplets in parallel streams
        print(f"[Thread {thread_id}] 🧩 Extracting knowledge triplets in parallel streams...")
        try:
            all_triples = _process_chunks_parallel(embedded_chunks, extractor, thread_id)
            print(f"[Thread {thread_id}] ✅ Extracted {len(all_triples)} triples from parallel processing")
            
            # Store triples in the database
            if all_triples:
                for chunk_id, triple in all_triples:
                    extractor.store_triple(chunk_id, triple, source_document_id=file_hash)
                print(f"[Thread {thread_id}] ✅ Knowledge graph updated with {len(all_triples)} triples")
            else:
                print(f"[Thread {thread_id}] ⚠️ No triples extracted")
                
        except Exception as triplet_error:
            print(f"[Thread {thread_id}] ⚠️ Parallel triplet extraction failed: {triplet_error}")
            print(f"[Thread {thread_id}] 💡 This may be due to network connectivity issues.")
            print(f"[Thread {thread_id}] 🔄 Continuing with wiki creation using available data...")

        # Step 7: Entity wiki creation (NOW with complete knowledge graph)
        print(f"[Thread {thread_id}] 📝 Creating entity wikis with full context...")
        for e_idx, e in enumerate(ents, 1):
            try:
                print(f"[Thread {thread_id}] 🔍 Assessing entity {e_idx}/{len(ents)}: {e['name']} ({e['type']})")
                decision = wiki.assess_entity_worthiness(e, doc_text, f)
                msg = f"      merit={decision['merit']} | facts={decision['facts_count']} | ubisoft={decision['ubisoft_linked']} | reason={decision['reason']}"
                print(f"[Thread {thread_id}] {msg}")

                if decision["merit"]:
                    print(f"[Thread {thread_id}] ✅ Creating/updating entity wiki...")
                    wiki.upsert_entity_wiki(e, doc_text, f, file_hash)
                else:
                    if CREATE_EMPTY_STUBS:
                        print(f"[Thread {thread_id}] 📝 Creating empty stub (config enabled)...")
                        wiki.upsert_entity_wiki(e, "", f, file_hash)
                    else:
                        print(f"[Thread {thread_id}] ⏸️ Skipped: insufficient merit")
            except Exception as entity_error:
                print(f"[Thread {thread_id}] ❌ Error processing entity {e}: {entity_error}")
                continue

        # ATOMIC CHECKPOINT: File already marked as processed in database
        print(f"[Thread {thread_id}] 💾 Checkpoint saved - {f} marked as processed in database")

        # Step 8: Automatically create source document wiki (with accurate statistics)
        print(f"[Thread {thread_id}] 📄 Creating source document wiki...")
        try:
            from .wiki_agent import WikiGenerationAgent
            wiki_agent = WikiGenerationAgent()
            try:
                # Create wiki for this source document with accurate stats
                doc_info = {
                    "filename": f,
                    "file_size": len(doc_text),
                    "processed_at": "now",
                    "content_preview": meaningful_content[:200],
                    "total_chunks": len(chunks),
                    "entities_extracted": len(ents)
                }
                wiki_content = wiki_agent._create_source_document_wiki(doc_info)
                if wiki_content:
                    # Save to Wikis folder with source-files tag
                    filename = f.replace('.', '_').replace(' ', '_')
                    filepath = Path("Wikis") / f"Source_{filename}.md"
                    
                    with open(filepath, 'w', encoding='utf-8') as wiki_file:
                        wiki_file.write(wiki_content)
                    
                    # Add source wiki name to the graph for linking
                    wiki_agent._add_source_wiki_to_graph(f"Source_{filename}", file_hash)
                    
                    print(f"[Thread {thread_id}] ✅ Source document wiki created: {filepath}")
                    print(f"[Thread {thread_id}] 📊 Wiki stats: {len(chunks)} chunks, {len(ents)} entities")
                else:
                    print(f"[Thread {thread_id}] ⚠️ Could not create source document wiki")
            finally:
                wiki_agent.close()
        except Exception as wiki_error:
            print(f"[Thread {thread_id}] ⚠️ Error creating source document wiki: {wiki_error}")

        print(f"[Thread {thread_id}] ✅ File {f} processed successfully!")
        return True, "Success"

    except Exception as e:
        print(f"[Thread {thread_id}] ❌ Error processing {f}: {e}")
        import traceback
        print(f"[Thread {thread_id}] Traceback: {traceback.format_exc()}")
        print(f"[Thread {thread_id}] ⚠️ File {f} will be retried on next run")
        return False, str(e)
def process_new_files():
    """Process new files with parallel processing - up to 10 files at a time."""
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

    # Get unprocessed files directly from database
    new_files = processed_docs_db.get_unprocessed_files(all_files)
    
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
    from .litellm_wrapper import LiteLLMChat
    llm = LiteLLMChat()

    components = (ingestor, extractor, wiki, harvester, llm)

    print(f"\n🚀 Starting parallel processing of {len(new_files)} files...")
    print("=" * 60)
    print(f"⚡ Processing up to 10 files in parallel with queue system")
    
    # Prepare file info for processing
    file_info_list = [(f, idx, len(new_files)) for idx, f in enumerate(new_files, 1)]
    
    # Process files in parallel with max 10 workers
    max_workers = min(10, len(new_files))
    successful_files = []
    failed_files = []
    
    print(f"🔧 Using {max_workers} parallel workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(_process_single_file, file_info, data_dir, components, thread_id): file_info[0]
            for thread_id, file_info in enumerate(file_info_list, 1)
        }
        
        # Process completed tasks
        for future in as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                success, message = future.result()
                if success:
                    successful_files.append(filename)
                    print(f"✅ Completed: {filename}")
                else:
                    failed_files.append((filename, message))
                    print(f"❌ Failed: {filename} - {message}")
            except Exception as e:
                failed_files.append((filename, str(e)))
                print(f"❌ Exception in {filename}: {e}")

    # Get processing statistics from database
    stats = processed_docs_db.get_processing_stats()
    print(f"\n🎉 Parallel processing complete!")
    print(f"📊 Summary: {len(successful_files)} files processed successfully, {len(failed_files)} failed")
    print(f"📈 Database stats: {stats['total_processed']} total documents in database")
    print(f"📈 Database stats: {stats['processed_today']} processed today, {stats['total_size_mb']} MB total size")
    
    if failed_files:
        print(f"\n⚠️ Failed files (will be retried on next run):")
        for filename, error in failed_files:
            print(f"   📄 {filename}: {error}")
    
    # Clean up any orphaned document nodes
    processed_docs_db.cleanup_orphaned_documents(all_files)


def query_graph():
    print("🧠 Starting knowledge graph query interface...")
    
    # Get iteration configuration
    while True:
        try:
            print("\n🔧 Query Configuration:")
            print("How many iterations would you like?")
            print("- 1: Simple single question/answer (current behavior)")
            print("- 2-10: Iterative questioning with follow-ups")
            
            iterations = int(input("Enter number of iterations (1-10): ").strip())
            if 1 <= iterations <= 10:
                break
            else:
                print("❌ Please enter a number between 1 and 10.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    agent = SemanticAgent()
    conversation_context = []
    
    while True:
        q = input("\nAsk a question (or 'exit' to quit): ")
        if q.strip().lower() == "exit":
            break
            
        try:
            if iterations == 1:
                # Simple single Q&A (original behavior)
                print("🔍 Searching knowledge graph...")
                response = agent.run_query(q)
                print("\n🧠 Answer:")
                print(response)
            else:
                # Iterative Q&A with follow-ups
                print(f"🔍 Starting iterative analysis ({iterations} iterations)...")
                current_question = q
                
                for i in range(iterations):
                    print(f"\n🔄 Iteration {i + 1}/{iterations}")
                    print(f"❓ Question: {current_question}")
                    
                    response = agent.run_query(current_question)
                    conversation_context.append({"question": current_question, "answer": response})
                    
                    print(f"🧠 Answer:")
                    print(response)
                    
                    if i < iterations - 1:  # Not the last iteration
                        # Generate follow-up question
                        conversation_text = "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in conversation_context[-3:]])
                        follow_up_prompt = f"""
                        Based on this conversation so far:
                        {conversation_text}
                        
                        Generate a relevant follow-up question that would deepen the understanding of the topic.
                        The question should be specific and build on the previous answers.
                        Return only the question, nothing else.
                        """
                        
                        try:
                            from .litellm_wrapper import LiteLLMChat
                            llm = LiteLLMChat()
                            current_question = llm.invoke(follow_up_prompt)
                            print(f"🎯 Generated follow-up: {current_question}")
                        except Exception as e:
                            print(f"❌ Error generating follow-up: {e}")
                            break
                
                # Final synthesis
                if len(conversation_context) > 1:
                    print(f"\n📋 Final Synthesis:")
                    conversation_text = "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in conversation_context])
                    synthesis_prompt = f"""
                    Based on this complete conversation:
                    {conversation_text}
                    
                    Provide a comprehensive summary that synthesizes all the information gathered.
                    Organize the key insights and findings clearly.
                    """
                    
                    try:
                        synthesis = agent.run_query(synthesis_prompt)
                        print(synthesis)
                    except Exception as e:
                        print(f"❌ Error in synthesis: {e}")
                
        except Exception as e:
            print(f"❌ Query failed: {e}")
        
        # Clear conversation context for next question
        conversation_context = []

# Legacy functions removed - not part of current workflow


# def force_reprocess_existing_documents():  # REMOVED - No longer needed
    # """Force reprocess all existing documents to fix .docx files and update the database."""
    # print("🔄 Force Reprocessing Existing Documents...")
    # 
    # data_dir = "data"
    # if not os.path.exists(data_dir):
    #     print(f"❌ Data directory '{data_dir}' not found!")
    #     return
    # 
    # # Get all files in data directory
    # all_files = [f for f in os.listdir(data_dir) 
    #              if f.endswith(('.md', '.txt', '.pdf', '.docx'))]
    # 
    # if not all_files:
    #     print("ℹ️ No files found in data directory.")
    #     return
    # 
    # print(f"📊 Found {len(all_files)} files to reprocess:")
    # for f in all_files:
    #     print(f"   📄 {f}")
    # 
    # # Initialize components
    # print("\n🔧 Initializing processing components...")
    # ingestor = DocumentIngestor()
    # extractor = TripletExtractor()
    # wiki = WikiCreator()
    # harvester = EntityHarvest()
    # 
    # # Initialize LLM for research study detection
    # from litellm_wrapper import LiteLLMChat
    # llm = LiteLLMChat()
    # 
    # print(f"\n🚀 Starting force reprocessing of {len(all_files)} files...")
    # print("=" * 60)
    # 
    # # Process files one at a time
    # for file_idx, f in enumerate(all_files, 1):
    #     print(f"\n📄 Reprocessing file {file_idx}/{len(all_files)}: {f}")
    #     print("-" * 50)
    # 
    #     try:
    #         path = os.path.join(data_dir, f)
    # 
    #         # Step 1: Generate file hash for unique identification
    #         import hashlib
    #         with open(path, 'rb') as file:
    #             file_hash = hashlib.md5(file.read()).hexdigest()
    #         print(f"🔐 File hash: {file_hash}")
    # 
    #         # Step 2: Load and preview document (with proper .docx handling)
    #         print(f"📖 Loading document...")
    # 
    #         # Handle different file types
    #         if f.endswith('.docx'):
    #         try:
    #             # Convert .docx to text using docx2txt (already in requirements.txt)
    #             import docx2txt
    #             doc_text = docx2txt.process(path)
    #             print(f"   📄 Converted .docx to text using docx2txt")
    #         except ImportError:
    #             print(f"   ❌ docx2txt not installed. Install with: pip install docx2txt")
    #             print(f"   ⚠️ Skipping .docx file: {f}")
    #             continue
    #         except Exception as e:
    #             print(f"   ❌ Error reading .docx file: {e}")
    #             print(f"   ⚠️ Skipping .pdf file: {f}")
    #             continue
    #     elif f.endswith('.pdf'):
    #         try:
    #             # Convert .pdf to text using PyMuPDF (already in requirements.txt)
    #             import fitz  # PyMuPDF
    #             doc = fitz.open(path)
    #             doc_text = ''
    #             for page in doc:
    #             doc_text += page.get_text() + '\n'
    #         doc.close()
    #         print(f"   📄 Converted .pdf to text using PyMuPDF")
    #     except ImportError:
    #         print(f"   ❌ PyMuPDF not installed. Install with: pip install PyMuPDF")
    #         print(f"   ⚠️ Skipping .pdf file: {f}")
    #         continue
    #     except Exception as e:
    #         print(f"   ❌ Error reading .pdf file: {e}")
    #         print(f"   ⚠️ Skipping .pdf file: {f}")
    #         continue
    #     else:
    #         # Handle .md and .txt files
    #         with open(path, 'r', encoding='utf-8', errors='ignore') as file:
    #             doc_text = file.read()
    #         print(f"   📄 Read text file")
    # 



def main():
    print("🎯 Knowledge Graph Processing System")
    print("=" * 50)
    
    try:
        while True:
            print("\nOptions:")
            print("1. Process new files")
            print("2. Query the knowledge graph")
            print("3. Generate High-Quality Wikis (Multi-Step Agent)")
            print("4. Create Research Wiki")
            print("5. Recreate Wiki Index")
            print("6. Rescan ALL Wikis for Cross-Linking")
            print("7. Exit")

            choice = input("\nChoose an option: ").strip()
            if choice == "1":
                process_new_files()
            elif choice == "2":
                query_graph()
            elif choice == "3":
                print("🔨 Running Multi-Step Wiki Generation Agent...")
                from .wiki_agent import WikiGenerationAgent
                agent = WikiGenerationAgent()
                try:
                    # Get recommendations and let user select
                    topics = agent.get_wiki_recommendations()
                    if topics:
                        selected_topics = agent.interactive_topic_selection(topics)
                        if selected_topics:
                            config = agent.get_iteration_limit()
                            
                            # Process each selected topic
                            for i, selected in enumerate(selected_topics, 1):
                                print(f"\n{'='*60}")
                                print(f"📝 Processing {i}/{len(selected_topics)}: {selected.name}")
                                print(f"{'='*60}")
                                
                                # Check if this is an update or new wiki
                                wiki_info = agent._check_existing_wiki(selected.name)
                                if wiki_info["exists"]:
                                    print(f"\n🔄 Updating existing wiki for {selected.name}...")
                                    wiki_content = agent.update_existing_wiki(selected, config["total"])
                                else:
                                    print(f"\n🆕 Creating new wiki for {selected.name}...")
                                    wiki_content = agent.generate_wiki(selected, config)
                                
                                filepath = agent.save_wiki(selected, wiki_content)
                                print(f"\n🎉 Wiki {'updated' if wiki_info['exists'] else 'generated'} successfully: {filepath}")
                            
                            # Ask if user wants to create cross-links for all processed wikis
                            if len(selected_topics) > 1:
                                create_links = input(f"\n🔗 Create cross-links for all {len(selected_topics)} processed wikis? (y/n): ").strip().lower()
                            else:
                                create_links = input("\n🔗 Create cross-links to other wikis? (y/n): ").strip().lower()
                            if create_links == 'y':
                                agent.create_cross_links()
                finally:
                    agent.close()
            elif choice == "4":
                print("🔬 Creating Research Wiki...")
                from .wiki_agent import WikiGenerationAgent
                agent = WikiGenerationAgent()
                try:
                    research_question = input("Enter your research question or topic: ").strip()
                    if research_question:
                        config = agent.get_iteration_limit()
                        wiki_content = agent.create_research_wiki(research_question, config["total"])
                        filepath = agent.save_research_wiki(research_question, wiki_content)
                        print(f"\n🎉 Research wiki created successfully: {filepath}")
                    else:
                        print("❌ No research question provided.")
                finally:
                    agent.close()
            elif choice == "5":
                recreate_wiki_index()
            elif choice == "6":
                print("🔗 Rescanning ALL wikis for cross-linking...")
                from .wiki_agent import WikiGenerationAgent
                agent = WikiGenerationAgent()
                try:
                    # Get all wiki files
                    wiki_folder = Path("Wikis")
                    wiki_files = [f for f in wiki_folder.glob("*.md") if f.name != "index.md"]
                    
                    print(f"📊 Found {len(wiki_files)} wikis to rescan")
                    
                    # Get list of all wiki names for linking
                    existing_wikis = set()
                    for wiki_file in wiki_files:
                        existing_wikis.add(wiki_file.stem)
                    
                    print(f"📋 Available wikis for cross-linking: {len(existing_wikis)}")
                    
                    updated_count = 0
                    
                    for wiki_file in wiki_files:
                        print(f"🔍 Processing: {wiki_file.name}")
                        
                        try:
                            # Read the wiki content
                            with open(wiki_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            # Skip if it's a Source_ wiki (they don't need cross-linking)
                            if wiki_file.name.startswith("Source_"):
                                print(f"   ⏭️ Skipping Source_ wiki")
                                continue
                            
                            # Apply smart wiki linking to the content
                            updated_content = agent._add_smart_wiki_links(content)
                            
                            # Check if content was updated
                            if updated_content != content:
                                # Write the updated content back
                                with open(wiki_file, 'w', encoding='utf-8') as f:
                                    f.write(updated_content)
                                
                                print(f"   ✅ Updated with cross-links")
                                updated_count += 1
                            else:
                                print(f"   ℹ️ No new cross-links needed")
                                
                        except Exception as e:
                            print(f"   ❌ Error processing {wiki_file.name}: {e}")
                            continue
                    
                    print(f"\n🎉 Cross-linking rescan complete!")
                    print(f"📊 Updated {updated_count} out of {len(wiki_files)} wikis")
                    
                    # Also update the index
                    print("\n🔄 Updating wiki index...")
                    agent.update_wiki_index()
                    
                except Exception as e:
                    print(f"❌ Error in cross-linking rescan: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    agent.close()
            elif choice == "7":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice, try again.")
    finally:
        # Ensure database connection is properly closed
        processed_docs_db.close()


if __name__ == "__main__":
    main()

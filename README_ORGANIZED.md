# Knowledge Graph Processing System - Organized Structure

## 📁 Project Structure

```
KnowledgeGraph/
├── main.py                     # Main entry point
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables
├── processed.json              # Legacy processed files list
├── ontology.json               # Ontology schema
├── glossary.json               # Glossary for canonical names
├── wiki_schema.json            # Wiki schema definitions
├── wiki_candidates.json        # Wiki generation candidates
│
├── core/                       # Core processing modules
│   ├── __init__.py
│   ├── main.py                 # Main processing logic
│   ├── document_ingestor.py    # Document loading & chunking
│   ├── triplet_extractor.py    # Knowledge triplet extraction
│   ├── entity_harvest.py       # Entity extraction
│   ├── db_utils.py             # Database utilities
│   ├── semantic_agent.py       # Q&A agent
│   ├── wiki_agent.py           # Wiki generation agent
│   ├── wiki_creator.py         # Wiki creation logic
│   ├── litellm_wrapper.py      # LLM API wrapper
│   ├── glossary_resolver.py    # Canonical name resolution
│   └── ontology_expander.py    # Ontology gap analysis
│
├── tools/                      # Utility tools
│   ├── create_athena_wiki.py   # Athena wiki creation
│   ├── llm_analyze_graph_db.py # Claude graph analysis
│   ├── demo_graph_analysis.py  # Graph analysis demo
│   └── get_models.py           # Model listing utility
│
├── docs/                       # Documentation
│   ├── CONTEXT_AWARENESS_SUMMARY.md
│   ├── GRAPH_ANALYSIS_README.md
│   └── USER_GUIDANCE_FEATURE_SUMMARY.md
│
├── archive/                    # Archived/unused files
│   ├── query_chunk.py          # Old query chunk tool
│   ├── query_chunk_simple.py   # Old simple query tool
│   ├── main_backup.py          # Backup of main.py
│   ├── Ontology Examples/      # Example ontologies
│   └── temp/                   # Old test files
│
├── analysis/                   # Analysis results
│   ├── graph_analysis_by_claude_1.md
│   └── graph_analysis_by_claude_2.md
│
├── data/                       # Input documents
│   └── *.md, *.pdf, *.docx, *.txt
│
└── Wikis/                      # Generated wiki files
    ├── index.md
    ├── Source_*.md
    ├── Company_*.md
    ├── Person_*.md
    └── Installment_*.md
```

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your Neo4j and LiteLLM credentials
   ```

3. **Run the system:**
   ```bash
   python main.py
   ```

## 🔧 Core Modules

### Processing Pipeline
- **`core/main.py`**: Main processing logic with parallel file processing
- **`core/document_ingestor.py`**: Document loading, chunking, and embedding
- **`core/triplet_extractor.py`**: Knowledge triplet extraction using LLM
- **`core/entity_harvest.py`**: Named entity extraction
- **`core/db_utils.py`**: Neo4j database operations

### AI Components
- **`core/semantic_agent.py`**: RAG-based Q&A system
- **`core/wiki_agent.py`**: Multi-step wiki generation
- **`core/wiki_creator.py`**: Wiki creation and management
- **`core/litellm_wrapper.py`**: LLM API integration

### Utilities
- **`core/glossary_resolver.py`**: Canonical name resolution
- **`core/ontology_expander.py`**: Ontology gap analysis

## 🛠️ Tools

- **`tools/create_athena_wiki.py`**: Create research wikis
- **`tools/llm_analyze_graph_db.py`**: Claude-powered graph analysis
- **`tools/demo_graph_analysis.py`**: Graph analysis demonstrations
- **`tools/get_models.py`**: List available LLM models

## 📊 Features

- **Parallel Processing**: Up to 10 files processed simultaneously
- **Nested Parallelism**: 3 streams for embeddings and triplets
- **Knowledge Extraction**: Entities, relationships, and triplets
- **Wiki Generation**: Automated markdown wiki creation
- **RAG System**: Semantic search and Q&A
- **Graph Analysis**: Claude-powered database analysis

## 🔄 Migration Notes

- All core functionality moved to `core/` folder
- Old unused files moved to `archive/`
- Import paths updated for new structure
- Main entry point remains `main.py`
- All functionality preserved, just reorganized


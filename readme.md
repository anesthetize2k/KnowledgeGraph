# 🧠 KnowledgeGraph

A modular, LangGraph-powered pipeline that ingests government PDFs into a Neo4j knowledge graph and supports semantic + symbolic querying using GraphRAG.

---

## 📦 Features

* Ingest PDFs, chunk text, embed with OpenAI
* Extract \[SUBJECT, RELATION, OBJECT] triplets using an ontology-aware LLM
* Store structured knowledge in Neo4j (Aura or local)
* Graph + Vector RAG query engine via LangGraph
* CLI tools for ingestion and querying
* Ontology auto-expansion + triplet parsing
* Switch between Neo4j cloud or Docker-based local DB
* **Multi-Step Wiki Generation System** with intelligent topic selection
* **Research Wiki Creation** for question-driven research articles
* **Automatic Cross-Linking** between wikis for Quartz4 compatibility

---

## 🚀 Installation

### 1. Clone the repo

```bash
git clone https://github.com/your-user/KnowledgeGraph.git
cd KnowledgeGraph
```

### 2. Create & activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
# OR
source .venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file:

```dotenv
# For local Neo4j (default)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=test1234

# Optional: OpenAI
OPENAI_API_KEY=sk-...

# Optional: LangSmith tracing
LANGCHAIN_API_KEY=...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=KnowledgeGraph

# Optional: Aura config (if you want to switch back)
# NEO4J_AURA_URI=neo4j+s://...databases.neo4j.io
# NEO4J_AURA_USERNAME=neo4j
# NEO4J_AURA_PASSWORD=...
```

---

## 🐳 Run Neo4j Locally via Docker

```bash
docker pull neo4j

docker run --name neo4j-local \
  -p 7474:7474 -p 7687:7687 \
  -d \
  -e NEO4J_AUTH=neo4j/test1234 \
  neo4j
```

Open your browser:

> [http://localhost:7474](http://localhost:7474)
> Username: `neo4j`
> Password: `test1234`

---

## 🧪 Usage

### ▶ Ingest all new PDFs

Put your `.pdf` files in `/data`, then run:

```bash
python main.py
# Select option 1: Process new PDFs
```

This will:

* Chunk text
* Embed with OpenAI
* Store in Neo4j
* Extract triplets
* Update ontology

Processed files are tracked in the Neo4j database as Document nodes.

### ▶ Ask a question

```bash
python main.py
# Select option 2: Query the knowledge graph
```

Then try:

```
💬 Who is the Defence Minister?
💬 Which ministry oversees the PMJDY scheme?
```

### ▶ Generate High-Quality Wikis

```bash
python main.py
# Select option 3: Generate High-Quality Wikis (Multi-Step Agent)
```

This advanced system:
- Analyzes your knowledge graph for high-quality wiki candidates
- Lets you select which topics deserve wikis
- Uses multi-iteration refinement (5-20 queries) for comprehensive content
- Automatically creates cross-links between wikis

### ▶ Create Research Wikis

```bash
python main.py
# Select option 4: Create Research Wiki
```

Create question-driven research articles:
- Start with any research question
- System automatically plans the investigation
- Gathers evidence from your entire knowledge graph
- Produces comprehensive, well-structured research articles

---

## 🧱 Project Structure

```
KnowledgeGraph/
├── data/                    # PDF files
├── Wikis/                   # Generated wikis and research articles
│   ├── Custom Research Wikis/  # Question-driven research articles
│   └── [Entity Wikis]         # Entity-specific wikis
├── temp/                    # Temporary files (git-ignored)
├── main.py                  # Main CLI interface
├── wiki_agent.py            # Multi-step wiki generation system
├── semantic_agent.py        # Classic graph-based QA
├── triplet_extractor.py     # Ontology-aware LLM extraction
├── document_ingestor.py     # Chunking + embedding
├── db_utils.py              # Database utilities for processed documents
├── ontology.json            # Ontology schema
├── requirements.txt         # Dependencies
├── .env                     # Local secrets (not committed)
```

---

## 🗄️ Database Management

### Processed Documents Storage

Processed documents are now stored in Neo4j as Document nodes instead of local files. This provides:

* **Centralized tracking**: All processing status is stored in the database
* **Better performance**: No need to read/write local files
* **Data consistency**: Processing status is always in sync with the knowledge graph
* **Scalability**: Can handle large numbers of processed documents efficiently

### Database Utilities

The `db_utils.py` module provides:

* `ProcessedDocumentsDB` class for managing processed documents
* Functions to check file processing status
* Statistics and reporting capabilities
* Cleanup utilities for orphaned document nodes

---

## 🎯 Multi-Step Wiki Generation System

The wiki generation system has been completely redesigned from a random, low-quality approach to a sophisticated, multi-step agent system that produces high-quality, comprehensive wikis through iterative knowledge graph exploration.

### Key Features

1. **Smart Topic Recommendations**
   - Analyzes knowledge graph for high-quality wiki candidates
   - Scores topics by relevance, evidence count, and centrality
   - Prioritizes entities with sufficient data for comprehensive wikis

2. **Interactive Topic Selection**
   - Presents ranked recommendations with detailed metrics
   - Shows evidence counts, source documents, and relevance scores
   - Allows manual topic selection for specific entities

3. **Multi-Iteration Generation**
   - Configurable iteration limits (5-20 queries)
   - Strategic questioning based on knowledge gaps
   - Iterative refinement of wiki content
   - Automatic integration of new information

4. **High-Quality Content**
   - Comprehensive evidence gathering from knowledge graph
   - Structured outline creation and refinement
   - Professional markdown formatting with YAML metadata
   - Source citations and evidence-based content

5. **Cross-Linking System**
   - Automatic detection of related entities
   - Creates `[[Entity Name]]` links for Quartz4 compatibility
   - Builds interconnected wiki network

### Usage

```bash
# Run the main system and select option 3
python main.py

# Or run the wiki agent directly
python wiki_agent.py
```

---

## 🔬 Research Wiki System

The Research Wiki feature allows you to create custom research articles based on specific questions or topics. Unlike entity-based wikis that focus on specific entities in your knowledge graph, research wikis are question-driven and explore topics across your entire knowledge base.

### How It Works

1. **Question Input**: Provide a research question or topic
2. **Research Planning**: System creates a comprehensive research plan
3. **Iterative Research**: Configurable iterations (5-20) with strategic questioning
4. **Final Synthesis**: Combines findings into a comprehensive, well-structured wiki

### Example Research Questions

- **Game Development**: "What are the most effective player onboarding strategies?"
- **Player Behavior**: "What factors contribute to player churn?"
- **Business & Marketing**: "What are the key metrics for measuring game success?"

### Usage

```bash
# Run the main system and select option 4
python main.py

# Or run the wiki agent directly
python wiki_agent.py
```

Research wikis are saved in `Wikis/Custom Research Wikis/` with proper Markdown formatting and YAML metadata.

---

## 🔁 Switching between Local & Aura

In code, use:

```python
from graph_config import get_neo4j_config
cfg = get_neo4j_config(use_aura=False)  # or True
```

You can maintain both in `.env`:

```dotenv
# Local
NEO4J_URI=bolt://localhost:7687
...

# Cloud (Aura)
NEO4J_AURA_URI=neo4j+s://...databases.neo4j.io
...
```

---

## 🔮 Roadmap / Advanced Ideas

* [ ] LangSmith tracing for all LangGraph nodes
* [ ] LangGraph memory (chat history)
* [ ] Docker Compose for persistent Neo4j
* [ ] LangChain eval: triplet quality tests
* [ ] Multiple domain ontologies (e.g., France, India, global policy)
* [ ] Template system for different wiki types
* [ ] Collaborative wiki editing
* [ ] Version control for wikis
* [ ] Quality metrics for generated content

---

## 📄 Sample Cypher Queries

```cypher
// Show all nodes and relationships
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100

// See all chunks for a document
MATCH (d:Document {source_id: "Ministry_of_Finance"})-[:HAS_CHUNK]->(c:Chunk)
RETURN d, c

// Find all policies implemented by a department
MATCH (d:department)-[:IMPLEMENTS]->(p:policy)
RETURN d.name, p.name

// Get processing statistics
MATCH (d:Document) RETURN count(d) as total_documents
```

---

## 🎯 Benefits of the New System

- **Quality Control**: You decide what deserves a wiki
- **Depth**: Multi-iteration approach creates comprehensive content
- **Efficiency**: Focuses on high-value topics with sufficient evidence
- **Consistency**: Structured approach ensures uniform wiki quality
- **Interconnected**: Cross-linking creates a true wiki network
- **Scalable**: Can handle large knowledge graphs efficiently
- **Research-Driven**: Create wikis for any research question, not just entities
- **Evidence-Based**: All content is grounded in your knowledge graph data

---

**Ready to start? Run `python main.py` to access all features!** 🚀



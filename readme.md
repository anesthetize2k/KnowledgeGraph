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
python run_ingestion.py
```

This will:

* Chunk text
* Embed with OpenAI
* Store in Neo4j
* Extract triplets
* Update ontology

Processed files are tracked in `processed.json`.

### ▶ Ask a question

```bash
python run_query.py
```

Then try:

```
💬 Who is the Defence Minister?
💬 Which ministry oversees the PMJDY scheme?
```

---

## 🧱 Project Structure

```
KnowledgeGraph/
├── data/                    # PDF files
├── nodes/                  # LangGraph pipeline nodes
├── run_ingestion.py        # CLI for ingestion
├── run_query.py            # CLI for querying
├── query_graph.py          # LangGraph: vector + graph query flow
├── ingestion_graph.py      # LangGraph: PDF → KG ingestion flow
├── semantic_agent.py       # (optional) classic graph-based QA
├── triplet_extractor.py    # Ontology-aware LLM extraction
├── document_ingestor.py    # Chunking + embedding
├── main.py                 # Combined CLI (legacy)
├── processed.json          # Tracks processed files
├── ontology.json           # Ontology schema
├── requirements.txt        # Dependencies
├── .env                    # Local secrets (not committed)
```

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
```



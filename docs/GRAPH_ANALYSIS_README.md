# Graph Database Analysis Tool

## Overview

The `llm_analyze_graph_db.py` tool uses Claude Sonnet 4 to analyze your Neo4j graph database like a true data analyst. It systematically explores the graph structure, identifies patterns, and finds potential issues or improvements.

## Features

### 🤖 AI-Powered Analysis
- Uses Claude Sonnet 4 for intelligent graph exploration
- Understands your graph database schema and business logic
- Provides contextual analysis of query results
- Identifies data quality issues and inconsistencies

### 🔍 Systematic Exploration
- Starts with basic exploratory queries (SELECT * LIMIT 100, etc.)
- Progressively dives deeper into graph structure
- Analyzes entity relationships and patterns
- Looks for missing connections and anomalies

### 📊 Issue Detection
- Identifies data quality problems
- Finds missing relationships
- Detects schema inconsistencies
- Suggests improvements to graph structure

### 📝 Comprehensive Reporting
- Creates separate analysis files: `analysis/graph_analysis_by_claude_1.md`, `analysis/graph_analysis_by_claude_2.md`, etc.
- Each run creates a new numbered analysis file
- Includes full conversation logs and query results
- Tracks issues found and recommendations

## How It Works

### 1. System Understanding
The tool includes a comprehensive system prompt that explains:
- Your graph database architecture and purpose
- Document processing pipeline (`process_new_files` logic)
- Entity types and relationship schemas
- Graph RAG system components
- Query capabilities and use cases

### 2. Analysis Process
1. **Initial Exploration**: Starts with basic queries to understand graph structure
2. **Progressive Analysis**: Claude generates follow-up queries based on results
3. **Issue Detection**: Identifies problems, inconsistencies, or missing data
4. **Deep Dive**: Explores specific areas of concern
5. **Reporting**: Saves comprehensive analysis report

### 3. Query Examples
The tool can execute queries like:
```cypher
// Basic exploration
MATCH (n) RETURN labels(n) as node_types, count(n) as count ORDER BY count DESC LIMIT 20
MATCH (n)-[r]->(m) RETURN type(r) as rel_types, count(r) as count ORDER BY count DESC LIMIT 20

// Entity analysis
MATCH (b:Brand) RETURN b.name, size((b)<-[:BELONGS_TO_BRAND]-()) as installments
MATCH (i:Installment) RETURN i.name, size((i)-[:DEVELOPED_BY]->()) as studios

// Relationship analysis
MATCH (n)-[r]->(m) WHERE type(r) = 'BELONGS_TO_BRAND'
RETURN n.name, m.name, count(r) as frequency ORDER BY frequency DESC
```

## Usage

### Quick Start
```bash
python llm_analyze_graph_db.py
```

### Interactive Modes
1. **Autonomous Analysis with User Guidance** (Recommended): Let Claude explore the graph systematically with your real-time guidance
2. **Guided Analysis**: Provide specific queries for Claude to analyze
3. **Continue from Previous Analysis**: Load a previous analysis and ask follow-up questions
4. **Demo Mode**: Run basic exploration without full AI analysis

### Demo Script
```bash
python demo_graph_analysis.py
```

## New Feature: Continue from Previous Analysis

### Overview
The new "Continue from Previous Analysis" option allows you to:
- **Load Previous Analysis**: Choose from a list of recent analysis files
- **Context-Aware Querying**: Ask specific questions with full analysis context
- **Interactive Exploration**: Execute additional queries based on previous findings

### How It Works
1. **File Selection**: Lists all previous analysis files with timestamps
2. **Context Loading**: Loads the complete analysis content as context
3. **Interactive Querying**: Ask specific questions about the graph database
4. **Query Execution**: Execute suggested Cypher queries interactively
5. **Analysis Integration**: Claude uses previous findings to provide better insights

### Usage Example
```
🔍 Interactive Graph Database Analysis
==================================================
Options:
1. Let Claude analyze autonomously (recommended)
2. Guide Claude with specific queries
3. Continue from a previous analysis
4. Exit

Choose an option (1-4): 3

📚 Previous Analysis Files
========================================
Available analysis files:
  1. graph_analysis_by_claude_1.md (modified: 2025-01-04 00:28:10)
  2. graph_analysis_by_claude_2.md (modified: 2025-01-03 15:30:45)

Choose an analysis file (1-2): 1

📖 Loading analysis from: graph_analysis_by_claude_1.md
📊 Analysis Summary:
   Total Queries: 9
   Issues Found: 9
   Analysis Date: 2025-01-04 00:28:10

🔍 Interactive Query Mode with Analysis Context
============================================================
Ask a question about the graph database (or 'done' to finish): 
```

### Benefits
- **Contextual Understanding**: Claude remembers previous analysis findings
- **Targeted Questions**: Ask specific questions about identified issues
- **Deep Dive**: Explore particular areas of concern in detail
- **Follow-up Analysis**: Build on previous discoveries
- **User Guidance**: Provide specific directions and hints to Claude
- **Interactive Analysis**: Guide Claude's thinking and analysis approach

## New Feature: User Guidance and Hints

### Overview
The enhanced "Continue from Previous Analysis" mode now allows you to provide specific guidance and hints to Claude, making the analysis more interactive and directed.

### How It Works
1. **Ask a Question**: Start with any question about the graph database
2. **Claude Responds**: Get Claude's initial analysis and suggestions with full system context
3. **Automatic Query Execution**: Claude automatically executes Cypher queries if needed
4. **Provide Guidance**: Give specific directions, hints, or focus areas
5. **Claude Adapts**: Claude adjusts its approach based on your guidance and previous analysis
6. **Automatic Analysis**: Claude analyzes query results and provides insights
7. **Context Awareness**: Claude maintains conversation context and analysis history

### Guidance Examples
```
💡 Provide guidance to Claude (optional):
   - Give specific directions or hints
   - Suggest what to focus on or investigate
   - Ask Claude to execute specific queries
   - Just press Enter to let Claude continue autonomously

Your guidance/hints: Focus on the Chunk node relationships and check for orphaned chunks
```

### Types of Guidance You Can Provide
- **Focus Areas**: "Look specifically at the Metric relationships"
- **Investigation Directions**: "Check for data quality issues in the Brand nodes"
- **Query Suggestions**: "Run a query to find all entities with missing relationships"
- **Analysis Priorities**: "Prioritize performance issues over data quality"
- **Specific Concerns**: "I'm worried about the high number of Concept nodes"

### Automatic Query Execution
Claude now automatically executes Cypher queries when needed:

- **Smart Detection**: Claude determines if a query is needed to answer your question
- **Automatic Execution**: No manual query input required
- **Error Handling**: Claude handles query errors and provides alternatives
- **Query Correction**: Claude automatically corrects failed queries and retries
- **Result Analysis**: Claude automatically analyzes query results and provides insights

### Example Interaction Flow
```
Ask a question about the graph database: What are the main issues in this graph?

🧠 Claude's Response:
[Claude provides initial analysis and suggests queries]

🔍 Claude is executing: MATCH (n) WHERE NOT (n)--() RETURN labels(n) as node_type, count(n) as count
✅ Query executed successfully - 5 results
📊 Sample results: [{"node_type": ["Chunk"], "count": 3}]

🔍 Claude's Analysis of Query Results:
[Claude analyzes the results and provides insights]

💡 Provide guidance to Claude (optional):
Your guidance/hints: Focus on the Chunk nodes and their connections to documents

🧠 Claude's Response to Your Guidance:
[Claude acknowledges your direction and adjusts approach]

🔍 Claude is executing: MATCH (c:Chunk) OPTIONAL MATCH (c)-[r]->(d:Document) RETURN count(c) as chunks, count(d) as connected_docs
✅ Query executed successfully - 1 results
📊 Sample results: [{"chunks": 8160, "connected_docs": 45}]

🔍 Claude's Analysis of Query Results:
[Claude analyzes the results in context of your guidance]
```

### Enhanced Error Handling
The system now includes intelligent error handling and query correction:

- **Automatic Error Detection**: Claude identifies query syntax and property errors
- **Query Correction**: Claude automatically provides corrected queries
- **Retry Logic**: Failed queries are automatically retried with corrections
- **Property Validation**: Claude uses correct property names and relationship types
- **Graceful Degradation**: System continues working even when queries fail

### Example Error Handling Flow
```
🔍 Claude is executing: MATCH (d:Document) WHERE NOT (d)--() RETURN d.id, d.title, d.source
❌ Query failed: 'frozenset' object is not callable

🔍 Claude's Analysis of the Error:
[Claude analyzes the error and provides corrections]

🔍 Claude is executing corrected query: MATCH (d:Document) WHERE NOT (d)--() RETURN d.name, d.path
✅ Corrected query executed successfully - 3 results
📊 Sample results: [{"d.name": "document1.pdf", "d.path": "/path/to/file"}]

🔍 Claude's Analysis of Corrected Query Results:
[Claude analyzes the corrected results and provides insights]
```

### Enhanced Context Awareness
The system now includes comprehensive context awareness:

- **System Prompt Integration**: Claude has full understanding of your graph database system
- **Analysis File Context**: Option 3 loads the complete previous analysis file as context
- **Conversation Memory**: Claude maintains context between queries and iterations
- **Business Logic Understanding**: Claude understands your domain and business rules
- **Progressive Analysis**: Each query builds on previous findings and insights

### Context-Aware Features
- **Option 1 (Autonomous)**: Claude maintains context between analysis iterations
- **Option 3 (Continue)**: Claude has full access to previous analysis findings
- **System Understanding**: Claude knows your graph structure, entity types, and relationships
- **Domain Knowledge**: Claude understands Ubisoft/gaming context and business logic
- **Progressive Discovery**: Each analysis builds on previous findings

## New Feature: User Guidance in Autonomous Analysis

### Overview
Option 1 (Autonomous Analysis) now includes real-time user guidance between each analysis iteration, allowing you to direct Claude's analysis in real-time.

### How It Works
1. **Claude Starts Analysis**: Begins with basic exploratory queries
2. **User Guidance Between Iterations**: You can provide specific directions between each query
3. **Claude Adapts**: Claude adjusts its next query based on your guidance
4. **Progressive Analysis**: Each iteration builds on previous findings and your guidance

### Example Interaction Flow
```
🔍 Starting Graph Database Analysis with Claude Sonnet 4
============================================================

🤖 Claude: [Initial analysis prompt]

🔍 Executing query 1: MATCH (n) RETURN labels(n) as node_types, count(n) as count ORDER BY count DESC LIMIT 20
✅ Query executed successfully - 10 results
📊 Sample results: [{"node_types": ["Chunk"], "count": 8160}]

🧠 Claude's analysis: [Claude analyzes the results]

💬 Provide guidance for the next analysis step (optional):
   - Give specific directions or hints
   - Suggest what to focus on or investigate
   - Ask Claude to explore specific areas
   - Just press Enter to let Claude continue autonomously

Your guidance: Focus on the Chunk nodes and check for orphaned chunks

🤖 Claude's next query: MATCH (c:Chunk) WHERE NOT (c)-[:HAS_CHUNK]-() RETURN count(c)
✅ Query executed successfully - 1 results
📊 Sample results: [{"count(c)": 3}]

🧠 Claude's analysis: [Claude analyzes results with your guidance context]
```

### Types of Guidance You Can Provide
- **Focus Areas**: "Look specifically at the Metric relationships"
- **Investigation Directions**: "Check for data quality issues in the Brand nodes"
- **Query Suggestions**: "Run a query to find all entities with missing relationships"
- **Analysis Priorities**: "Prioritize performance issues over data quality"
- **Specific Concerns**: "I'm worried about the high number of Concept nodes"

### Benefits
- **Real-time Control**: Guide Claude's analysis as it progresses
- **Targeted Investigation**: Focus on specific areas of concern
- **Interactive Analysis**: Not just autonomous, but collaborative
- **Flexible Guidance**: Provide guidance or let Claude continue autonomously

## Configuration

### Environment Variables
Make sure your `.env` file contains:
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
LITELLM_API_KEY=your_api_key
```

### Analysis Settings
- **Max Queries**: Default 10 queries per analysis session
- **Analysis Folder**: Results saved to `analysis/` folder
- **Report Files**: `analysis/graph_analysis_by_claude_1.md`, `analysis/graph_analysis_by_claude_2.md`, etc.
- **Auto-incrementing**: Each run creates a new numbered file

## Analysis Output

### Console Output
- Real-time query execution and results
- Claude's analysis of each query result
- Issue detection and recommendations
- Progress tracking and statistics

### Report File
The analysis report includes:
- **Summary**: Total queries, issues found, analysis duration
- **Issues**: Detailed descriptions of problems found
- **Conversation Log**: Full dialogue between system and Claude
- **Query Results**: All Cypher queries and their results
- **Recommendations**: Suggested improvements and fixes

## Example Analysis Session

```
🔍 Starting Graph Database Analysis with Claude Sonnet 4
============================================================

🤖 Claude: You are now analyzing a Neo4j graph database...

🔍 Executing query 1: MATCH (n) RETURN labels(n) as node_types, count(n) as count ORDER BY count DESC LIMIT 20
✅ Query executed successfully - 8 results
📊 Sample results: [{"node_types": ["Document"], "count": 45}, {"node_types": ["Chunk"], "count": 1203}]

🧠 Claude's analysis: I can see the graph contains 45 documents and 1203 chunks, which suggests a well-populated knowledge base. Let me explore the entity types next...

🔍 Executing query 2: MATCH (n) WHERE NOT 'Document' IN labels(n) AND NOT 'Chunk' IN labels(n) RETURN labels(n) as entity_types, count(n) as count ORDER BY count DESC LIMIT 10
✅ Query executed successfully - 6 results
📊 Sample results: [{"entity_types": ["Brand"], "count": 12}, {"entity_types": ["Installment"], "count": 8}]

🧠 Claude's analysis: I see 12 brands and 8 installments. Let me check if all installments are properly connected to brands...

⚠️ Issue detected! Total issues found: 1
```

## Integration with Your System

### Graph RAG Components
The tool understands your complete system:
- **Document Processing**: File ingestion, chunking, embedding
- **Entity Extraction**: LLM-based entity recognition
- **Triplet Extraction**: Relationship identification
- **Knowledge Graph**: Neo4j storage and querying
- **Wiki Generation**: Markdown wiki creation
- **Semantic Search**: Vector similarity search

### Business Logic
Claude understands your domain:
- **Ubisoft Context**: Game franchises, studios, platforms
- **Entity Types**: Brands, installments, studios, publishers, platforms
- **Relationships**: belongs_to_brand, developed_by, published_by, etc.
- **Research Studies**: Document classification and analysis
- **Cross-References**: Entity linking across documents

## Troubleshooting

### Common Issues
1. **Database Connection**: Ensure Neo4j is running and credentials are correct
2. **API Keys**: Verify LiteLLM API key is set in environment
3. **Permissions**: Check Neo4j user has read permissions
4. **Network**: Ensure network connectivity to Neo4j instance

### Error Messages
- `Database connection failed`: Check Neo4j URI and credentials
- `Query failed`: Verify Cypher syntax and database state
- `API call failed`: Check LiteLLM API key and rate limits

## Advanced Usage

### Custom Analysis
You can modify the system prompt in `_create_system_prompt()` to:
- Add domain-specific knowledge
- Include custom entity types
- Specify analysis priorities
- Add business rules and constraints

### Query Templates
The tool includes query templates for common analysis patterns:
- Node type distribution
- Relationship analysis
- Entity connectivity
- Data quality checks
- Schema validation

## Contributing

To extend the analysis tool:
1. Add new query templates to the system prompt
2. Implement custom analysis patterns
3. Add domain-specific issue detection
4. Enhance reporting capabilities

## Support

For issues or questions:
1. Check the console output for error messages
2. Verify database connectivity and permissions
3. Review the analysis report for detailed logs
4. Check environment variables and API keys

#!/usr/bin/env python3
"""
LLM Graph Database Analyst - Uses Claude Sonnet 4 to analyze Neo4j graph database
as a true analyst, exploring the graph structure and identifying issues.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
from litellm_wrapper import LiteLLMChat

load_dotenv()

class GraphDBAnalyst:
    def __init__(self):
        """Initialize the graph database analyst with Claude Sonnet 4."""
        self.llm = LiteLLMChat("claude-sonnet-4")
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
        )
        self.analysis_folder = Path("analysis")
        self.analysis_folder.mkdir(exist_ok=True)
        self.analysis_file = self._get_next_analysis_file()
        
        # Create system prompt explaining the graph database system
        self.system_prompt = self._create_system_prompt()
    
    def _get_next_analysis_file(self):
        """Get the next analysis file with incrementing sequence number."""
        import glob
        
        # Find all existing analysis files with the pattern
        pattern = str(self.analysis_folder / "graph_analysis_by_claude_*.md")
        existing_files = glob.glob(pattern)
        
        if not existing_files:
            # No existing files, start with sequence 1
            return self.analysis_folder / "graph_analysis_by_claude_1.md"
        
        # Extract sequence numbers from existing files
        sequence_numbers = []
        for file_path in existing_files:
            filename = Path(file_path).name
            # Extract number from filename like "graph_analysis_by_claude_1.md"
            if "_" in filename:
                try:
                    # Get the part after the last underscore, before .md
                    number_part = filename.split("_")[-1].replace(".md", "")
                    sequence_numbers.append(int(number_part))
                except ValueError:
                    continue
        
        if not sequence_numbers:
            # No valid sequence numbers found, start with 1
            return self.analysis_folder / "graph_analysis_by_claude_1.md"
        
        # Get the next sequence number
        next_sequence = max(sequence_numbers) + 1
        return self.analysis_folder / f"graph_analysis_by_claude_{next_sequence}.md"
        
    def _create_system_prompt(self):
        """Create comprehensive system prompt explaining the graph database system."""
        return """You are Claude, an expert graph database analyst working with a Neo4j knowledge graph that captures information from Ubisoft's internal documents. Your role is to analyze this graph database like a true data analyst, exploring its structure, understanding its content, and identifying potential issues or improvements.

## GRAPH DATABASE SYSTEM OVERVIEW

### Purpose & Architecture
This is a sophisticated Graph RAG (Retrieval-Augmented Generation) system that processes Ubisoft's internal documents to create a comprehensive knowledge graph. The system:

1. **Document Processing Pipeline**: 
   - Ingests documents (PDF, DOCX, MD, TXT) from a `data/` folder
   - Creates document nodes with metadata (filename, hash, size, preview)
   - Splits documents into chunks with embeddings for semantic search
   - Links chunks to their source documents

2. **Entity Extraction & Knowledge Graph Creation**:
   - Extracts entities using LLM-based entity recognition
   - Identifies relationships between entities using triplet extraction
   - Creates a rich knowledge graph with typed nodes and relationships
   - Links entities to source documents and chunks

3. **Graph Structure**:
   - **Document Nodes**: Source documents with metadata
   - **Chunk Nodes**: Text chunks with embeddings for semantic search
   - **Entity Nodes**: Typed entities (Brand, Installment, Studio, Person, etc.)
   - **Relationship Edges**: Semantic relationships between entities
   - **Mention Edges**: Links entities to documents/chunks where they appear

### Entity Types & Relationships
The system uses a controlled ontology with specific entity types:
- **brand**: Game franchises (e.g., "Assassin's Creed")
- **installment**: Specific games (e.g., "Assassin's Creed Shadows")
- **studio**: Development studios (e.g., "Ubisoft Quebec")
- **publisher**: Publishing companies (e.g., "Ubisoft")
- **platform**: Gaming platforms (e.g., "PlayStation 5", "PC")
- **person**: People mentioned in documents
- **company**: Companies and organizations
- **research_study**: Research documents and studies
- **time_period**: Time references (e.g., "Q1 2024")
- **insight**: Key insights or findings
- **metric**: Performance metrics and KPIs
- **region**: Geographic regions
- **audience_type**: Player segments and demographics

### Relationship Types
- **belongs_to_brand**: Installment → Brand
- **developed_by**: Installment → Studio
- **published_by**: Installment → Publisher
- **released_on**: Installment → Platform
- **measured_by**: Metric → Entity
- **analyzes**: Study → Entity
- **compares_with**: Entity → Entity
- **has_insight**: Entity → Insight
- **targets_segment**: Entity → Audience
- **covers_period**: Entity → Time Period
- **is_synonym_of**: Entity → Entity (for name variations)

### Processing Workflow (process_new_files logic)
The system processes documents through this pipeline:

1. **File Discovery**: Scans `data/` folder for new documents
2. **Document Loading**: Handles multiple formats (PDF, DOCX, MD, TXT)
3. **Hash Generation**: Creates unique file identifiers
4. **Document Node Creation**: Creates Document nodes in Neo4j
5. **Entity Extraction**: Uses LLM to identify entities in documents
6. **Research Study Detection**: Classifies documents as research studies
7. **Chunking & Embedding**: Splits documents into searchable chunks with embeddings
8. **Triplet Extraction**: Extracts entity relationships using LLM
9. **Graph Population**: Stores entities and relationships in Neo4j
10. **Wiki Generation**: Creates markdown wikis for entities and documents
11. **Cross-linking**: Links related entities and documents

### Key Files & Components
- **main.py**: Main processing pipeline with parallel file processing
- **document_ingestor.py**: Handles document loading and chunking
- **triplet_extractor.py**: Extracts entity relationships from text
- **entity_harvest.py**: Identifies entities in documents
- **semantic_agent.py**: Provides graph-based question answering
- **wiki_creator.py**: Generates markdown wikis for entities
- **db_utils.py**: Database utilities and document tracking

### Graph Query Capabilities
The system supports:
- **Semantic Search**: Vector similarity search on chunk embeddings
- **Entity Relationship Queries**: Traverse entity connections
- **Document Retrieval**: Find documents containing specific entities
- **Cross-Reference Analysis**: Link entities across multiple documents
- **Temporal Analysis**: Time-based relationship queries
- **Hierarchical Queries**: Brand → Installment → Platform relationships

## YOUR ANALYSIS TASK

As a graph database analyst, you should:

1. **Explore the Graph Structure**: Start with basic queries to understand the data
2. **Identify Patterns**: Look for common entity types, relationship patterns
3. **Find Anomalies**: Detect inconsistencies, missing relationships, data quality issues
4. **Analyze Completeness**: Check if the graph captures expected relationships
5. **Suggest Improvements**: Recommend schema changes, data quality fixes, or new relationships

### Analysis Approach
- Start with exploratory queries (SELECT * LIMIT 100, etc.)
- Progress to more specific analysis queries
- Look for data quality issues, missing relationships, inconsistencies
- Focus on business logic and domain-specific patterns
- Identify potential improvements to the graph structure

### Query Examples to Start With
```cypher
// Basic exploration
MATCH (n) RETURN labels(n) as node_types, count(n) as count ORDER BY count DESC LIMIT 20
MATCH (n)-[r]->(m) RETURN type(r) as rel_types, count(r) as count ORDER BY count DESC LIMIT 20
MATCH (n) RETURN n LIMIT 100

// Entity analysis
MATCH (b:Brand) RETURN b.name, size((b)<-[:BELONGS_TO_BRAND]-()) as installments
MATCH (i:Installment) RETURN i.name, size((i)-[:DEVELOPED_BY]->()) as studios
MATCH (p:Platform) RETURN p.name, size((p)<-[:RELEASED_ON]-()) as games

// Relationship analysis
MATCH (n)-[r]->(m) 
WHERE type(r) = 'BELONGS_TO_BRAND'
RETURN n.name, m.name, count(r) as frequency
ORDER BY frequency DESC LIMIT 20
```

Remember: You are analyzing a real production graph database. Be thorough, systematic, and look for both obvious issues and subtle problems that could affect the system's effectiveness.
"""

    def execute_cypher_query(self, query: str) -> dict:
        """Execute a Cypher query and return results."""
        try:
            with self.driver.session() as session:
                result = session.run(query)
                records = list(result)
                
                # Convert to a more readable format
                if not records:
                    return {"results": [], "count": 0, "message": "No results found"}
                
                # Get column names
                columns = list(records[0].keys()) if records else []
                
                # Convert records to dictionaries
                formatted_results = []
                for record in records:
                    formatted_record = {}
                    for key in columns:
                        value = record[key]
                        # Handle Neo4j node objects
                        if hasattr(value, 'labels') and hasattr(value, 'items'):
                            formatted_record[key] = {
                                "labels": list(value.labels()),
                                "properties": dict(value.items())
                            }
                        else:
                            formatted_record[key] = value
                    formatted_results.append(formatted_record)
                
                return {
                    "results": formatted_results,
                    "count": len(formatted_results),
                    "columns": columns,
                    "query": query
                }
        except Exception as e:
            return {
                "error": str(e),
                "query": query,
                "results": [],
                "count": 0
            }

    def analyze_graph(self, max_queries=10):
        """Main analysis loop - let Claude analyze the graph like a true analyst."""
        print("🔍 Starting Graph Database Analysis with Claude Sonnet 4")
        print("=" * 60)
        
        conversation_history = []
        query_count = 0
        issues_found = []
        
        # Create comprehensive system prompt for autonomous analysis
        system_prompt = self._create_system_prompt()
        
        # Initial analysis prompt with full system context
        initial_prompt = f"""{system_prompt}

You are now analyzing a Neo4j graph database containing Ubisoft's internal knowledge. 

Your task is to explore this graph systematically and identify any issues, inconsistencies, or areas for improvement. Start with basic exploratory queries to understand the data structure and content.

You can execute Cypher queries to explore the graph. I'll execute them and return the results to you.

Begin your analysis by running some initial exploratory queries to understand:
1. What types of nodes exist in the graph
2. What types of relationships exist
3. The overall structure and content of the data
4. Any obvious patterns or anomalies

Start with a simple query like: `MATCH (n) RETURN labels(n) as node_types, count(n) as count ORDER BY count DESC LIMIT 20`

What would you like to explore first?"""

        print("🤖 Claude: " + initial_prompt)
        conversation_history.append({"role": "assistant", "content": initial_prompt})
        
        while query_count < max_queries:
            try:
                # Get user input (in this case, we'll simulate Claude's responses)
                if query_count == 0:
                    # First query - let's start with basic exploration
                    first_query = "MATCH (n) RETURN labels(n) as node_types, count(n) as count ORDER BY count DESC LIMIT 20"
                    print(f"\n🔍 Executing query {query_count + 1}: {first_query}")
                else:
                    # Ask for user comment/guidance between iterations
                    print(f"\n💬 Provide guidance for the next analysis step (optional):")
                    print("   - Give specific directions or hints")
                    print("   - Suggest what to focus on or investigate")
                    print("   - Ask Claude to explore specific areas")
                    print("   - Just press Enter to let Claude continue autonomously")
                    
                    user_comment = input("\nYour guidance: ").strip()
                    
                    # Generate next query based on previous results with full context and user guidance
                    analysis_prompt = f"""{system_prompt}

Based on the previous query results and conversation history, generate the next Cypher query to continue your analysis. 

Previous conversation:
{json.dumps(conversation_history[-3:], indent=2)}

User guidance for this iteration: {user_comment if user_comment else "Continue autonomously"}

Focus on:
1. Exploring the graph structure more deeply
2. Looking for data quality issues
3. Identifying missing relationships
4. Finding inconsistencies or anomalies
5. Analyzing business logic and domain patterns

Return ONLY a valid Cypher query, nothing else."""
                    
                    claude_response = self.llm.invoke(analysis_prompt)
                    # Extract query from response (remove any markdown formatting)
                    query_text = claude_response.strip()
                    if query_text.startswith("```cypher"):
                        query_text = query_text[9:]
                    if query_text.startswith("```"):
                        query_text = query_text[3:]
                    if query_text.endswith("```"):
                        query_text = query_text[:-3]
                    query_text = query_text.strip()
                    
                    print(f"\n🤖 Claude's next query: {query_text}")
                    first_query = query_text
                
                # Execute the query
                results = self.execute_cypher_query(first_query)
                
                if "error" in results:
                    print(f"❌ Query failed: {results['error']}")
                    conversation_history.append({"role": "user", "content": f"Query failed: {results['error']}"})
                else:
                    print(f"✅ Query executed successfully - {results['count']} results")
                    if results['count'] > 0:
                        print(f"📊 Sample results: {json.dumps(results['results'][:3], indent=2)}")
                    
                # Add to conversation history
                conversation_entry = {
                    "role": "user", 
                    "content": f"Query: {first_query}\nResults: {json.dumps(results, indent=2)}"
                }
                
                # Add user comment if provided
                if query_count > 0 and 'user_comment' in locals() and user_comment:
                    conversation_entry["user_guidance"] = user_comment
                
                conversation_history.append(conversation_entry)
                
                # Analyze the results with full context
                user_guidance_text = ""
                if query_count > 0 and 'user_comment' in locals() and user_comment:
                    user_guidance_text = f"\nUser guidance for this iteration: {user_comment}"
                
                analysis_prompt = f"""{system_prompt}

Analyze these query results and determine if you've found any issues or if you need to explore further.

Query: {first_query}
Results: {json.dumps(results, indent=2)}
{user_guidance_text}

Previous conversation context:
{json.dumps(conversation_history[-2:], indent=2)}

Based on these results:
1. What patterns or anomalies do you see?
2. Are there any data quality issues?
3. Do you need to explore further with another query?
4. Have you found any significant issues that need attention?

If you've found an issue, describe it clearly. If you need to explore more, suggest the next query. If you're satisfied with the analysis and found no major issues, say "ANALYSIS_COMPLETE_NO_ISSUES_FOUND"."""

                analysis_response = self.llm.invoke(analysis_prompt)
                print(f"\n🧠 Claude's analysis: {analysis_response}")
                conversation_history.append({"role": "assistant", "content": analysis_response})
                
                # Check if analysis is complete
                if "ANALYSIS_COMPLETE_NO_ISSUES_FOUND" in analysis_response:
                    print("\n🎉 Analysis complete - no major issues found!")
                    break
                
                # Check if an issue was found
                if any(keyword in analysis_response.lower() for keyword in ["issue", "problem", "inconsistency", "error", "missing", "anomaly"]):
                    issues_found.append({
                        "query": first_query,
                        "results": results,
                        "analysis": analysis_response,
                        "timestamp": datetime.now().isoformat()
                    })
                    print(f"\n⚠️ Issue detected! Total issues found: {len(issues_found)}")
                
                query_count += 1
                
            except Exception as e:
                print(f"❌ Error in analysis loop: {e}")
                break
        
        # Final analysis
        if query_count >= max_queries:
            print(f"\n⏰ Reached maximum query limit ({max_queries})")
        
        # Generate final report
        self._generate_analysis_report(conversation_history, issues_found, query_count)
        
        return {
            "total_queries": query_count,
            "issues_found": len(issues_found),
            "issues": issues_found,
            "conversation": conversation_history
        }

    def _generate_analysis_report(self, conversation_history, issues_found, query_count):
        """Generate and save the analysis report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_content = f"""# Graph Database Analysis Report
**Generated by Claude Sonnet 4 on {timestamp}**

## Analysis Summary
- **Total Queries Executed**: {query_count}
- **Issues Found**: {len(issues_found)}
- **Analysis Duration**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Issues Identified
"""
        
        if issues_found:
            for i, issue in enumerate(issues_found, 1):
                report_content += f"""
### Issue #{i}
**Query**: `{issue['query']}`
**Analysis**: {issue['analysis']}
**Timestamp**: {issue['timestamp']}

**Query Results**:
```json
{json.dumps(issue['results'], indent=2)}
```
"""
        else:
            report_content += "\n✅ No issues found during analysis.\n"
        
        report_content += f"""
## Full Conversation Log
"""
        
        for i, entry in enumerate(conversation_history, 1):
            role = "Claude" if entry["role"] == "assistant" else "System"
            report_content += f"""
### {role} - Entry #{i}
{entry['content']}
"""
        
        # Create new analysis file (don't append to existing)
        try:
            with open(self.analysis_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"📄 Analysis report saved to: {self.analysis_file}")
        except Exception as e:
            print(f"❌ Error saving analysis report: {e}")

    def interactive_analysis(self):
        """Interactive analysis mode where user can guide the analysis."""
        print("🔍 Interactive Graph Database Analysis")
        print("=" * 50)
        print("Claude will analyze your graph database step by step.")
        print("You can guide the analysis or let Claude explore autonomously.")
        print("\nOptions:")
        print("1. Let Claude analyze autonomously (recommended)")
        print("2. Guide Claude with specific queries")
        print("3. Continue from a previous analysis")
        print("4. Exit")
        
        choice = input("\nChoose an option (1-4): ").strip()
        
        if choice == "1":
            print("\n🤖 Starting autonomous analysis...")
            return self.analyze_graph()
        elif choice == "2":
            return self._guided_analysis()
        elif choice == "3":
            return self._continue_from_previous_analysis()
        elif choice == "4":
            print("👋 Goodbye!")
            return None
        else:
            print("❌ Invalid choice")
            return self.interactive_analysis()

    def _guided_analysis(self):
        """Guided analysis where user provides specific queries."""
        print("\n🎯 Guided Analysis Mode")
        print("Enter Cypher queries for Claude to analyze. Type 'done' when finished.")
        
        conversation_history = []
        issues_found = []
        query_count = 0
        
        while True:
            query = input(f"\nQuery #{query_count + 1}: ").strip()
            
            if query.lower() == 'done':
                break
            
            if not query:
                continue
            
            print(f"🔍 Executing: {query}")
            results = self.execute_cypher_query(query)
            
            if "error" in results:
                print(f"❌ Error: {results['error']}")
            else:
                print(f"✅ Results: {results['count']} records")
                if results['count'] > 0:
                    print(f"📊 Sample: {json.dumps(results['results'][:2], indent=2)}")
            
            # Let Claude analyze the results
            analysis_prompt = f"""Analyze these query results and identify any issues or patterns:

Query: {query}
Results: {json.dumps(results, indent=2)}

What do you observe? Are there any data quality issues, inconsistencies, or areas for improvement?"""

            analysis = self.llm.invoke(analysis_prompt)
            print(f"\n🧠 Claude's analysis: {analysis}")
            
            conversation_history.append({
                "query": query,
                "results": results,
                "analysis": analysis
            })
            
            if any(keyword in analysis.lower() for keyword in ["issue", "problem", "inconsistency", "error", "missing", "anomaly"]):
                issues_found.append({
                    "query": query,
                    "results": results,
                    "analysis": analysis
                })
                print(f"⚠️ Issue detected! Total issues: {len(issues_found)}")
            
            query_count += 1
        
        # Generate report
        self._generate_analysis_report(conversation_history, issues_found, query_count)
        
        return {
            "total_queries": query_count,
            "issues_found": len(issues_found),
            "issues": issues_found
        }

    def _continue_from_previous_analysis(self):
        """Continue analysis from a previous analysis file."""
        print("\n📚 Previous Analysis Files")
        print("=" * 40)
        
        # Find all existing analysis files
        import glob
        pattern = str(self.analysis_folder / "graph_analysis_by_claude_*.md")
        existing_files = glob.glob(pattern)
        
        if not existing_files:
            print("❌ No previous analysis files found.")
            print("💡 Run a new analysis first using option 1 or 2.")
            return None
        
        # Sort files by modification time (most recent first)
        existing_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        print("Available analysis files:")
        for i, file_path in enumerate(existing_files, 1):
            filename = Path(file_path).name
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S")
            print(f"  {i}. {filename} (modified: {mod_time})")
        
        # Let user choose a file
        while True:
            try:
                choice = input(f"\nChoose an analysis file (1-{len(existing_files)}): ").strip()
                choice_num = int(choice)
                if 1 <= choice_num <= len(existing_files):
                    selected_file = existing_files[choice_num - 1]
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {len(existing_files)}")
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n👋 Cancelled")
                return None
        
        print(f"\n📖 Loading analysis from: {Path(selected_file).name}")
        
        # Load the analysis content
        try:
            with open(selected_file, 'r', encoding='utf-8') as f:
                analysis_content = f.read()
        except Exception as e:
            print(f"❌ Error reading analysis file: {e}")
            return None
        
        # Extract key information from the analysis
        analysis_summary = self._extract_analysis_summary(analysis_content)
        
        print(f"\n📊 Analysis Summary:")
        print(f"   Total Queries: {analysis_summary.get('total_queries', 'Unknown')}")
        print(f"   Issues Found: {analysis_summary.get('issues_found', 'Unknown')}")
        print(f"   Analysis Date: {analysis_summary.get('analysis_date', 'Unknown')}")
        
        # Start interactive querying with context
        return self._interactive_query_with_context(analysis_content, analysis_summary, selected_file)
    
    def _extract_analysis_summary(self, analysis_content):
        """Extract summary information from analysis content."""
        summary = {}
        
        # Extract total queries
        import re
        queries_match = re.search(r'Total Queries Executed.*?(\d+)', analysis_content)
        if queries_match:
            summary['total_queries'] = int(queries_match.group(1))
        
        # Extract issues found
        issues_match = re.search(r'Issues Found.*?(\d+)', analysis_content)
        if issues_match:
            summary['issues_found'] = int(issues_match.group(1))
        
        # Extract analysis date
        date_match = re.search(r'Generated by Claude Sonnet 4 on ([\d-]+ [\d:]+)', analysis_content)
        if date_match:
            summary['analysis_date'] = date_match.group(1)
        
        return summary
    
    def _interactive_query_with_context(self, analysis_content, analysis_summary, selected_file):
        """Interactive querying with previous analysis as context."""
        print(f"\n🔍 Interactive Query Mode with Analysis Context")
        print("=" * 60)
        print("You can now ask specific questions about the graph database.")
        print("Claude will use the previous analysis as context.")
        print("Type 'done' when finished.")
        
        # Create comprehensive system prompt with full analysis context
        system_prompt = self._create_system_prompt()
        
        # Create context-aware system prompt with full analysis file
        context_prompt = f"""{system_prompt}

## Previous Analysis Context:
{analysis_content}

## Your Task:
Answer specific questions about the graph database using both your knowledge and the previous analysis context. You can execute Cypher queries to explore the data and provide detailed insights.

When asked a question:
1. Understand what the user is asking
2. Execute relevant Cypher queries if needed
3. Provide comprehensive answers with context from the previous analysis
4. Identify any new issues or patterns that emerge

You have access to execute Cypher queries on the Neo4j database. Be thorough and analytical in your responses."""

        conversation_history = []
        
        while True:
            print(f"\n" + "="*60)
            question = input("Ask a question about the graph database (or 'done' to finish): ").strip()
            
            if question.lower() == 'done':
                break
            
            if not question:
                continue
            
            print(f"\n🤖 Processing question: {question}")
            
            # Let Claude analyze the question with context and automatically execute queries if needed
            analysis_prompt = f"""{context_prompt}

User Question: {question}

Please provide a detailed response that:
1. Answers the user's specific question
2. References relevant information from the previous analysis
3. If you need to execute a Cypher query to answer the question, format it as: EXECUTE_QUERY: [your cypher query]
4. If no query is needed, just respond normally

IMPORTANT: 
- If you need to execute a query, format it as: EXECUTE_QUERY: [your cypher query]
- If no query is needed, just respond normally
- Use correct property names: 'name' (not 'id', 'title', 'source')
- Use correct relationship types: 'HAS_CHUNK' (not 'CONTAINS', 'PART_OF')
- Be specific about what you'll investigate and how you'll approach it"""

            claude_response = self.llm.invoke(analysis_prompt)
            print(f"\n🧠 Claude's Response:")
            print(claude_response)
            
            # Check if Claude wants to execute a query automatically
            if "EXECUTE_QUERY:" in claude_response:
                # Extract the query from the response
                query_start = claude_response.find("EXECUTE_QUERY:") + len("EXECUTE_QUERY:")
                query_end = claude_response.find("\n", query_start)
                if query_end == -1:
                    query_end = len(claude_response)
                
                query_input = claude_response[query_start:query_end].strip()
                
                if query_input:
                    print(f"\n🔍 Claude is executing: {query_input}")
                    results = self.execute_cypher_query(query_input)
                    
                    if "error" in results:
                        print(f"❌ Query failed: {results['error']}")
                        # Let Claude analyze the error and provide a corrected query
                        error_prompt = f"""The query failed with this error: {results['error']}

Original question: {question}
Failed query: {query_input}

Please analyze the error and provide a corrected Cypher query. Common issues:
1. Property names that don't exist (use 'name' instead of 'id', 'title', 'source')
2. Relationship types that don't exist (use 'HAS_CHUNK' instead of 'CONTAINS', 'PART_OF')
3. Syntax errors in the query

Provide a corrected query in this format: EXECUTE_QUERY: [corrected cypher query]"""
                        error_response = self.llm.invoke(error_prompt)
                        print(f"\n🔍 Claude's Analysis of the Error:")
                        print(error_response)
                        
                        # Check if Claude provided a corrected query
                        if "EXECUTE_QUERY:" in error_response:
                            # Extract the corrected query
                            query_start = error_response.find("EXECUTE_QUERY:") + len("EXECUTE_QUERY:")
                            query_end = error_response.find("\n", query_start)
                            if query_end == -1:
                                query_end = len(error_response)
                            
                            corrected_query = error_response[query_start:query_end].strip()
                            
                            if corrected_query:
                                print(f"\n🔍 Claude is executing corrected query: {corrected_query}")
                                corrected_results = self.execute_cypher_query(corrected_query)
                                
                                if "error" in corrected_results:
                                    print(f"❌ Corrected query also failed: {corrected_results['error']}")
                                else:
                                    print(f"✅ Corrected query executed successfully - {corrected_results['count']} results")
                                    if corrected_results['count'] > 0:
                                        print(f"📊 Sample results: {json.dumps(corrected_results['results'][:3], indent=2)}")
                                    
                                    # Let Claude analyze the corrected results
                                    corrected_analysis_prompt = f"""You executed this corrected query: {corrected_query}

Results: {json.dumps(corrected_results, indent=2)}

Original question: {question}

Please analyze these results and provide insights about what they reveal."""
                                    
                                    corrected_analysis = self.llm.invoke(corrected_analysis_prompt)
                                    print(f"\n🔍 Claude's Analysis of Corrected Query Results:")
                                    print(corrected_analysis)
                    else:
                        print(f"✅ Query executed successfully - {results['count']} results")
                        if results['count'] > 0:
                            print(f"📊 Sample results: {json.dumps(results['results'][:3], indent=2)}")
                        
                        # Let Claude analyze the results automatically
                        follow_up_prompt = f"""You executed this query: {query_input}

Results: {json.dumps(results, indent=2)}

Original question: {question}

Please analyze these results and provide insights about what they reveal. Answer the user's question based on these results."""
                        
                        follow_up_response = self.llm.invoke(follow_up_prompt)
                        print(f"\n🔍 Claude's Analysis of Query Results:")
                        print(follow_up_response)
            else:
                # No query needed, Claude provided a natural language response
                print(f"\n💬 Claude's natural language response (no query needed)")
            
            # Ask for user guidance and hints
            print(f"\n💡 Provide guidance to Claude (optional):")
            print("   - Give specific directions or hints")
            print("   - Suggest what to focus on or investigate")
            print("   - Ask Claude to execute specific queries")
            print("   - Just press Enter to let Claude continue autonomously")
            
            user_guidance = input("\nYour guidance/hints: ").strip()
            
            if user_guidance:
                # Let Claude process the user guidance and automatically execute queries
                guidance_prompt = f"""{context_prompt}

The user has provided additional guidance and hints for your analysis:

User Guidance: {user_guidance}

Original Question: {question}
Your Previous Response: {claude_response}

Based on this guidance, please:
1. Acknowledge the user's direction
2. Determine if you need to execute any Cypher queries to answer their question
3. If you need to execute a query, provide ONLY a valid Cypher query in this format: EXECUTE_QUERY: [cypher query here]
4. If no query is needed, provide your response in natural language

IMPORTANT: 
- If you need to execute a query, format it as: EXECUTE_QUERY: [your cypher query]
- If no query is needed, just respond normally
- Use correct property names: 'name' (not 'id', 'title', 'source')
- Use correct relationship types: 'HAS_CHUNK' (not 'CONTAINS', 'PART_OF')
- Be specific about what you'll investigate and how you'll approach it"""
                
                guidance_response = self.llm.invoke(guidance_prompt)
                print(f"\n🧠 Claude's Response to Your Guidance:")
                print(guidance_response)
                
                # Check if Claude wants to execute a query
                if "EXECUTE_QUERY:" in guidance_response:
                    # Extract the query from the response
                    query_start = guidance_response.find("EXECUTE_QUERY:") + len("EXECUTE_QUERY:")
                    query_end = guidance_response.find("\n", query_start)
                    if query_end == -1:
                        query_end = len(guidance_response)
                    
                    query_input = guidance_response[query_start:query_end].strip()
                    
                    if query_input:
                        print(f"\n🔍 Claude is executing: {query_input}")
                        results = self.execute_cypher_query(query_input)
                        
                        if "error" in results:
                            print(f"❌ Query failed: {results['error']}")
                            # Let Claude analyze the error and provide a corrected query
                            error_prompt = f"""The query failed with this error: {results['error']}

Original guidance: {user_guidance}
Original question: {question}
Failed query: {query_input}

Please analyze the error and provide a corrected Cypher query. Common issues:
1. Property names that don't exist (use 'name' instead of 'id', 'title', 'source')
2. Relationship types that don't exist (use 'HAS_CHUNK' instead of 'CONTAINS', 'PART_OF')
3. Syntax errors in the query

Provide a corrected query in this format: EXECUTE_QUERY: [corrected cypher query]"""
                            error_response = self.llm.invoke(error_prompt)
                            print(f"\n🔍 Claude's Analysis of the Error:")
                            print(error_response)
                            
                            # Check if Claude provided a corrected query
                            if "EXECUTE_QUERY:" in error_response:
                                # Extract the corrected query
                                query_start = error_response.find("EXECUTE_QUERY:") + len("EXECUTE_QUERY:")
                                query_end = error_response.find("\n", query_start)
                                if query_end == -1:
                                    query_end = len(error_response)
                                
                                corrected_query = error_response[query_start:query_end].strip()
                                
                                if corrected_query:
                                    print(f"\n🔍 Claude is executing corrected query: {corrected_query}")
                                    corrected_results = self.execute_cypher_query(corrected_query)
                                    
                                    if "error" in corrected_results:
                                        print(f"❌ Corrected query also failed: {corrected_results['error']}")
                                    else:
                                        print(f"✅ Corrected query executed successfully - {corrected_results['count']} results")
                                        if corrected_results['count'] > 0:
                                            print(f"📊 Sample results: {json.dumps(corrected_results['results'][:3], indent=2)}")
                                        
                                        # Let Claude analyze the corrected results
                                        corrected_analysis_prompt = f"""You executed this corrected query: {corrected_query}

Results: {json.dumps(corrected_results, indent=2)}

Original guidance: {user_guidance}
Original question: {question}

Please analyze these results and provide insights about what they reveal."""
                                        
                                        corrected_analysis = self.llm.invoke(corrected_analysis_prompt)
                                        print(f"\n🔍 Claude's Analysis of Corrected Query Results:")
                                        print(corrected_analysis)
                        else:
                            print(f"✅ Query executed successfully - {results['count']} results")
                            if results['count'] > 0:
                                print(f"📊 Sample results: {json.dumps(results['results'][:3], indent=2)}")
                            
                            # Let Claude analyze the results automatically
                            follow_up_prompt = f"""The user provided this guidance: {user_guidance}

You executed this query: {query_input}

Results: {json.dumps(results, indent=2)}

Original question: {question}

Please analyze these results and provide insights about what they reveal. Answer the user's question based on these results."""
                            
                            follow_up_response = self.llm.invoke(follow_up_prompt)
                            print(f"\n🔍 Claude's Analysis of Query Results:")
                            print(follow_up_response)
                else:
                    # No query needed, Claude provided a natural language response
                    print(f"\n💬 Claude's natural language response (no query needed)")
            
            conversation_history.append({
                "question": question,
                "response": claude_response,
                "user_guidance": user_guidance if user_guidance else None,
                "guidance_response": guidance_response if user_guidance else None,
                "timestamp": datetime.now().isoformat()
            })
        
        print(f"\n✅ Interactive query session completed!")
        print(f"📊 Total questions asked: {len(conversation_history)}")
        
        return {
            "mode": "interactive_query",
            "questions_asked": len(conversation_history),
            "conversation": conversation_history
        }

    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()

def main():
    """Main function to run the graph database analysis."""
    analyst = GraphDBAnalyst()
    
    try:
        print("🚀 Graph Database Analyst - Claude Sonnet 4")
        print("=" * 60)
        print("This tool uses Claude Sonnet 4 to analyze your Neo4j graph database")
        print("like a true data analyst, exploring structure and identifying issues.")
        print()
        
        # Test database connection
        print("🔌 Testing database connection...")
        test_results = analyst.execute_cypher_query("MATCH (n) RETURN count(n) as total_nodes")
        if "error" in test_results:
            print(f"❌ Database connection failed: {test_results['error']}")
            return
        else:
            print(f"✅ Connected to database - {test_results['results'][0]['total_nodes']} total nodes")
        
        # Run analysis
        results = analyst.interactive_analysis()
        
        if results:
            print(f"\n📊 Analysis Complete!")
            print(f"   Queries executed: {results['total_queries']}")
            print(f"   Issues found: {results['issues_found']}")
            print(f"   Report saved to: {analyst.analysis_file}")
    
    except KeyboardInterrupt:
        print("\n\n⏹️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    finally:
        analyst.close()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Multi-step Wiki Generation Agent System

This system provides:
1. Smart topic recommendations based on knowledge graph analysis
2. Interactive topic selection with relevance scoring
3. Multi-iteration wiki generation with configurable query limits
4. Automatic cross-linking between wikis
5. High-quality content through iterative refinement
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from dotenv import load_dotenv
from .litellm_wrapper import LiteLLMChat

load_dotenv()

@dataclass
class WikiTopic:
    """Represents a potential wiki topic with relevance scoring."""
    name: str
    entity_type: str
    relevance_score: float
    evidence_count: int
    source_documents: int
    centrality_score: float
    description: str
    existing_wiki: Optional[str] = None

@dataclass
class WikiGenerationSession:
    """Tracks a wiki generation session with iteration limits."""
    topic: WikiTopic
    max_iterations: int
    breadth: int = 1  # Parallel streams
    depth: int = 10   # Sequential iterations per stream
    current_iteration: int = 0
    queries_made: List[str] = None
    knowledge_gathered: Dict[str, Any] = None
    current_draft: str = ""
    
    def __post_init__(self):
        if self.queries_made is None:
            self.queries_made = []
        if self.knowledge_gathered is None:
            self.knowledge_gathered = {}

class WikiGenerationAgent:
    """
    Multi-step agent that builds high-quality wikis through iterative knowledge graph exploration.
    """
    
    def __init__(self):
        self.llm = LiteLLMChat()
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))
        )
        self.wiki_folder = Path("Wikis")
        self.wiki_folder.mkdir(exist_ok=True)
        
    def close(self):
        """Close database connection."""
        if self.driver:
            self.driver.close()
    
    def get_wiki_recommendations(self, limit: int = 20) -> List[WikiTopic]:
        """
        Analyze knowledge graph to recommend high-quality wiki topics.
        Uses glossary for deduplication and LLM for quality filtering.
        Returns topics ordered by relevance and potential.
        """
        print("🔍 Analyzing knowledge graph for wiki recommendations...")
        
        # Load glossary for deduplication
        glossary = self._load_glossary()
        
        with self.driver.session() as session:
            # Get more candidates initially for better filtering
            result = session.run("""
            MATCH (e)
            WHERE any(l IN labels(e) WHERE l IN [
                'Brand', 'Installment', 'Concept', 'Insight', 'Feature', 
                'Audience_type', 'Content', 'Player_segment', 'Gameplay_feature',
                'Research_study', 'Event', 'Market', 'Time_period'
            ])
            
            // Get evidence metrics
            OPTIONAL MATCH (e)-[:MENTIONED_IN_CHUNK]->(c:Chunk)
            WITH e, count(c) AS mentions
            
            // Get relationship centrality
            OPTIONAL MATCH (e)-[r1]-(n1)
            WHERE NOT 'Chunk' IN labels(n1)
            WITH e, mentions, count(r1) AS connections
            
            // Get knowledge depth (unique facts)
            OPTIONAL MATCH (e)-[:MENTIONED_IN_CHUNK]->(c2:Chunk)
            WITH e, mentions, connections, count(DISTINCT c2.text) AS unique_chunks
            
            // Calculate relevance score
            WITH e, mentions, connections, unique_chunks,
                 (mentions * 0.4 + connections * 0.4 + unique_chunks * 0.2) AS relevance_score
            
            WHERE mentions >= 3
            
            WITH e, mentions, connections, unique_chunks, relevance_score, e.name AS name
            
            RETURN 
                name,
                labels(e)[0] AS entity_type,
                relevance_score,
                mentions AS evidence_count,
                1 AS source_documents,  // Simplified for now
                connections AS centrality_score
            ORDER BY relevance_score DESC
            LIMIT $limit
            """, limit=limit * 2)  # Get more candidates for filtering
            
            raw_topics = []
            for row in result:
                raw_topics.append({
                    "name": row["name"],
                    "entity_type": row["entity_type"],
                    "relevance_score": row["relevance_score"],
                    "evidence_count": row["evidence_count"],
                    "source_documents": row["source_documents"],
                    "centrality_score": row["centrality_score"]
                })
        
        # Apply glossary-based deduplication
        deduplicated_topics = self._deduplicate_with_glossary(raw_topics, glossary)
        
        # Apply LLM-based quality filtering
        filtered_topics = self._filter_topics_with_llm(deduplicated_topics)
        
        # Convert to WikiTopic objects
        topics = []
        for topic_data in filtered_topics[:limit]:  # Limit to requested number
            # Check if wiki file exists and get its metadata
            wiki_info = self._check_existing_wiki(topic_data["name"])
            
            # Generate description based on entity type and metrics
            description = self._generate_topic_description(
                topic_data["name"], topic_data["entity_type"], 
                topic_data["evidence_count"], topic_data["source_documents"]
            )
            
            topic = WikiTopic(
                name=topic_data["name"],
                entity_type=topic_data["entity_type"],
                relevance_score=round(topic_data["relevance_score"], 2),
                evidence_count=topic_data["evidence_count"],
                source_documents=topic_data["source_documents"],
                centrality_score=topic_data["centrality_score"],
                description=description,
                existing_wiki=wiki_info["filepath"] if wiki_info["exists"] else None
            )
            topics.append(topic)
        
        print(f"✅ Found {len(topics)} high-quality wiki topics (filtered from {len(raw_topics)} candidates)")
        return topics
    
    def _load_glossary(self) -> Dict[str, List[str]]:
        """Load the glossary for entity deduplication."""
        try:
            with open("glossary.json", 'r', encoding='utf-8') as f:
                glossary_data = json.load(f)
            return glossary_data.get("synonyms", {})
        except Exception as e:
            print(f"⚠️ Could not load glossary: {e}")
            return {}
    
    def _deduplicate_with_glossary(self, topics: List[Dict], glossary: Dict[str, List[str]]) -> List[Dict]:
        """Remove duplicate topics using glossary mappings."""
        print("🔍 Applying glossary-based deduplication...")
        
        # Create reverse mapping from synonyms to canonical names
        canonical_map = {}
        for canonical, synonyms in glossary.items():
            for synonym in synonyms:
                canonical_map[synonym.lower()] = canonical
        
        # Group topics by canonical name
        canonical_groups = {}
        for topic in topics:
            name = topic["name"]
            canonical_name = canonical_map.get(name.lower(), name)
            
            if canonical_name not in canonical_groups:
                canonical_groups[canonical_name] = []
            canonical_groups[canonical_name].append(topic)
        
        # For each canonical group, keep the best topic (highest relevance score)
        deduplicated = []
        for canonical_name, group in canonical_groups.items():
            if len(group) == 1:
                # No duplicates, keep as is
                deduplicated.append(group[0])
            else:
                # Multiple variants, keep the best one
                best_topic = max(group, key=lambda t: t["relevance_score"])
                best_topic["name"] = canonical_name  # Use canonical name
                deduplicated.append(best_topic)
                print(f"   🔄 Merged {len(group)} variants of '{canonical_name}' (kept best: {best_topic['relevance_score']:.2f})")
        
        print(f"   📊 Deduplicated: {len(topics)} → {len(deduplicated)} topics")
        return deduplicated
    
    def _filter_topics_with_llm(self, topics: List[Dict]) -> List[Dict]:
        """Use LLM to filter out generic, low-value, or redundant topics."""
        print("🤖 Applying LLM-based quality filtering...")
        
        if not topics:
            return topics
        
        # Prepare topic list for LLM analysis
        topic_list = []
        for i, topic in enumerate(topics, 1):
            topic_list.append(f"{i}. {topic['name']} ({topic['entity_type']}) - Score: {topic['relevance_score']:.2f}")
        
        filter_prompt = f"""
        Analyze these potential wiki topics and identify which ones would make high-quality, valuable wiki entries.
        
        Topics to evaluate:
        {chr(10).join(topic_list)}
        
        Filtering criteria:
        1. REMOVE generic terms like "Players", "Shadows", "RED" when a more specific version exists
        2. REMOVE overly broad concepts that lack specificity
        3. REMOVE topics that are too narrow or lack sufficient content potential
        4. KEEP specific brands, games, characters, and well-defined concepts
        5. KEEP topics with high relevance scores and good evidence
        6. PRIORITIZE topics that would provide unique value to readers
        
        Return ONLY a JSON list of the topic numbers (1-based) that should be KEPT.
        Example: [1, 3, 5, 7, 9]
        
        Be selective - only keep the most valuable topics.
        """
        
        try:
            response = self.llm.invoke(filter_prompt)
            # Clean and parse JSON response
            cleaned = response.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            selected_indices = json.loads(cleaned)
            selected_indices = [i - 1 for i in selected_indices]  # Convert to 0-based
            
            filtered_topics = [topics[i] for i in selected_indices if 0 <= i < len(topics)]
            
            print(f"   📊 LLM filtered: {len(topics)} → {len(filtered_topics)} topics")
            return filtered_topics
            
        except Exception as e:
            print(f"   ⚠️ LLM filtering failed: {e}, keeping all topics")
            return topics
    
    def _generate_topic_description(self, name: str, entity_type: str, 
                                  evidence_count: int, source_docs: int) -> str:
        """Generate a human-readable description of the topic."""
        if entity_type == "Brand":
            return f"Brand with {evidence_count} mentions across {source_docs} documents"
        elif entity_type == "Game":
            return f"Game title with {evidence_count} data points from {source_docs} sources"
        elif entity_type == "Research Study":
            return f"Research study with {evidence_count} findings across {source_docs} documents"
        elif entity_type == "Person":
            return f"Person mentioned {evidence_count} times in {source_docs} documents"
        else:
            return f"{entity_type} with {evidence_count} references across {source_docs} sources"
    
    def _check_existing_wiki(self, entity_name: str) -> Dict:
        """
        Check if a wiki already exists for an entity and get its metadata.
        Returns dict with: exists, filepath, last_updated, needs_update
        """
        # Create safe filename
        safe_name = re.sub(r'[^a-zA-Z0-9\s\-_]', '', entity_name)
        safe_name = re.sub(r'\s+', '_', safe_name)
        filename = f"{safe_name}.md"
        filepath = self.wiki_folder / filename
        
        if not filepath.exists():
            return {
                "exists": False,
                "filepath": None,
                "last_updated": None,
                "needs_update": False
            }
        
        try:
            # Read the wiki file to get metadata
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract YAML front matter
            metadata = self._extract_wiki_metadata(content)
            
            # Check if it needs updating based on last update time
            needs_update = self._check_if_update_needed(metadata)
            
            return {
                "exists": True,
                "filepath": str(filepath),
                "last_updated": metadata.get("last_updated"),
                "needs_update": needs_update,
                "metadata": metadata
            }
            
        except Exception as e:
            print(f"      ⚠️ Error reading wiki metadata for {entity_name}: {e}")
            return {
                "exists": True,
                "filepath": str(filepath),
                "last_updated": "Unknown",
                "needs_update": True,  # Assume it needs updating if we can't read it
                "metadata": {}
            }
    
    def _extract_wiki_metadata(self, content: str) -> Dict:
        """Extract YAML front matter from wiki content."""
        metadata = {}
        
        # Look for YAML front matter between --- markers
        if content.startswith('---'):
            try:
                # Find the end of YAML front matter
                end_marker = content.find('---', 3)
                if end_marker != -1:
                    yaml_content = content[3:end_marker].strip()
                    
                    # Parse YAML-like content (simple key-value parsing)
                    for line in yaml_content.split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            metadata[key] = value
            except:
                pass
        
        return metadata
    
    def _check_if_update_needed(self, metadata: Dict) -> bool:
        """
        Determine if a wiki needs updating based on its metadata.
        Returns True if update is recommended.
        """
        # Check if we have a last_updated timestamp
        last_updated = metadata.get("last_updated")
        if not last_updated:
            return True  # No timestamp, assume it needs updating
        
        try:
            # Try to parse the timestamp
            from datetime import datetime
            import re
            
            # Handle different timestamp formats
            if re.match(r'\d{4}-\d{2}-\d{2}', last_updated):
                # Date format: YYYY-MM-DD
                last_date = datetime.strptime(last_updated, '%Y-%m-%d')
            elif re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', last_updated):
                # ISO format: YYYY-MM-DDTHH:MM:SS
                last_date = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
            else:
                # Unknown format, assume it needs updating
                return True
            
            # Check if it's been more than 30 days since last update
            days_since_update = (datetime.now() - last_date).days
            return days_since_update > 30
            
        except:
            # If we can't parse the timestamp, assume it needs updating
            return True
    
    def interactive_topic_selection(self, topics: List[WikiTopic]) -> Optional[List[WikiTopic]]:
        """
        Present topics to user for selection with relevance scoring.
        Returns a list of selected topics or None if user wants to exit.
        Supports multiple selections separated by commas (e.g., "1,2,3" or "1, 2, 3").
        """
        print("\n📚 Wiki Topic Recommendations")
        print("=" * 60)
        print("Topics ordered by relevance and potential quality:")
        print()
        
        for i, topic in enumerate(topics, 1):
            # Get detailed wiki status
            wiki_info = self._check_existing_wiki(topic.name)
            
            if wiki_info["exists"]:
                if wiki_info["needs_update"]:
                    status = "🔄 UPDATE NEEDED"
                    update_info = f"Last updated: {wiki_info['last_updated'] or 'Unknown'} (30+ days ago)"
                else:
                    status = "✅ UP TO DATE"
                    update_info = f"Last updated: {wiki_info['last_updated'] or 'Unknown'}"
            else:
                status = "🆕 NEW"
                update_info = "No existing wiki"
            
            print(f"{i:2d}. {status} | {topic.name}")
            print(f"    📊 Type: {topic.entity_type}")
            print(f"    🎯 Relevance: {topic.relevance_score}/10")
            print(f"    📈 Evidence: {topic.evidence_count} mentions, {topic.source_documents} docs")
            print(f"    🔗 Connections: {topic.centrality_score}")
            print(f"    📝 {topic.description}")
            print(f"    📅 {update_info}")
            print()
        
        while True:
            try:
                choice = input("Select topic number(s) separated by commas (e.g., '1,2,3' or '1, 2, 3') or 'q' to quit: ").strip()
                if choice.lower() == 'q':
                    return None
                
                # Parse comma-separated numbers
                selected_numbers = []
                for num_str in choice.split(','):
                    num_str = num_str.strip()
                    if num_str:
                        try:
                            num = int(num_str)
                            if 1 <= num <= len(topics):
                                selected_numbers.append(num)
                            else:
                                print(f"❌ Number {num} is out of range (1-{len(topics)}).")
                                break
                        except ValueError:
                            print(f"❌ '{num_str}' is not a valid number.")
                            break
                else:
                    # If we didn't break out of the loop, all numbers were valid
                    if selected_numbers:
                        selected_topics = [topics[num - 1] for num in selected_numbers]
                        print(f"\n✅ Selected {len(selected_topics)} topic(s):")
                        for topic in selected_topics:
                            print(f"   • {topic.name} ({topic.entity_type})")
                        return selected_topics
                    else:
                        print("❌ No valid selections made.")
                        
            except Exception as e:
                print(f"❌ Error parsing selection: {e}")
                print("❌ Please enter numbers separated by commas (e.g., '1,2,3') or 'q' to quit.")
    
    def get_iteration_limit(self) -> dict:
        """Get iteration configuration with parallel/sequential options."""
        while True:
            try:
                print("\n🔧 Wiki Generation Configuration:")
                print("Configure how thorough the wiki generation should be:")
                print("- Breadth: How many parallel research streams (1-5)")
                print("- Depth: How many sequential iterations per stream (1-10)")
                print("- Total steps will be breadth × depth (max 20 total)")
                
                breadth = int(input("Enter breadth (parallel streams, 1-5): ").strip())
                if breadth < 1 or breadth > 5:
                    print("❌ Breadth must be between 1 and 5")
                    continue
                
                depth = int(input("Enter depth (iterations per stream, 1-10): ").strip())  
                if depth < 1 or depth > 10:
                    print("❌ Depth must be between 1 and 10")
                    continue
                
                total_steps = breadth * depth
                if total_steps > 20:
                    print(f"❌ Total steps ({total_steps}) exceeds maximum of 20. Please reduce breadth or depth.")
                    continue
                
                print(f"✅ Configuration: {breadth} parallel streams × {depth} iterations = {total_steps} total steps")
                return {"breadth": breadth, "depth": depth, "total": total_steps}
                
            except ValueError:
                print("❌ Please enter valid numbers.")
    
    def generate_wiki(self, topic: WikiTopic, config: dict) -> str:
        """Generate a comprehensive wiki using multi-step agent approach with parallel/sequential execution."""
        print(f"\n🎯 Starting wiki generation for: {topic.name}")
        print(f"   📊 Config: {config['breadth']} parallel streams × {config['depth']} iterations = {config['total']} total steps")
        
        # Create generation session
        session = WikiGenerationSession(
            topic=topic,
            max_iterations=config['total'],
            current_iteration=0,
            queries_made=[],
            knowledge_gathered={},
            current_draft=""
        )
        
        # Step 1: Initial research and outline
        print("\n📋 Step 1: Initial research and outline...")
        self._initial_research(session)
        
        # Step 2: Iterative knowledge gathering
        print(f"\n🔍 Step 2: Iterative knowledge gathering ({config['total']} iterations)...")
        for i in range(config['total']):
            session.current_iteration = i + 1
            print(f"\n🔄 Iteration {i + 1}/{config['total']}")
            
            if not self._iterate_knowledge_gathering(session):
                print("   ⏸️ No more questions needed, moving to finalization")
                break
        
        # Step 3: Final synthesis and formatting
        print("\n📝 Step 3: Final synthesis and formatting...")
        final_wiki = self._finalize_wiki(session)
        
        return final_wiki
    
    def _initial_research(self, session: WikiGenerationSession):
        """Perform initial research to understand the topic and create an outline."""
        topic = session.topic
        
        # Get comprehensive initial evidence
        evidence = self._gather_comprehensive_evidence(topic)
        session.knowledge_gathered["initial_evidence"] = evidence
        
        # Generate initial outline
        outline_prompt = f"""
        Based on the evidence below, create a comprehensive wiki outline for {topic.name} ({topic.entity_type}).
        
        Evidence summary:
        - {topic.evidence_count} mentions across {topic.source_documents} documents
        - Entity type: {topic.entity_type}
        
        Create a structured outline with main sections and subsections.
        Focus on what can be substantiated with the available evidence.
        
        Return ONLY the outline structure, no content.
        """
        
        outline = self.llm.invoke(outline_prompt)
        session.knowledge_gathered["outline"] = outline
        session.current_draft = outline
        
        print(f"   ✅ Initial outline created")
        print(f"   📊 Evidence gathered: {len(evidence.get('chunks', []))} chunks, {len(evidence.get('facts', []))} facts")
    
    def _iterate_knowledge_gathering(self, session: WikiGenerationSession) -> bool:
        """
        Perform one iteration of knowledge gathering.
        Returns True if more questions are needed, False if complete.
        """
        topic = session.topic
        
        # Generate strategic questions based on current knowledge
        questions = self._generate_strategic_questions(session)
        
        if not questions:
            return False
        
        # Select the best question to ask
        best_question = questions[0]
        print(f"   🤔 Question: {best_question}")
        
        # Query the knowledge graph
        answer = self._query_knowledge_graph(topic, best_question)
        session.queries_made.append(f"Q: {best_question}\nA: {answer}")
        
        # Update knowledge and draft
        session.knowledge_gathered[f"iteration_{session.current_iteration}"] = {
            "question": best_question,
            "answer": answer
        }
        
        # Update the draft with new information
        self._update_draft_with_new_knowledge(session, best_question, answer)
        
        print(f"   ✅ Answer gathered and integrated")
        return True
    
    def _generate_strategic_questions(self, session: WikiGenerationSession) -> List[str]:
        """Generate strategic questions to fill gaps in current knowledge."""
        topic = session.topic
        current_knowledge = session.knowledge_gathered
        
        # Analyze what we know and what's missing
        analysis_prompt = f"""
        Analyze the current wiki draft and knowledge for {topic.name} ({topic.entity_type}).
        
        Current draft:
        {session.current_draft[:1000]}...
        
        Knowledge gathered so far:
        {json.dumps(list(current_knowledge.keys()), indent=2)}
        
        Based on this, what are the 3 most important questions to ask next?
        Focus on:
        1. Missing critical information
        2. Areas that need more detail
        3. Connections to other entities
        
        Return ONLY a JSON list of questions, no explanations.
        """
        
        try:
            response = self.llm.invoke(analysis_prompt)
            # Clean response and parse JSON
            cleaned = response.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            
            questions = json.loads(cleaned)
            return questions[:3]  # Limit to top 3
        except:
            # Fallback questions
            return [
                f"What are the key achievements or milestones for {topic.name}?",
                f"How does {topic.name} relate to other entities in the knowledge graph?",
                f"What are the most recent developments or updates about {topic.name}?"
            ]
    
    def _query_knowledge_graph(self, topic: WikiTopic, question: str) -> str:
        """Query the knowledge graph with a specific question."""
        with self.driver.session() as session:
            # Use semantic search to find relevant information
            result = session.run("""
            MATCH (e {name: $entity_name})
            MATCH (e)-[:MENTIONED_IN_CHUNK]->(c:Chunk)
            WITH e, c, c.text AS chunk_text
            
            // Find chunks that might answer the question
            WHERE chunk_text CONTAINS $entity_name
            
            RETURN c.text AS text, c.chunk_id AS id
            ORDER BY size(c.text) DESC
            LIMIT 5
            """, entity_name=topic.name, question=question)
            
            chunks = [row["text"] for row in result]
            
            if not chunks:
                return "No specific information found in the knowledge graph."
            
            # Use LLM to synthesize answer from chunks
            synthesis_prompt = f"""
            Question: {question}
            
            Context from knowledge graph:
            {chr(10).join(chunks)}
            
            Based on this context, provide a clear, factual answer to the question.
            If the information is not available, say so clearly.
            Keep the answer concise but informative.
            """
            
            return self.llm.invoke(synthesis_prompt)
    
    def _update_draft_with_new_knowledge(self, session: WikiGenerationSession, 
                                       question: str, answer: str):
        """Update the current draft with newly gathered knowledge."""
        update_prompt = f"""
        Update the wiki draft for {session.topic.name} with new information.
        
        Current draft:
        {session.current_draft}
        
        New information to integrate:
        Q: {question}
        A: {answer}
        
        Integrate this information into the appropriate sections of the draft.
        Maintain the existing structure and flow.
        Return the updated draft.
        """
        
        session.current_draft = self.llm.invoke(update_prompt)
    
    def _finalize_wiki(self, session: WikiGenerationSession) -> str:
        """Finalize the wiki with proper formatting and cross-linking."""
        topic = session.topic
        
        # Final synthesis
        final_prompt = f"""
        Create the final, polished wiki content for {topic.name} ({topic.entity_type}).
        
        Current draft:
        {session.current_draft}
        
        Requirements:
        1. Use proper Markdown formatting
        2. Start with a main heading (# Title)
        3. Organize information logically
        4. Make it engaging and readable
        5. Include source citations where appropriate
        6. Do NOT include YAML front matter - that will be added separately
        
        Return only the markdown content (no YAML front matter).
        """
        
        final_wiki = self.llm.invoke(final_prompt)
        
        # Add smart wiki-to-wiki linking
        final_wiki = self._add_smart_wiki_links(final_wiki)
        
        # Add source document references
        final_wiki = self._add_source_document_references(final_wiki, topic)
        
        # Add related research links
        related_research = self._generate_related_research_links(topic)
        if related_research:
            final_wiki += f"\n\n## 🔗 Related Research\n\n{related_research}"
        
        # Generate intelligent tags
        intelligent_tags = self._generate_intelligent_tags(topic)
        
        # Add YAML front matter with proper timestamps and intelligent tags
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d")
        
        # Combine base tags with intelligent tags
        all_tags = intelligent_tags
        
        front_matter = f"""---
title: "{topic.name}"
categories: ["{topic.entity_type}"]
tags: {all_tags}
description: "Comprehensive wiki for {topic.name} ({topic.entity_type}) generated from knowledge graph data"
date: {current_time}
lastmod: {current_time}
---

"""
        
        return front_matter + final_wiki
    
    def _add_smart_wiki_links(self, wiki_content: str) -> str:
        """Add smart wiki-to-wiki linking by detecting entity names and linking to existing wikis."""
        # Get all existing wiki filenames (without extension)
        existing_wikis = set()
        for wiki_file in self.wiki_folder.glob("*.md"):
            if wiki_file.name != "index.md":
                existing_wikis.add(wiki_file.stem)
        
        # Use LLM to identify entity names in the text that should be linked
        link_prompt = f"""
        Analyze the following wiki content and identify entity names that should be linked to other wikis.
        
        Wiki content:
        {wiki_content[:2000]}...
        
        Available wiki files (use exact names):
        {sorted(list(existing_wikis))[:20]}  # Show first 20 for context
        
        Rules:
        1. Only identify entity names that have corresponding wiki files in the available list
        2. Focus on proper nouns, brand names, game titles, people, companies, etc.
        3. Be conservative - only link obvious entity references
        4. Return ONLY a JSON list of entity names to link, no explanations
        
        Example: ["Assassins_Creed", "Ubisoft", "Valhalla"]
        """
        
        try:
            response = self.llm.invoke(link_prompt)
            # Clean and parse JSON response
            cleaned = response.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            entities_to_link = json.loads(cleaned)
            
            # Add wiki links for identified entities
            import re
            for entity in entities_to_link:
                if entity in existing_wikis:
                    # Create wiki link format: [[Entity_Name]]
                    wiki_link = f"[[{entity}]]"
                    
                    # Create a more flexible pattern that handles common variations
                    entity_variations = [
                        entity,  # Exact match
                        entity.replace('_', ' '),  # Replace underscores with spaces
                        entity.replace('_', "'"),  # Replace underscores with apostrophes
                        entity.replace('_', '-'),  # Replace underscores with hyphens
                        # Handle singular/plural variations
                        entity.replace('s_', "'s "),  # Assassins_Creed -> Assassin's Creed
                        entity.replace('s ', "'s "),  # Assassins Creed -> Assassin's Creed
                    ]
                    
                    # Try each variation and stop at the first match to avoid multiple replacements
                    for variation in entity_variations:
                        # Create a pattern that matches the variation but NOT if it's already in brackets
                        # This prevents creating nested brackets like [[[[Entity]]]]
                        pattern = re.compile(rf'(?<!\[\[)(?<!\[)\b{re.escape(variation)}\b(?!\])(?!\]\])', re.IGNORECASE)
                        if pattern.search(wiki_content):
                            wiki_content = pattern.sub(wiki_link, wiki_content)
                            break  # Stop after first successful replacement
            
            return wiki_content
            
        except Exception as e:
            print(f"   ⚠️ Error in smart wiki linking: {e}")
            return wiki_content
    
    def _generate_intelligent_tags(self, topic: WikiTopic) -> List[str]:
        """Generate intelligent tags using LLM analysis of top referenced nodes."""
        try:
            # Get top 50 referenced nodes from the knowledge graph
            with self.driver.session() as session:
                result = session.run("""
                MATCH (e {name: $entity_name})
                MATCH (e)-[r]-(n)
                WHERE NOT 'Chunk' IN labels(n) AND NOT 'Document' IN labels(n)
                
                WITH n, count(r) AS connection_count
                ORDER BY connection_count DESC
                LIMIT 50
                
                RETURN n.name AS node_name, labels(n)[0] AS node_type, connection_count
                """, entity_name=topic.name)
                
                referenced_nodes = []
                for row in result:
                    referenced_nodes.append({
                        "name": row["node_name"],
                        "type": row["node_type"],
                        "connections": row["connection_count"]
                    })
            
            if not referenced_nodes:
                # Fallback to basic tags based on entity type and name
                fallback_tags = [topic.entity_type.lower().replace(' ', '-')]
                if "assassin" in topic.name.lower() or "creed" in topic.name.lower():
                    fallback_tags.extend(["assassins-creed", "gaming"])
                elif "ubisoft" in topic.name.lower():
                    fallback_tags.extend(["ubisoft", "gaming"])
                elif "naoe" in topic.name.lower():
                    fallback_tags.extend(["naoe", "assassins-creed", "character"])
                return fallback_tags[:5]
            
            # Use LLM to select the 5 most relevant tags
            tag_prompt = f"""
            Analyze the following referenced nodes from a knowledge graph and select the 5 most relevant tags for a wiki about "{topic.name}" ({topic.entity_type}).
            
            Referenced nodes (top 50 by connection count):
            {json.dumps(referenced_nodes[:20], indent=2)}
            
            Rules:
            1. Select 5 nodes that are most relevant and meaningful as tags
            2. Prefer nodes that are closely related to the main topic
            3. Avoid generic or overly broad terms
            4. Focus on entities, concepts, or themes that add value as tags
            5. Return ONLY a JSON list of 5 tag names, no explanations
            6. Use lowercase with hyphens for multi-word tags
            
            Example: ["ubisoft", "stealth-gameplay", "historical-fiction", "open-world", "assassins-creed"]
            """
            
            response = self.llm.invoke(tag_prompt)
            # Clean and parse JSON response
            cleaned = response.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            try:
                intelligent_tags = json.loads(cleaned)
                if isinstance(intelligent_tags, list) and len(intelligent_tags) > 0:
                    return intelligent_tags[:5]  # Ensure we only return 5 tags
                else:
                    raise ValueError("Invalid JSON format")
            except (json.JSONDecodeError, ValueError):
                # Fallback to basic tags if JSON parsing fails
                fallback_tags = [topic.entity_type.lower().replace(' ', '-')]
                if "assassin" in topic.name.lower() or "creed" in topic.name.lower():
                    fallback_tags.extend(["assassins-creed", "gaming"])
                elif "ubisoft" in topic.name.lower():
                    fallback_tags.extend(["ubisoft", "gaming"])
                elif "naoe" in topic.name.lower():
                    fallback_tags.extend(["naoe", "assassins-creed", "character"])
                return fallback_tags[:5]
            
        except Exception as e:
            print(f"   ⚠️ Error generating intelligent tags: {e}")
            # Fallback to basic tags based on entity type and name
            fallback_tags = [topic.entity_type.lower().replace(' ', '-')]
            if "assassin" in topic.name.lower() or "creed" in topic.name.lower():
                fallback_tags.extend(["assassins-creed", "gaming"])
            elif "ubisoft" in topic.name.lower():
                fallback_tags.extend(["ubisoft", "gaming"])
            elif "naoe" in topic.name.lower():
                fallback_tags.extend(["naoe", "assassins-creed", "character"])
            return fallback_tags[:5]
    
    def _add_source_wiki_to_graph(self, source_wiki_name: str, document_hash: str):
        """Add source wiki name to the graph for linking purposes."""
        try:
            with self.driver.session() as session:
                # Create a SourceWiki node and link it to the document
                session.run("""
                MATCH (d:Document {source_id: $doc_hash})
                MERGE (sw:SourceWiki {name: $wiki_name})
                MERGE (sw)-[:DOCUMENTED_IN]->(d)
                SET sw.wiki_type = 'Source Document',
                    sw.created_at = datetime()
                """, doc_hash=document_hash, wiki_name=source_wiki_name)
                
                print(f"      🔗 Added source wiki '{source_wiki_name}' to graph")
        except Exception as e:
            print(f"      ⚠️ Error adding source wiki to graph: {e}")
    
    def _add_source_document_references(self, wiki_content: str, topic: WikiTopic) -> str:
        """Add source document references section to wiki content."""
        try:
            # Find source documents that mention this entity
            with self.driver.session() as session:
                result = session.run("""
                MATCH (e {name: $entity_name})
                MATCH (e)-[:MENTIONED_IN_CHUNK]->(c:Chunk)-[:HAS_CHUNK]-(d:Document)
                RETURN DISTINCT d.name AS document_name
                ORDER BY d.name
                """, entity_name=topic.name)
                
                source_documents = [row["document_name"] for row in result if row["document_name"]]
            
            if not source_documents:
                return wiki_content
            
            # Add source references section
            source_section = f"""

## 📚 Source References

The following source documents were referenced in this wiki:

"""
            
            for doc_name in source_documents:
                # Create a link to the existing Source_ wiki if it exists
                # Remove file extension and create safe name
                doc_base_name = doc_name.split('.')[0]  # Remove extension
                safe_source_name = doc_base_name.replace('.', '_').replace(' ', '_')
                source_wiki_name = f"Source_{safe_source_name}"
                
                # Check if the Source_ wiki exists
                source_wiki_path = self.wiki_folder / f"{source_wiki_name}.md"
                if source_wiki_path.exists():
                    source_section += f"- [[{source_wiki_name}]]\n"
                else:
                    # Try alternative naming patterns
                    # Pattern 1: Direct match with spaces and special chars
                    alt_source_name = f"Source_{doc_base_name}"
                    alt_source_path = self.wiki_folder / f"{alt_source_name}.md"
                    if alt_source_path.exists():
                        source_section += f"- [[{alt_source_name}]]\n"
                    else:
                        # If no Source_ wiki exists, just show the filename
                        source_section += f"- {doc_name}\n"
            
            # Append source section to wiki content
            return wiki_content + source_section
            
        except Exception as e:
            print(f"   ⚠️ Error adding source document references: {e}")
            return wiki_content
    
    def _generate_related_research_links(self, topic: WikiTopic) -> str:
        """Generate links to related research wikis based on the topic."""
        try:
            # Find research wikis that might be related to this topic
            research_folder = self.wiki_folder / "Research"
            if not research_folder.exists():
                return "- [[Research Index]]\n- [[Knowledge Graph Query Interface]]"
            
            related_links = []
            
            # Look for research wikis that mention this entity or related entities
            for research_file in research_folder.glob("*.md"):
                if research_file.name != "index.md":
                    try:
                        with open(research_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Check if this research wiki is related to our topic
                        if (topic.name.lower() in content.lower() or 
                            any(keyword in content.lower() for keyword in topic.name.lower().split())):
                            related_links.append(f"- [[{research_file.stem}]]")
                    except Exception:
                        continue
            
            # If no related research found, show generic links
            if not related_links:
                return "- [[Research Index]]\n- [[Knowledge Graph Query Interface]]"
            
            # Limit to top 5 related research wikis
            return "\n".join(related_links[:5])
            
        except Exception as e:
            print(f"   ⚠️ Error generating related research links: {e}")
            return "- [[Research Index]]\n- [[Knowledge Graph Query Interface]]"
    
    def _generate_related_research_links_from_question(self, research_question: str) -> str:
        """Generate links to related research wikis based on the research question."""
        try:
            # Find research wikis that might be related to this question
            research_folder = self.wiki_folder / "Research"
            if not research_folder.exists():
                return "- [[Research Index]]\n- [[Knowledge Graph Query Interface]]"
            
            related_links = []
            
            # Extract key terms from the research question
            question_keywords = [word.lower() for word in research_question.split() 
                               if len(word) > 3 and word.lower() not in ['what', 'how', 'why', 'when', 'where', 'which', 'who']]
            
            # Look for research wikis that mention these keywords
            for research_file in research_folder.glob("*.md"):
                if research_file.name != "index.md":
                    try:
                        with open(research_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Check if this research wiki is related to our question
                        if any(keyword in content.lower() for keyword in question_keywords):
                            related_links.append(f"- [[{research_file.stem}]]")
                    except Exception:
                        continue
            
            # If no related research found, show generic links
            if not related_links:
                return "- [[Research Index]]\n- [[Knowledge Graph Query Interface]]"
            
            # Limit to top 5 related research wikis
            return "\n".join(related_links[:5])
            
        except Exception as e:
            print(f"   ⚠️ Error generating related research links: {e}")
            return "- [[Research Index]]\n- [[Knowledge Graph Query Interface]]"
    
    def _generate_research_tags(self, research_question: str) -> List[str]:
        """Generate intelligent tags for research wikis based on the research question."""
        try:
            # Extract key terms from the research question
            question_keywords = [word.lower() for word in research_question.split() 
                               if len(word) > 3 and word.lower() not in ['what', 'how', 'why', 'when', 'where', 'which', 'who', 'critical', 'response', 'analysis']]
            
            # Use LLM to generate relevant tags based on the research question
            tag_prompt = f"""
            Analyze the following research question and generate 5 relevant tags for a research wiki:
            
            Research Question: "{research_question}"
            
            Key terms: {question_keywords}
            
            Rules:
            1. Generate 5 tags that are relevant to the research topic
            2. Use lowercase with hyphens for multi-word tags
            3. Focus on the main entities, concepts, or themes being researched
            4. Avoid generic terms like "research" or "analysis"
            5. Return ONLY a JSON list of 5 tag names, no explanations
            
            Example: ["naoe", "assassins-creed", "character-analysis", "gameplay", "shadows"]
            """
            
            response = self.llm.invoke(tag_prompt)
            # Clean and parse JSON response
            cleaned = response.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            try:
                intelligent_tags = json.loads(cleaned)
                if isinstance(intelligent_tags, list) and len(intelligent_tags) > 0:
                    return intelligent_tags[:5]  # Ensure we only return 5 tags
                else:
                    raise ValueError("Invalid JSON format")
            except (json.JSONDecodeError, ValueError):
                # Fallback to basic tags if JSON parsing fails
                fallback_tags = []
                if "naoe" in research_question.lower():
                    fallback_tags.extend(["naoe", "assassins-creed", "character"])
                elif "assassin" in research_question.lower() or "creed" in research_question.lower():
                    fallback_tags.extend(["assassins-creed", "gaming"])
                elif "ubisoft" in research_question.lower():
                    fallback_tags.extend(["ubisoft", "gaming"])
                return fallback_tags[:5]
            
        except Exception as e:
            print(f"   ⚠️ Error generating research tags: {e}")
            # Fallback to basic tags based on research question
            fallback_tags = []
            if "naoe" in research_question.lower():
                fallback_tags.extend(["naoe", "assassins-creed", "character"])
            elif "assassin" in research_question.lower() or "creed" in research_question.lower():
                fallback_tags.extend(["assassins-creed", "gaming"])
            elif "ubisoft" in research_question.lower():
                fallback_tags.extend(["ubisoft", "gaming"])
            return fallback_tags[:5]
    
    def _add_research_source_references(self, wiki_content: str, research_question: str) -> str:
        """Add source document references section to research wiki content."""
        try:
            # Find source documents that might be related to this research question
            with self.driver.session() as session:
                # Extract key terms from research question for matching
                question_keywords = [word.lower() for word in research_question.split() 
                                   if len(word) > 3 and word.lower() not in ['what', 'how', 'why', 'when', 'where', 'which', 'who', 'critical', 'response', 'analysis']]
                
                # Find documents that mention these keywords
                source_documents = set()
                for keyword in question_keywords:
                    result = session.run("""
                    MATCH (d:Document)
                    WHERE toLower(d.name) CONTAINS toLower($keyword)
                    RETURN DISTINCT d.name AS document_name
                    """, keyword=keyword)
                    
                    for row in result:
                        if row["document_name"]:
                            source_documents.add(row["document_name"])
            
            if not source_documents:
                return wiki_content
            
            # Add source references section
            source_section = f"""

## 📚 Source References

The following source documents were referenced during this research:

"""
            
            for doc_name in sorted(source_documents):
                # Create a link to the existing Source_ wiki if it exists
                # Remove file extension and create safe name
                doc_base_name = doc_name.split('.')[0]  # Remove extension
                safe_source_name = doc_base_name.replace('.', '_').replace(' ', '_')
                source_wiki_name = f"Source_{safe_source_name}"
                
                # Check if the Source_ wiki exists
                source_wiki_path = self.wiki_folder / f"{source_wiki_name}.md"
                if source_wiki_path.exists():
                    source_section += f"- [[{source_wiki_name}]]\n"
                else:
                    # Try alternative naming patterns
                    # Pattern 1: Direct match with spaces and special chars
                    alt_source_name = f"Source_{doc_base_name}"
                    alt_source_path = self.wiki_folder / f"{alt_source_name}.md"
                    if alt_source_path.exists():
                        source_section += f"- [[{alt_source_name}]]\n"
                    else:
                        # If no Source_ wiki exists, just show the filename
                        source_section += f"- {doc_name}\n"
            
            # Append source section to wiki content
            return wiki_content + source_section
            
        except Exception as e:
            print(f"   ⚠️ Error adding research source document references: {e}")
            return wiki_content
    
    def _gather_comprehensive_evidence(self, topic: WikiTopic) -> Dict:
        """Gather comprehensive evidence about the topic from the knowledge graph."""
        with self.driver.session() as session:
            # Get chunks mentioning the entity
            chunks = session.run("""
            MATCH (e {name: $entity_name})
            MATCH (e)-[:MENTIONED_IN_CHUNK]->(c:Chunk)
            RETURN c.text AS text, c.chunk_id AS id, 'Document' AS doc_name
            ORDER BY size(c.text) DESC
            LIMIT 20
            """, entity_name=topic.name).data()
            
            # Get relationships
            facts = session.run("""
            MATCH (e {name: $entity_name})
            OPTIONAL MATCH (e)-[r]->(n)
            WHERE NOT 'Chunk' IN labels(n)
            RETURN type(r) AS relationship, labels(n)[0] AS target_type, n.name AS target_name
            LIMIT 50
            """, entity_name=topic.name).data()
            
            return {
                "chunks": chunks,
                "facts": facts
            }
    
    def save_wiki(self, topic: WikiTopic, content: str) -> str:
        """Save the wiki to disk and create database record."""
        # Create filename
        safe_name = re.sub(r'[^a-zA-Z0-9\s\-_]', '', topic.name)
        safe_name = re.sub(r'\s+', '_', safe_name)
        filename = f"{safe_name}.md"
        filepath = self.wiki_folder / filename
        
        # Save file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Create database record - we'll create a simple relationship for now
        with self.driver.session() as session:
            # For now, just add a property to the entity indicating it has a wiki
            session.run("""
            MATCH (e {name: $entity_name})
            SET e.has_wiki = true,
                e.wiki_filepath = $filepath,
                e.wiki_generated_at = datetime(),
                e.wiki_last_updated = datetime(),
                e.wiki_entity_type = $entity_type,
                e.wiki_relevance_score = $relevance_score
            """, entity_name=topic.name, filepath=str(filepath), 
                 entity_type=topic.entity_type, relevance_score=topic.relevance_score)
        
        print(f"   💾 Wiki saved: {filepath}")
        
        # Update the wiki index
        self.update_wiki_index()
        
        return str(filepath)
    
    def create_cross_links(self):
        """Create cross-links between existing wikis."""
        print("\n🔗 Creating cross-links between wikis...")
        
        # Get all wikis
        wiki_files = list(self.wiki_folder.glob("*.md"))
        if len(wiki_files) < 2:
            print("   ℹ️ Need at least 2 wikis to create cross-links")
            return
        
        print(f"   📊 Found {len(wiki_files)} wikis to cross-link")
        
        # Analyze each wiki for potential links
        for wiki_file in wiki_files:
            self._add_cross_links_to_wiki(wiki_file)
        
        print("   ✅ Cross-linking complete")
    
    def _add_cross_links_to_wiki(self, wiki_file: Path):
        """Add cross-links to a specific wiki file."""
        try:
            with open(wiki_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract entity names mentioned in the wiki
            entity_names = self._extract_entity_names(content)
            
            # Find which entities have wikis
            linked_entities = []
            for entity in entity_names:
                if (self.wiki_folder / f"{entity}.md").exists():
                    linked_entities.append(entity)
            
            if not linked_entities:
                return
            
            # Add cross-links section
            links_section = "\n\n## Related Wikis\n\n"
            for entity in linked_entities:
                links_section += f"- [[{entity}]]\n"
            
            # Append if not already present
            if "## Related Wikis" not in content:
                with open(wiki_file, 'a', encoding='utf-8') as f:
                    f.write(links_section)
                
                print(f"      🔗 Added {len(linked_entities)} links to {wiki_file.name}")
        
        except Exception as e:
            print(f"      ⚠️ Error processing {wiki_file.name}: {e}")
    
    def _extract_entity_names(self, content: str) -> List[str]:
        """Extract potential entity names from wiki content."""
        # Simple extraction - look for capitalized phrases
        words = content.split()
        entities = []
        
        for i, word in enumerate(words):
            if (word[0].isupper() and len(word) > 2 and 
                not word.startswith('http') and 
                not word.startswith('#')):
                
                # Check if it's part of a multi-word entity
                entity = word
                j = i + 1
                while (j < len(words) and 
                       words[j][0].isupper() and 
                       not words[j].startswith('#')):
                    entity += " " + words[j]
                    j += 1
                
                if len(entity) > 2:
                    entities.append(entity)
        
        return list(set(entities))  # Remove duplicates
    
    def _create_test_session(self, topic: WikiTopic) -> WikiGenerationSession:
        """Create a test session for testing purposes."""
        return WikiGenerationSession(
            topic=topic,
            max_iterations=5,
            current_iteration=0,
            queries_made=[],
            knowledge_gathered={"outline": "# Test Outline\n\n## Introduction\n## Main Content"},
            current_draft="# Test Outline\n\n## Introduction\n## Main Content"
        )
    
    def update_existing_wiki(self, topic: WikiTopic, max_iterations: int) -> str:
        """
        Update an existing wiki with fresh information from the knowledge graph.
        This is similar to generate_wiki but updates the existing file.
        """
        print(f"🔄 Updating existing wiki for: {topic.name}")
        print(f"📊 Entity type: {topic.entity_type}")
        print(f"🔄 Max iterations: {max_iterations}")
        print("=" * 60)
        
        # Check if wiki exists
        wiki_info = self._check_existing_wiki(topic.name)
        if not wiki_info["exists"]:
            print(f"❌ No existing wiki found for {topic.name}")
            return self.generate_wiki(topic, max_iterations)
        
        # Read existing content to preserve some structure
        try:
            with open(wiki_info["filepath"], 'r', encoding='utf-8') as f:
                existing_content = f.read()
            print(f"   📖 Found existing wiki: {wiki_info['filepath']}")
            print(f"   📅 Last updated: {wiki_info['last_updated'] or 'Unknown'}")
        except Exception as e:
            print(f"   ⚠️ Error reading existing wiki: {e}")
            existing_content = ""
        
        session = WikiGenerationSession(
            topic=topic,
            max_iterations=max_iterations,
            current_iteration=0,
            queries_made=[],
            knowledge_gathered={"existing_wiki": existing_content},
            current_draft=existing_content
        )
        
        # Step 1: Analyze existing content and gather new evidence
        print("\n📋 Step 1: Analyzing existing content and gathering new evidence...")
        self._analyze_existing_and_research(session)
        
        # Step 2: Iterative knowledge gathering
        print(f"\n🔍 Step 2: Iterative knowledge gathering ({max_iterations} iterations)...")
        for i in range(max_iterations):
            session.current_iteration = i + 1
            print(f"\n🔄 Iteration {i + 1}/{max_iterations}")
            
            if not self._iterate_knowledge_gathering(session):
                print("   ⏸️ No more questions needed, moving to finalization")
                break
        
        # Step 3: Final synthesis and formatting
        print("\n📝 Step 3: Final synthesis and formatting...")
        final_wiki = self._finalize_wiki_update(session, existing_content)
        
        return final_wiki
    
    def _analyze_existing_and_research(self, session: WikiGenerationSession):
        """Analyze existing wiki content and gather fresh evidence."""
        topic = session.topic
        
        # Get comprehensive evidence
        evidence = self._gather_comprehensive_evidence(topic)
        session.knowledge_gathered["fresh_evidence"] = evidence
        
        # Analyze what's already covered vs. what's new
        analysis_prompt = f"""
        Analyze the existing wiki content for {topic.name} and identify areas for improvement.
        
        Existing content:
        {session.current_draft[:2000]}...
        
        Fresh evidence summary:
        - {topic.evidence_count} mentions across {topic.source_documents} documents
        - Entity type: {topic.entity_type}
        
        Identify:
        1. What information is missing or outdated
        2. What sections could be expanded
        3. What new connections or facts could be added
        
        Return a brief analysis of what needs updating.
        """
        
        analysis = self.llm.invoke(analysis_prompt)
        session.knowledge_gathered["update_analysis"] = analysis
        
        print(f"   ✅ Analysis complete")
        print(f"   📊 Fresh evidence gathered: {len(evidence.get('chunks', []))} chunks, {len(evidence.get('facts', []))} facts")
    
    def _finalize_wiki_update(self, session: WikiGenerationSession, existing_content: str) -> str:
        """Finalize the wiki update with proper formatting and update tracking."""
        topic = session.topic
        
        # Final synthesis with update context
        final_prompt = f"""
        Update the existing wiki content for {topic.name} ({topic.entity_type}) with new information.
        
        Existing content:
        {existing_content[:2000]}...
        
        New information gathered:
        {session.current_draft[:2000]}...
        
        Requirements:
        1. Preserve the best parts of the existing content
        2. Integrate new information seamlessly
        3. Update outdated information
        4. Maintain the existing structure where possible
        5. Use proper Markdown formatting
        6. Do NOT include YAML front matter - that will be added separately
        
        Return only the updated markdown content (no YAML front matter).
        """
        
        final_wiki = self.llm.invoke(final_prompt)
        
        # Add smart wiki-to-wiki linking
        final_wiki = self._add_smart_wiki_links(final_wiki)
        
        # Add source document references
        final_wiki = self._add_source_document_references(final_wiki, topic)
        
        # Generate intelligent tags
        intelligent_tags = self._generate_intelligent_tags(topic)
        
        # Add updated YAML front matter
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d")
        
        # Extract existing metadata to preserve some fields
        existing_metadata = self._extract_wiki_metadata(existing_content)
        
        # Combine base tags with intelligent tags
        all_tags = intelligent_tags
        
        front_matter = f"""---
title: "{topic.name}"
categories: ["{topic.entity_type}"]
tags: {all_tags}
description: "Updated wiki for {topic.name} ({topic.entity_type}) with latest knowledge graph data"
date: {existing_metadata.get('date', current_time)}
lastmod: {current_time}
---

"""
        
        return front_matter + final_wiki
    
    def create_research_wiki(self, research_question: str, max_iterations: int) -> str:
        """Create a research wiki using the same high-quality SemanticAgent process as main.py."""
        print(f"\n🔬 Creating research wiki for: {research_question}")
        print("🚀 Using high-quality SemanticAgent for research (same as main.py query system)")
        
        try:
            # Import and initialize the SemanticAgent (same as main.py)
            from .semantic_agent import SemanticAgent
            agent = SemanticAgent()
            
            # Step 1: Start with exploratory question to understand context and intent
            print(f"\n📋 Step 1: Exploratory question to understand research context...")
            exploratory_question = self._generate_exploratory_question(research_question)
            print(f"\n🔄 Exploratory Question: {exploratory_question}")
            
            try:
                print("   🔍 Searching knowledge graph...")
                exploratory_response = agent.run_query(exploratory_question)
                print(f"   ✅ Exploratory answer received ({len(exploratory_response)} characters)")
                
                # Store the exploratory Q&A pair
                exploratory_qa = {"question": exploratory_question, "answer": exploratory_response, "type": "exploratory"}
                research_findings = [exploratory_qa]
                conversation_context = [exploratory_qa]
                
            except Exception as e:
                print(f"   ❌ Error in exploratory query: {e}")
                # Fallback to basic context
                exploratory_qa = {"question": exploratory_question, "answer": f"Research topic: {research_question}", "type": "exploratory"}
                research_findings = [exploratory_qa]
                conversation_context = [exploratory_qa]
            
            # Step 2: Generate focused follow-up questions based on context
            print(f"\n🔍 Step 2: Generating focused follow-up questions...")
            remaining_iterations = max_iterations - 1  # -1 for exploratory question
            
            for i in range(remaining_iterations):
                print(f"\n🔄 Follow-up {i + 1}/{remaining_iterations}")
                
                # Generate follow-up question based on conversation context and research intent
                follow_up_question = self._generate_contextual_follow_up_question(research_question, conversation_context, i + 1)
                print(f"   ❓ Follow-up: {follow_up_question}")
                
                try:
                    # Query with follow-up
                    print("   🔍 Searching knowledge graph...")
                    response = agent.run_query(follow_up_question)
                    print(f"   ✅ Answer received ({len(response)} characters)")
                    
                    # Store the Q&A pair
                    qa_pair = {"question": follow_up_question, "answer": response, "type": "follow_up"}
                    research_findings.append(qa_pair)
                    conversation_context.append(qa_pair)
                    
                except Exception as e:
                    print(f"   ❌ Error querying: {e}")
                    continue
            
            # Step 3: Synthesize findings into wiki
            print(f"\n📝 Step 3: Synthesizing research findings...")
            wiki_content = self._synthesize_semantic_research_wiki(research_question, research_findings)
            
            # Clean up - SemanticAgent doesn't have close method
            try:
                if hasattr(agent, 'close'):
                    agent.close()
            except:
                pass
            
            return wiki_content
            
        except Exception as e:
            print(f"❌ Error in research wiki creation: {e}")
            # Even if there's an error, try to return the content if we have findings
            if research_findings:
                print("   🔄 Attempting to create wiki from available findings...")
                try:
                    wiki_content = self._synthesize_semantic_research_wiki(research_question, research_findings)
                    return wiki_content
                except Exception as e2:
                    print(f"   ❌ Error in synthesis: {e2}")
            
            return f"❌ Failed to create research wiki: {e}"
    
    def _find_relevant_research_data(self, research_question: str) -> dict:
        """Find entities and documents relevant to the research question."""
        with self.driver.session() as session:
            # Search for entities mentioned in the question
            question_terms = research_question.lower().split()
            
            # First, try to find entities by name
            result = session.run("""
            MATCH (e)
            WHERE NOT 'Chunk' IN labels(e) AND NOT 'Document' IN labels(e)
            AND any(term IN $terms WHERE toLower(e.name) CONTAINS term)
            
            OPTIONAL MATCH (e)-[:MENTIONED_IN_CHUNK]->(c:Chunk)-[:HAS_CHUNK]-(d:Document)
            
            RETURN DISTINCT
                e.name AS entity_name,
                labels(e)[0] AS entity_type,
                collect(DISTINCT d.name) AS related_documents
            LIMIT 20
            """, terms=question_terms)
            
            entities = []
            documents = set()
            
            for row in result:
                entities.append({
                    "name": row["entity_name"],
                    "type": row["entity_type"]
                })
                if row["related_documents"]:
                    documents.update(row["related_documents"])
            
            # Search in chunk content for more precise matches
            print(f"   🔍 Searching in chunk content for precise matches...")
            result = session.run("""
            MATCH (c:Chunk)-[:HAS_CHUNK]-(d:Document)
            WHERE any(term IN $terms WHERE toLower(c.text) CONTAINS term)
            
            RETURN DISTINCT
                d.name AS document_name,
                d.path AS document_path,
                c.text AS chunk_preview
            ORDER BY size(c.text) DESC
            LIMIT 15
            """, terms=question_terms)
            
            chunk_docs = set()
            for row in result:
                if row["document_name"]:
                    chunk_docs.add(row["document_name"])
                    print(f"   📄 Found relevant content in: {row['document_name']}")
            
            # Merge all found documents
            documents.update(chunk_docs)
            
            # Also search for any documents that might contain the terms in their names
            if not documents:
                print(f"   🔍 Searching for documents containing search terms in names...")
                result = session.run("""
                MATCH (d:Document)
                WHERE any(term IN $terms WHERE toLower(d.name) CONTAINS term)
                RETURN DISTINCT d.name AS document_name
                LIMIT 10
                """, terms=question_terms)
                
                for row in result:
                    if row["document_name"]:
                        documents.add(row["document_name"])
                        print(f"   📁 Found document: {row['document_name']}")
            
            print(f"   📊 Found {len(entities)} entities and {len(documents)} documents")
            
            return {
                "entities": entities,
                "documents": list(documents),
                "terms": question_terms
            }
    
    def _research_query_with_validation(self, question: str, relevant_data: dict) -> dict:
        """Query knowledge graph with strict validation for research."""
        findings = {"content": "", "sources": [], "entities_referenced": []}
        
        with self.driver.session() as session:
            # Query relevant entities if they exist
            if relevant_data["entities"]:
                for entity in relevant_data["entities"][:5]:  # Limit to top 5 entities
                    result = session.run("""
                    MATCH (e {name: $entity_name})-[:MENTIONED_IN_CHUNK]->(c:Chunk)-[:HAS_CHUNK]-(d:Document)
                    WHERE any(term IN $question_terms WHERE toLower(c.text) CONTAINS term)
                    
                    RETURN 
                        c.text AS chunk_text,
                        d.name AS source_document,
                        d.path AS source_path
                    ORDER BY size(c.text) DESC
                    LIMIT 3
                    """, entity_name=entity["name"], question_terms=relevant_data["terms"])
                    
                    for row in result:
                        if row["chunk_text"] and row["source_document"]:
                            # Validate content is relevant
                            if self._validate_research_relevance(question, row["chunk_text"]):
                                findings["content"] += f"\n\nFrom {row['source_document']}:\n{row['chunk_text']}"
                                findings["sources"].append(row["source_document"])
                                findings["entities_referenced"].append(entity["name"])
            
            # If no entities or no findings from entities, search directly in chunks
            if not findings["content"] and relevant_data["documents"]:
                print(f"   🔍 Searching directly in document chunks...")
                for doc_name in relevant_data["documents"][:5]:  # Limit to top 5 documents
                    result = session.run("""
                    MATCH (c:Chunk)-[:HAS_CHUNK]-(d:Document {name: $doc_name})
                    WHERE any(term IN $question_terms WHERE toLower(c.text) CONTAINS term)
                    
                    RETURN 
                        c.text AS chunk_text,
                        d.name AS source_document,
                        d.path AS source_path
                    ORDER BY size(c.text) DESC
                    LIMIT 5
                    """, doc_name=doc_name, question_terms=relevant_data["terms"])
                    
                    for row in result:
                        if row["chunk_text"] and row["source_document"]:
                            # Validate content is relevant
                            if self._validate_research_relevance(question, row["chunk_text"]):
                                findings["content"] += f"\n\nFrom {row['source_document']}:\n{row['chunk_text']}"
                                findings["sources"].append(row["source_document"])
                                findings["entities_referenced"].append("Content from chunks")
        
        return findings if findings["content"] else None
    
    def _validate_research_relevance(self, question: str, content: str) -> bool:
        """Validate that content is actually relevant to the research question."""
        question_terms = set(question.lower().split())
        content_terms = set(content.lower().split())
        
        # For single-word queries, be more lenient
        if len(question_terms) == 1:
            # Single word queries just need the word to appear
            overlap = question_terms.intersection(content_terms)
            return len(overlap) >= 1
        else:
            # Multi-word queries need at least 2 terms to match
            # But be more lenient - if any significant term matches, consider it relevant
            overlap = question_terms.intersection(content_terms)
            if len(overlap) >= 2:
                return True
            
            # Check if any important terms match (for longer queries)
            important_terms = [term for term in question_terms if len(term) > 3]  # Skip short words
            if important_terms:
                important_overlap = set(important_terms).intersection(content_terms)
                return len(important_overlap) >= 1
            
            return False
    
    def _generate_parallel_research_questions(self, research_topic: str, count: int = 3) -> List[str]:
        """Generate initial parallel research questions based on the topic."""
        prompt = f"""
        Generate {count} focused research questions about: {research_topic}
        
        These should be parallel questions that explore different aspects of the topic:
        - One question about what exists/current state
        - One question about key findings/insights  
        - One question about evidence/data sources
        
        Return ONLY the questions, one per line, no numbering or explanations.
        """
        
        try:
            response = self.llm.invoke(prompt)
            # Clean and parse response
            questions = [q.strip() for q in response.strip().split('\n') if q.strip()]
            # Ensure we have exactly the requested number
            if len(questions) >= count:
                return questions[:count]
            else:
                # Fallback questions
                return [
                    f"What specific information exists about {research_topic}?",
                    f"What are the key findings related to {research_topic}?",
                    f"What evidence and data sources support conclusions about {research_topic}?"
                ]
        except Exception as e:
            print(f"   ⚠️ Error generating parallel questions: {e}")
            # Fallback questions
            return [
                f"What specific information exists about {research_topic}?",
                f"What are the key findings related to {research_topic}?",
                f"What evidence and data sources support conclusions about {research_topic}?"
            ]
    
    def _generate_exploratory_question(self, research_topic: str) -> str:
        """Generate an exploratory question to query the database and understand the user's intent."""
        prompt = f"""
        You are an AI assistant that helps analysts at Ubisoft explore and analyze research documents and internal knowledge. When asked a question, you try to think holistically and respond to the user's question in a 360-degree manner.
        
        Given the research topic '{research_topic}', generate a concise, direct question that can be used to query the knowledge graph to understand the user's intent and the primary context of the topic. The question should aim to identify the core nature of the topic and what aspects of it are most relevant in the knowledge graph.
        
        Examples:
        - For "Yasuke": "What is Yasuke and what context, type, and background information exists about this entity in the knowledge graph?"
        - For "performance": "What are the key concepts and entities related to performance in the knowledge graph?"
        - For "player behavior": "What types of player behavior data and insights are available in the knowledge graph?"
        
        Return only the question. Do not include any conversational elements or explanations.
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            print(f"   ⚠️ Error generating exploratory question: {e}")
            return f"What is {research_topic} and what context should I understand about this topic?"
    
    def _generate_contextual_follow_up_question(self, research_topic: str, conversation_context: List[dict], iteration: int) -> str:
        """Generate a contextual follow-up question based on conversation context and research intent."""
        if not conversation_context:
            return f"What specific information exists about {research_topic}?"
        
        # Get the exploratory answer to understand context
        exploratory_answer = ""
        if conversation_context and conversation_context[0].get("type") == "exploratory":
            exploratory_answer = conversation_context[0]["answer"]
        
        # Build conversation text from recent Q&A pairs (excluding exploratory)
        recent_context = [item for item in conversation_context[-3:] if item.get("type") != "exploratory"]
        conversation_text = ""
        if recent_context:
            conversation_text = "\n".join([f"Q: {item['question']}\nA: {item['answer'][:500]}..." for item in recent_context])
        
        prompt = f"""
        You are an AI assistant that helps analysts at Ubisoft explore and analyze research documents and internal knowledge. When asked a question, you try to think holistically and respond to the user's question in a 360-degree manner.
        
        Based on this research context about {research_topic}:
        
        EXPLORATORY CONTEXT:
        {exploratory_answer}
        
        RECENT CONVERSATION:
        {conversation_text if conversation_text else "No previous follow-up questions yet."}
        
        RESEARCH ITERATION: {iteration}
        
        Generate a focused follow-up question that explores a DIFFERENT angle or aspect than previous questions. Think broadly and diversely:
        
        For example, if the topic is "performance" and previous questions focused on gameplay, ask about:
        - Critical reception and reviews
        - Player perception and satisfaction
        - Financial performance and sales data
        - Technical performance metrics
        - Market positioning and competitive analysis
        
        Focus on:
        - Different perspectives and angles
        - Unexplored aspects of the topic
        - Broader business and market context
        - Comparative analysis with other entities
        - Future implications and trends
        
        Return only the question, nothing else.
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            print(f"   ⚠️ Error generating contextual follow-up question: {e}")
            return f"What specific details or evidence exist about {research_topic}?"
    
    def _generate_follow_up_question(self, research_topic: str, conversation_context: List[dict]) -> str:
        """Generate a follow-up question based on the conversation context."""
        if not conversation_context:
            return f"What additional information exists about {research_topic}?"
        
        # Build conversation text from recent Q&A pairs
        recent_context = conversation_context[-3:]  # Last 3 Q&A pairs
        conversation_text = "\n".join([f"Q: {item['question']}\nA: {item['answer'][:500]}..." for item in recent_context])
        
        prompt = f"""
        Based on this conversation about {research_topic}:
        
        {conversation_text}
        
        Generate a relevant follow-up question that would deepen the understanding of the topic.
        The question should be specific and build on the previous answers.
        Focus on areas that haven't been fully explored yet.
        
        Return only the question, nothing else.
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            print(f"   ⚠️ Error generating follow-up question: {e}")
            return f"What additional verified information exists about {research_topic}?"
    
    def _generate_research_question(self, base_question: str, iteration: int, previous_findings: List[dict]) -> str:
        """Generate focused research questions for each iteration."""
        focus_areas = [
            f"What specific data exists about {base_question}?",
            f"What are the key findings related to {base_question}?",
            f"What evidence supports conclusions about {base_question}?",
            f"What methodologies were used to study {base_question}?",
            f"What are the limitations or gaps in data about {base_question}?"
        ]
        
        if iteration < len(focus_areas):
            return focus_areas[iteration]
        else:
            return f"What additional verified information exists about {base_question}?"
    
    def _synthesize_research_wiki(self, research_question: str, findings: List[dict], sources: set) -> str:
        """Synthesize research findings into a comprehensive wiki with strict source attribution."""
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d")
        
        # Generate safe filename for the research question
        safe_title = re.sub(r'[^\w\s-]', '', research_question)
        safe_title = re.sub(r'[-\s]+', '_', safe_title)[:50]
        
        # Create frontmatter
        frontmatter = f"""---
title: "Research: {research_question[:100]}..."
categories: ["Research", "Analysis", "Custom"]
tags: ["research", "knowledge-graph", "data-driven", "custom-analysis", "user-triggered"]
description: "Research analysis on: {research_question}"
date: {current_time}
lastmod: {current_time}
research_question: "{research_question}"
sources_analyzed: {len(sources)}
data_only: true
---

"""
        
        # Create content with strict source attribution
        content = f"""# 🔬 Research Analysis: {research_question}

## 📋 Research Overview

**Research Question:** {research_question}

**Methodology:** This research wiki is based EXCLUSIVELY on data from processed documents in the knowledge graph. No external information or assumptions have been added.

**Sources Analyzed:** {len(sources)} documents
**Data Restriction:** All findings below are directly extracted from the source documents listed at the bottom of this page.

## 🔍 Key Findings

"""
        
        if not findings:
            content += """
**⚠️ No Relevant Data Found**

Based on the processed documents in the knowledge graph, no specific data was found that directly addresses this research question. This may indicate:

1. The relevant documents have not been processed yet
2. The research question is too specific for the available data
3. The topic is not covered in the current document set

Please ensure relevant documents have been processed before conducting this research.
"""
        else:
            # Organize findings by source
            findings_by_source = {}
            for finding in findings:
                for source in finding.get("sources", []):
                    if source not in findings_by_source:
                        findings_by_source[source] = []
                    findings_by_source[source].append(finding["content"])
            
            for source, source_findings in findings_by_source.items():
                content += f"""
### Findings from {source}

"""
                for i, finding in enumerate(source_findings, 1):
                    content += f"{finding}\n\n"
        
        # Add cross-references to related entities
        all_entities = set()
        for finding in findings:
            all_entities.update(finding.get("entities_referenced", []))
        
        if all_entities:
            content += f"""
## 🔗 Related Entities

The following entities from the knowledge graph are referenced in this research:

"""
            for entity in sorted(all_entities):
                content += f"- [[{entity}]]\n"
        
        # Add source attribution section
        content += f"""

## 📚 Source Documents

This research is based on the following source documents:

"""
        
        for source in sorted(sources):
            content += f"- [[Source_{source.replace('.', '_').replace(' ', '_')}|{source}]]\n"
        
        content += f"""

---

**⚠️ Data Integrity Notice:** This research wiki contains ONLY information extracted from the processed source documents listed above. No external information, assumptions, or generated content has been added. All findings can be traced back to specific source documents in the knowledge graph.

*Generated on {current_time}*
"""
        
        return frontmatter + content
    
    def save_research_wiki(self, research_question: str, wiki_content: str) -> str:
        """Save research wiki to main Wikis folder (not subfolder) for Quartz compatibility."""
        # Generate safe filename
        safe_title = re.sub(r'[^\w\s-]', '', research_question)
        safe_title = re.sub(r'[-\s]+', '_', safe_title)[:50]
        
        # Save to main Wikis folder for flat structure
        filepath = self.wiki_folder / f"Research_{safe_title}.md"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(wiki_content)
        
        print(f"   💾 Research wiki saved: {filepath}")
        
        # Update the wiki index
        self.update_wiki_index()
        
        return str(filepath)
    
    def fix_existing_wiki_frontmatter(self, wiki_filepath: str) -> bool:
        """Fix malformed frontmatter in an existing wiki to make it Quartz-compatible."""
        try:
            with open(wiki_filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if content has proper frontmatter
            if not content.startswith('---'):
                print(f"   ⚠️ No frontmatter found in {wiki_filepath}")
                return False
            
            # Extract existing metadata
            existing_metadata = self._extract_wiki_metadata(content)
            
            # Find the end of frontmatter
            end_marker = content.find('---', 3)
            if end_marker == -1:
                print(f"   ⚠️ Malformed frontmatter in {wiki_filepath}")
                return False
            
            # Extract body content (everything after frontmatter)
            body_content = content[end_marker + 3:].strip()
            
            # Generate clean frontmatter
            clean_frontmatter = self._generate_clean_frontmatter(existing_metadata)
            
            # Write back with clean frontmatter
            with open(wiki_filepath, 'w', encoding='utf-8') as f:
                f.write(clean_frontmatter + body_content)
            
            print(f"   ✅ Fixed frontmatter in {wiki_filepath}")
            return True
    
        except Exception as e:
            print(f"   ❌ Error fixing frontmatter in {wiki_filepath}: {e}")
            return False
    
    def _generate_clean_frontmatter(self, metadata: Dict) -> str:
        """Generate clean, Quartz-compatible frontmatter from metadata."""
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d")
        
        lines = ["---"]
        
        # Title (required)
        title = metadata.get('title', 'Untitled Wiki')
        lines.append(f'title: "{title}"')
        
        # Categories (convert entity_type if available)
        categories = []
        if 'entity_type' in metadata:
            categories.append(metadata['entity_type'])
        elif 'wiki_type' in metadata:
            categories.append(metadata['wiki_type'])
        else:
            categories.append("Knowledge Graph")
        
        lines.append(f'categories: {json.dumps(categories, ensure_ascii=False)}')
        
        # Tags
        tags = ["knowledge-graph", "generated"]
        if 'subtype' in metadata and metadata['subtype']:
            tags.append(metadata['subtype'])
        if 'approved' in metadata:
            tags.append(f"status:{metadata['approved']}")
        
        lines.append(f'tags: {json.dumps(tags, ensure_ascii=False)}')
        
        # Description
        description = metadata.get('description', f'Wiki entry for {title}')
        lines.append(f'description: "{description}"')
        
        # Date fields
        lines.append(f'date: {current_time}')
        lines.append(f'lastmod: {current_time}')
        
        # Additional metadata as custom fields
        if 'source_files' in metadata:
            lines.append(f'sources: {json.dumps(metadata["source_files"], ensure_ascii=False)}')
        if 'schema_version' in metadata:
            lines.append(f'schema_version: {metadata["schema_version"]}')
        
        lines.append("---\n")
        return "\n".join(lines)
    
    def fix_all_wiki_frontmatter(self):
        """Fix frontmatter in all existing wikis to ensure Quartz compatibility."""
        print("\n🔧 Fixing frontmatter in all existing wikis...")
        
        # Fix entity wikis
        entity_wikis = list(self.wiki_folder.glob("*.md"))
        fixed_count = 0
        
        for wiki_file in entity_wikis:
            if wiki_file.name != "index.md":  # Skip index file
                if self.fix_existing_wiki_frontmatter(str(wiki_file)):
                    fixed_count += 1
        
        # Fix research wikis
        research_folder = Path("Wikis/Custom Research Wikis")
        if research_folder.exists():
            research_wikis = list(research_folder.glob("*.md"))
            for wiki_file in research_wikis:
                if self.fix_existing_wiki_frontmatter(str(wiki_file)):
                    fixed_count += 1
        
        print(f"   ✅ Fixed frontmatter in {fixed_count} wiki files")
        return fixed_count

    def update_wiki_index(self):
        """Updates the index.md file to reflect the latest wikis."""
        index_path = self.wiki_folder / "index.md"
        if not index_path.exists():
            print("   ⚠️ index.md not found, creating new one.")
            with open(index_path, 'w', encoding='utf-8') as f:
                f.write("# Wiki Index\n\n")
                f.write("This is a list of all generated wikis. For more details, click on the links.\n\n")
                f.write("## Entity Wikis\n\n")
                for wiki_file in self.wiki_folder.glob("*.md"):
                    if wiki_file.name != "index.md":
                        with open(wiki_file, 'r', encoding='utf-8') as f_read:
                            content = f_read.read()
                        metadata = self._extract_wiki_metadata(content)
                        # Use the actual filename (stem) for the wiki link to match Quartz URL generation
                        f.write(f"- [[{wiki_file.stem}]] ({wiki_file.name})\n")
                f.write("\n## Research Wikis\n\n")
                research_folder = Path("Wikis/Custom Research Wikis")
                if research_folder.exists():
                    for wiki_file in research_folder.glob("*.md"):
                        with open(wiki_file, 'r', encoding='utf-8') as f_read:
                            content = f_read.read()
                        metadata = self._extract_wiki_metadata(content)
                        # Use the actual filename (stem) for the wiki link to match Quartz URL generation
                        f.write(f"- [[{wiki_file.stem}]] ({wiki_file.name})\n")
            print(f"   ✅ index.md created at {index_path}")
            return

        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract existing entity wikis
            entity_wikis_in_index = []
            for line in content.splitlines():
                if line.startswith("- [[") and line.endswith("]]"):
                    title = line[3:-2]
                    if not (self.wiki_folder / f"{title}.md").exists():
                        entity_wikis_in_index.append(title)
            
            # Extract existing research wikis
            research_wikis_in_index = []
            research_folder = Path("Wikis/Custom Research Wikis")
            if research_folder.exists():
                for line in content.splitlines():
                    if line.startswith("- [[") and line.endswith("]]"):
                        title = line[3:-2]
                        if not (research_folder / f"{title}.md").exists():
                            research_wikis_in_index.append(title)

            # Re-generate index content
            new_content = "## Entity Wikis\n\n"
            for wiki_file in self.wiki_folder.glob("*.md"):
                if wiki_file.name != "index.md":
                    with open(wiki_file, 'r', encoding='utf-8') as f_read:
                        content = f_read.read()
                    metadata = self._extract_wiki_metadata(content)
                    # Use the actual filename (stem) for the wiki link to match Quartz URL generation
                    new_content += f"- [[{wiki_file.stem}]] ({wiki_file.name})\n"
            new_content += "\n## Research Wikis\n\n"
            if research_folder.exists():
                for wiki_file in research_folder.glob("*.md"):
                    with open(wiki_file, 'r', encoding='utf-8') as f_read:
                        content = f_read.read()
                    metadata = self._extract_wiki_metadata(content)
                    # Use the actual filename (stem) for the wiki link to match Quartz URL generation
                    new_content += f"- [[{wiki_file.stem}]] ({wiki_file.name})\n"

            # Compare and update if necessary
            if new_content != content:
                with open(index_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"   ✅ index.md updated at {index_path}")
            else:
                print(f"   ✅ index.md is already up-to-date at {index_path}")

        except Exception as e:
            print(f"   ❌ Error updating index.md: {e}")

    def create_source_document_wikis(self):
        """Create wikis for all processed source documents."""
        print("📄 Creating wikis for source documents...")
        
        # Get all source documents from database
        source_docs = self._get_source_documents()
        
        if not source_docs:
            print("   ❌ No source documents found in database")
            return
        
        created_count = 0
        for doc in source_docs:
            try:
                wiki_content = self._create_source_document_wiki(doc)
                if wiki_content:
                    # Save to Wikis folder with source-files tag
                    filename = doc["filename"].replace('.', '_').replace(' ', '_')
                    filepath = self.wiki_folder / f"Source_{filename}.md"
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(wiki_content)
                    
                    created_count += 1
                    print(f"   ✅ Created wiki for {doc['filename']}")
                    
            except Exception as e:
                print(f"   ❌ Error creating wiki for {doc['filename']}: {e}")
        
        print(f"🎉 Created {created_count} source document wikis")
        
        # Update index
        self.update_wiki_index()
        
        return created_count
    
    def _get_source_documents(self) -> List[Dict]:
        """Get all processed source documents from the database."""
        with self.driver.session() as session:
            result = session.run("""
            MATCH (d:Document)
            RETURN 
                d.filename AS filename,
                d.file_size AS file_size,
                d.processed_at AS processed_at,
                d.content_preview AS content_preview
            ORDER BY d.filename
            """)
            
            documents = []
            for row in result:
                documents.append({
                    "filename": row["filename"],
                    "file_size": row["file_size"] or 0,
                    "processed_at": row["processed_at"] or "Unknown",
                    "content_preview": row["content_preview"] or "No preview available"
                })
            
            return documents

    def _create_source_document_wiki(self, doc: dict) -> str:
        """Create wiki content for a source document."""
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d")
        
        # Debug: Print what we received
        print(f"      🔍 Debug: Received doc type: {type(doc)}")
        print(f"      🔍 Debug: Received doc content: {doc}")
        
        # Ensure doc has the expected structure - handle both dict and object cases
        filename = "Unknown"
        file_size = 0
        processed_at = "Unknown"
        content_preview = "No preview available"
        
        try:
            if isinstance(doc, dict):
                # It's a dictionary
                filename = doc.get("filename", "Unknown")
                file_size = doc.get("file_size", 0)
                processed_at = doc.get("processed_at", "Unknown")
                content_preview = doc.get("content_preview", "No preview available")
                print(f"      🔍 Debug: Using dict access - filename: {filename}")
            elif hasattr(doc, 'get') and callable(getattr(doc, 'get', None)):
                # It's a dictionary-like object
                filename = doc.get("filename", "Unknown")
                file_size = doc.get("file_size", 0)
                processed_at = doc.get("processed_at", "Unknown")
                content_preview = doc.get("content_preview", "No preview available")
                print(f"      🔍 Debug: Using dict-like access - filename: {filename}")
            elif hasattr(doc, 'filename'):
                # It's an object with attributes
                filename = getattr(doc, "filename", "Unknown")
                file_size = getattr(doc, "file_size", 0)
                processed_at = getattr(doc, "processed_at", "Unknown")
                content_preview = getattr(doc, "content_preview", "No preview available")
                print(f"      🔍 Debug: Using attr access - filename: {filename}")
            else:
                # Fallback: try to convert to string and extract filename
                doc_str = str(doc) if doc else "Unknown"
                filename = doc_str[:50]
                print(f"      🔍 Debug: Using fallback - filename: {filename}")
        except Exception as e:
            print(f"      🔍 Debug: Error extracting data: {e}")
            # Final fallback
            filename = str(doc)[:50] if doc else "Unknown"
            print(f"      🔍 Debug: Using final fallback - filename: {filename}")
        
        # Get chunks and entities from this document
        with self.driver.session() as session:
            result = session.run("""
            MATCH (d:Document {filename: $filename})
            OPTIONAL MATCH (d)-[:HAS_CHUNK]-(c:Chunk)
            OPTIONAL MATCH (e)-[:MENTIONED_IN_CHUNK]->(c)
            WHERE NOT 'Chunk' IN labels(e) AND NOT 'Document' IN labels(e)
            
            RETURN 
                c.text AS chunk_text,
                c.chunk_id AS chunk_id,
                collect(DISTINCT e.name) AS entities,
                d.content_preview AS preview,
                d.file_size AS file_size,
                d.processed_at AS processed_at
            ORDER BY c.chunk_id
            """, filename=filename)
            
            chunks = []
            all_entities = set()
            
            for row in result:
                if row["chunk_text"]:
                    chunks.append({
                        "id": row["chunk_id"],
                        "text": row["chunk_text"]
                    })
                if row["entities"]:
                    all_entities.update(row["entities"])
        
        # Generate frontmatter
        frontmatter = f"""---
title: "Source: {filename}"
categories: ["Source Document"]
tags: ["source-files", "document", "processed", "knowledge-graph"]
description: "Source document: {filename}"
date: {current_time}
lastmod: {current_time}
file_size: {file_size}
processed_at: "{processed_at}"
total_chunks: {len(chunks)}
entities_extracted: {len(all_entities)}
---

"""
        
        # Generate content
        content = f"""# 📄 Source Document: {filename}

## 📊 Document Metadata

- **File Size:** {file_size} bytes
- **Processed:** {processed_at}
- **Total Chunks:** {len(chunks)}
- **Entities Extracted:** {len(all_entities)}

## 📝 Document Preview

{content_preview}

## 🧠 Extracted Entities

"""
        
        if all_entities:
            for entity in sorted(all_entities):
                content += f"- [[{entity}]]\n"
        else:
            content += "*No entities extracted from this document.*\n"
        
        content += f"""

## 📚 Document Chunks

*This document was processed into {len(chunks)} chunks for knowledge extraction.*

"""
        
        # Add first few chunks as preview
        for i, chunk in enumerate(chunks[:3]):
            content += f"### Chunk {chunk['id']}\n\n"
            content += f"{chunk['text'][:500]}...\n\n"
        
        if len(chunks) > 3:
            content += f"*...and {len(chunks) - 3} more chunks*\n\n"
        
        content += f"""
---

*This source document wiki was automatically generated from the knowledge graph data.*
"""
        
        return frontmatter + content
    
    def _synthesize_semantic_research_wiki(self, research_question: str, findings: List[dict]) -> str:
        """Synthesize SemanticAgent research findings into a comprehensive wiki."""
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d")
        
        # Generate safe filename for the research question
        safe_title = re.sub(r'[^\w\s-]', '', research_question)
        safe_title = re.sub(r'[-\s]+', '_', safe_title)[:50]
        
        # Generate intelligent tags for research question
        intelligent_tags = self._generate_research_tags(research_question)
        
        # Create frontmatter
        frontmatter = f"""---
title: "Research: {research_question[:100]}..."
categories: ["Research", "Analysis", "Custom"]
tags: {intelligent_tags}
description: "Research analysis on: {research_question} using SemanticAgent"
date: {current_time}
lastmod: {current_time}
research_question: "{research_question}"
questions_asked: {len(findings)}
methodology: "SemanticAgent knowledge graph queries"
---

"""
        
        # Create content
        content = f"""# 🔬 Research Analysis: {research_question}

## 📋 Research Overview

**Research Question:** {research_question}

**Methodology:** This research wiki was created using the same high-quality SemanticAgent system that powers the main knowledge graph query interface. All answers are generated from actual knowledge graph data.

**Questions Asked:** {len(findings)} total questions
**Research Approach:** Exploratory question followed by focused follow-up questions

## 🎯 Executive Summary

"""
        
        # Generate executive summary from all findings
        try:
            executive_summary = self._generate_executive_summary(research_question, findings)
            content += executive_summary
        except Exception as e:
            print(f"   ⚠️ Error generating executive summary: {e}")
            # Fallback summary
            content += f"""**TL;DR Summary:**
This research explored **{research_question}** through {len(findings)} systematic questions, uncovering insights across multiple dimensions including gameplay mechanics, market positioning, player engagement, and technical implementation. The findings provide a comprehensive understanding of the topic based on knowledge graph data.

*Note: Detailed executive summary generation failed due to technical constraints. See detailed findings below for complete analysis.*
"""
        
        content += """

## 🔍 Detailed Research Findings

"""
        
        if not findings:
            content += """
**⚠️ No Research Data Collected**

The research process did not generate any findings. This may indicate:
1. Issues with the SemanticAgent
2. No relevant data in the knowledge graph
3. Technical errors during the research process
"""
        else:
            # Organize by question type (exploratory vs follow-up)
            exploratory_qa = None
            follow_up_qa_list = []
            
            for qa in findings:
                if qa.get("type") == "exploratory":
                    exploratory_qa = qa
                else:
                    follow_up_qa_list.append(qa)
            
            # Add exploratory findings
            if exploratory_qa:
                content += "### 🚀 Exploratory Research\n\n"
                content += f"**Question:** {exploratory_qa['question']}\n\n"
                # Extract clean answer without source references
                clean_answer = self._extract_clean_answer(exploratory_qa['answer'])
                content += f"{clean_answer}\n\n"
                content += "---\n\n"
            
            # Add follow-up findings
            if follow_up_qa_list:
                content += "### 🔄 Follow-up Research Findings\n\n"
                for i, qa in enumerate(follow_up_qa_list, 1):
                    content += f"#### Follow-up {i}: {qa['question']}\n\n"
                    # Extract clean answer without source references
                    clean_answer = self._extract_clean_answer(qa['answer'])
                    content += f"{clean_answer}\n\n"
                    content += "---\n\n"
        
        # Add research summary
        content += f"""
## 📊 Research Summary

This research explored **{research_question}** through a systematic approach:

1. **Exploratory Understanding**: Initial question to understand context and intent
2. **Focused Investigation**: {len(follow_up_qa_list)} follow-up questions building on previous answers
3. **Knowledge Integration**: All findings synthesized from the knowledge graph

## 🔗 Related Research

{self._generate_related_research_links_from_question(research_question)}

## 🤖 AI Methodology

This research wiki was created using an AI-driven approach with the following questions:

### Exploratory Question
{exploratory_qa['question'] if exploratory_qa else 'No exploratory question generated'}

### Follow-up Questions
"""
        
        # Add all follow-up questions
        for i, qa in enumerate(follow_up_qa_list, 1):
            content += f"{i}. {qa['question']}\n"
        
        # Add source references section
        content += f"""
## 📚 Source References

The following source documents were referenced during this research:

"""
        
        # Collect only actual document filenames from all findings
        document_sources = set()
        for qa in findings:
            answer_text = qa.get('answer', '')
            
            if isinstance(answer_text, str):
                # Look specifically for document filenames with extensions
                # Focus on patterns that clearly indicate document names
                doc_patterns = [
                    # Look for document names in brackets (most common format)
                    r'\[([^\[\]]*\.(?:docx|pdf|md|txt)[^\[\]]*)\]',
                    # Look for document names after "from" or "in" 
                    r'(?:from|in|based on|according to)\s+([^\.\n]+\.(?:docx|pdf|md|txt))',
                    # Look for document names in quotes
                    r'"([^"]*\.(?:docx|pdf|md|txt))"',
                    r"'([^']*\.(?:docx|pdf|md|txt))'",
                    # Look for any standalone document filename
                    r'\b([a-zA-Z0-9_\-\s]+\.(?:docx|pdf|md|txt))\b'
                ]
                
                for pattern in doc_patterns:
                    matches = re.findall(pattern, answer_text, re.IGNORECASE)
                    for match in matches:
                        if match and len(match.strip()) > 5:  # Basic validation
                            clean_filename = match.strip().strip('"\'')
                            if clean_filename and not clean_filename.startswith('--'):
                                document_sources.add(clean_filename)
        
        # If no document sources found, try to query the database for relevant documents
        if not document_sources:
            print("   🔍 No document sources found in answers, querying database for relevant documents...")
            db_documents = self._find_relevant_documents_from_db(research_question)
            if db_documents:
                document_sources.update(db_documents)
                print(f"   ✅ Found {len(db_documents)} relevant documents from database")
        
        if not document_sources:
            content += "*No specific source documents referenced.*\n"
            content += "\n*Note: The research was conducted using the knowledge graph, but specific source document attribution was not available.*\n"
        else:
            # Clean up document names and remove duplicates
            cleaned_docs = set()
            for doc_name in document_sources:
                # Remove "from:" prefixes and clean up
                clean_name = doc_name.replace('from:', '').strip()
                if clean_name and len(clean_name) > 5:
                    cleaned_docs.add(clean_name)
            
            if cleaned_docs:
                content += "The following source documents were referenced during this research:\n\n"
                for doc_name in sorted(cleaned_docs):
                    # Create a link to the existing Source_ wiki if it exists
                    safe_source_name = doc_name.replace('.', '_').replace(' ', '_')
                    source_wiki_name = f"Source_{safe_source_name}"
                    
                    # Check if the Source_ wiki exists
                    source_wiki_path = self.wiki_folder / f"{source_wiki_name}.md"
                    if source_wiki_path.exists():
                        content += f"- [[{source_wiki_name}|{doc_name}]]\n"
                    else:
                        # If no Source_ wiki exists, just show the filename
                        content += f"- {doc_name}\n"
            else:
                content += "*No specific source documents referenced.*\n"
        
        content += f"""

---

**📝 Research Methodology:** This wiki was generated using the SemanticAgent system, which provides the same high-quality responses as the main knowledge graph query interface. All information comes directly from the processed knowledge graph data.

*Generated on {current_time} using SemanticAgent*
"""
        
        # Add smart wiki-to-wiki linking
        content = self._add_smart_wiki_links(content)
        
        # Add source document references
        content = self._add_research_source_references(content, research_question)
        
        return frontmatter + content
    
    def _generate_executive_summary(self, research_question: str, findings: List[dict]) -> str:
        """Generate a comprehensive executive summary from all research findings."""
        try:
            # Prepare context from all findings
            context_parts = []
            for i, qa in enumerate(findings, 1):
                qa_type = qa.get("type", "unknown")
                question = qa.get("question", "Unknown question")
                answer = qa.get("answer", "No answer")
                
                # Clean the answer to remove source references
                clean_answer = self._extract_clean_answer(answer)
                
                context_parts.append(f"Q{i} ({qa_type}): {question}\nA{i}: {clean_answer[:1000]}...")
            
            # Combine all context
            full_context = "\n\n".join(context_parts)
            
            # Generate executive summary prompt
            prompt = f"""You are an AI assistant that helps analysts at Ubisoft create executive summaries of research findings.

Based on the following research question and all the Q&A findings below, create a comprehensive executive summary that:

1. **Summarizes the key insights** discovered across all questions
2. **Identifies the main themes** and patterns that emerged
3. **Highlights the most important findings** that answer the research question
4. **Provides actionable insights** where possible
5. **Is written in executive summary style** - concise, clear, and business-focused

Research Question: {research_question}

Research Findings:
{full_context}

Create a comprehensive executive summary that captures the essence of what was discovered. Focus on synthesizing the findings into coherent insights rather than just listing what was asked.

Return the executive summary in this format:
**Executive Summary:**
[Your comprehensive summary here]

**Key Insights:**
- [Insight 1]
- [Insight 2]
- [Insight 3]

**Main Themes:**
- [Theme 1]
- [Theme 2]
- [Theme 3]
"""

            # Generate the summary
            response = self.llm.invoke(prompt)
            return response.strip()
            
        except Exception as e:
            # If there's an error (e.g., context too long), return a fallback
            print(f"   ⚠️ Executive summary generation failed: {e}")
            raise Exception(f"Executive summary generation failed: {e}")
    
    def _extract_clean_answer(self, answer_data) -> str:
        """Extract clean answer text from SemanticAgent response, removing source references."""
        if isinstance(answer_data, str):
            answer_text = answer_data
        elif isinstance(answer_data, dict) and 'answer' in answer_data:
            answer_text = answer_data['answer']
        else:
            return str(answer_data)
        
        if not isinstance(answer_text, str):
            return str(answer_text)
        
        # Remove source reference sections
        lines = answer_text.split('\n')
        clean_lines = []
        in_source_section = False
        
        for line in lines:
            # Check if this line starts a source section
            if any(pattern in line.lower() for pattern in ['--references', '--adjacent nodes', '--documents accessed', '--related wikis']):
                in_source_section = True
                continue
            
            # If we're in a source section, skip until we hit a new header
            if in_source_section:
                if line.startswith('##') or line.startswith('###'):
                    in_source_section = False
                    # Only add the header if it's not a source-related one
                    if not any(keyword in line.lower() for keyword in ['source', 'reference', 'document', 'chunk']):
                        clean_lines.append(line)
                continue
            
            # Add clean lines
            clean_lines.append(line)
        
        return '\n'.join(clean_lines).strip()
    

    
    def _find_potentially_relevant_documents(self, research_question: str) -> set:
        """Find potentially relevant documents from the database based on the research question."""
        try:
            # Extract key terms from the research question
            question_terms = [term.lower() for term in research_question.split() if len(term) > 3]
            
            if not question_terms:
                return set()
            
            with self.driver.session() as session:
                # Search for documents that contain the research terms
                result = session.run("""
                MATCH (c:Chunk)-[:HAS_CHUNK]-(d:Document)
                WHERE any(term IN $terms WHERE toLower(c.text) CONTAINS term)
                RETURN DISTINCT d.name AS document_name
                ORDER BY d.name
                LIMIT 20
                """, terms=question_terms)
                
                documents = set()
                for row in result:
                    if row["document_name"]:
                        documents.add(row["document_name"])
                
                return documents
                
        except Exception as e:
            print(f"   ⚠️ Error querying database for relevant documents: {e}")
            return set()
    
    def _find_relevant_documents_from_db(self, research_question: str) -> set:
        """Find relevant documents from the database based on the research question."""
        try:
            # Extract key terms from the research question
            question_terms = [term.lower() for term in research_question.split() if len(term) > 3]
            
            if not question_terms:
                return set()
            
            with self.driver.session() as session:
                # Search for documents that contain the research terms in their chunks
                result = session.run("""
                MATCH (c:Chunk)-[:HAS_CHUNK]-(d:Document)
                WHERE any(term IN $terms WHERE toLower(c.text) CONTAINS term)
                RETURN DISTINCT d.name AS document_name
                ORDER BY d.name
                LIMIT 15
                """, terms=question_terms)
                
                documents = set()
                for row in result:
                    if row["document_name"]:
                        documents.add(row["document_name"])
                
                return documents
                
        except Exception as e:
            print(f"   ⚠️ Error querying database for relevant documents: {e}")
            return set()
    
    def _get_source_documents(self) -> List[Dict]:
        """Get all processed source documents from the database."""
        with self.driver.session() as session:
            result = session.run("""
            MATCH (d:Document)
            RETURN 
                d.filename AS filename,
                d.file_size AS file_size,
                d.processed_at AS processed_at,
                d.content_preview AS content_preview
            ORDER BY d.filename
            """)
            
            documents = []
            for row in result:
                documents.append({
                    "filename": row["filename"],
                    "file_size": row["file_size"] or 0,
                    "processed_at": row["processed_at"] or "Unknown",
                    "content_preview": row["content_preview"] or "No preview available"
                })
            
            return documents

def main():
    """Main interactive wiki generation interface."""
    agent = WikiGenerationAgent()
    
    try:
        print("🎯 Multi-Step Wiki Generation Agent")
        print("=" * 50)
        
        while True:
            print("\nOptions:")
            print("1. Get wiki recommendations")
            print("2. Generate specific wiki")
            print("3. Create cross-links between wikis")
            print("4. Create Research Wiki")
            print("5. Exit")
            
            choice = input("\nChoose an option: ").strip()
            
            if choice == "1":
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
            
            elif choice == "2":
                entity_name = input("Enter entity name: ").strip()
                entity_type = input("Enter entity type (Brand, Game, Person, etc.): ").strip()
                
                # Create a topic object
                topic = WikiTopic(
                    name=entity_name,
                    entity_type=entity_type,
                    relevance_score=8.0,
                    evidence_count=5,
                    source_documents=3,
                    centrality_score=10,
                    description=f"Manual selection: {entity_name}"
                )
                
                config = agent.get_iteration_limit()
                wiki_content = agent.generate_wiki(topic, config)
                filepath = agent.save_wiki(topic, wiki_content)
                print(f"\n🎉 Wiki generated successfully: {filepath}")
            
            elif choice == "3":
                agent.create_cross_links()
            
            elif choice == "4":
                print("🔬 Creating Research Wiki...")
                research_question = input("Enter your research question or topic: ").strip()
                if research_question:
                    config = agent.get_iteration_limit()
                    wiki_content = agent.create_research_wiki(research_question, config["total"])
                    filepath = agent.save_research_wiki(research_question, wiki_content)
                    print(f"\n🎉 Research wiki created successfully: {filepath}")
                else:
                    print("❌ No research question provided.")
            
            elif choice == "5":
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice, try again.")
    
    finally:
        agent.close()

if __name__ == "__main__":
    main()

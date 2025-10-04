import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from .litellm_wrapper import LiteLLMChat, LiteLLMEmbeddings
from .glossary_resolver import GlossaryResolver

load_dotenv()


class SemanticAgent:
    def __init__(self):
        # Use LiteLLM wrappers
        self.llm = LiteLLMChat(model=os.getenv("LITELLM_MODEL", "gpt-4o"))
        self.embeddings = LiteLLMEmbeddings(model=os.getenv("LITELLM_EMBED_MODEL", "text-embedding-3-large"))

        # Initialize glossary resolver for synonym expansion
        self.glossary_resolver = GlossaryResolver()

        # Connect to Neo4j
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
        )

    def get_relevant_chunks(self, question, top_k=10):
        embedding = self.embeddings.embed_query(question)
        with self.driver.session() as session:
            result = session.run(
                """
                CALL db.index.vector.queryNodes('vector', $topK, $embedding)
                YIELD node, score
                RETURN node.chunk_id AS cid, node.text AS chunk, score
                ORDER BY score DESC
                LIMIT $topK
                """,
                embedding=embedding,
                topK=top_k,
            )
            return [{"id": row["cid"], "text": row["chunk"]} for row in result]

    def get_document_info(self, chunk_ids):
        """Get document information for given chunk IDs."""
        documents = {}
        with self.driver.session() as session:
            for cid in chunk_ids:
                result = session.run(
                    """
                    MATCH (c:Chunk {chunk_id: $cid})<-[:HAS_CHUNK]-(d:Document)
                    RETURN d.name AS doc_name, d.path AS doc_path, d.source_id AS doc_id
                    """,
                    cid=cid,
                )
                for row in result:
                    doc_name = row["doc_name"]
                    doc_path = row["doc_path"]
                    doc_id = row["doc_id"]
                    if doc_name not in documents:
                        documents[doc_name] = {
                            "path": doc_path,
                            "source_id": doc_id,
                            "chunks": []
                        }
                    documents[doc_name]["chunks"].append(cid)
        return documents

    def get_chunk_document_info(self, chunk_ids):
        """Get document info for each chunk."""
        chunk_docs = {}
        with self.driver.session() as session:
            for cid in chunk_ids:
                result = session.run(
                    """
                    MATCH (c:Chunk {chunk_id: $cid})<-[:HAS_CHUNK]-(d:Document)
                    RETURN d.name AS doc_name
                    """,
                    cid=cid,
                )
                for row in result:
                    chunk_docs[cid] = row["doc_name"]
        return chunk_docs

    def get_adjacent_nodes(self, entity_ids, chunk_entities, max_relationships=50):
        """Get adjacent nodes for entities mentioned in the selected chunks, prioritized by connection count."""
        adjacent_nodes = []
        
        print(f"🔍 Found {len(entity_ids)} entities in chunks, finding all relationships")
        
        # Only process entities that are actually mentioned in the selected chunks
        chunk_entity_set = set(chunk_entities)
        
        # First pass: collect ALL relationships for entities in chunks
        with self.driver.session() as session:
            for name, typ in entity_ids:
                # Skip if this entity is not in our selected chunks
                if (name, typ) not in chunk_entity_set:
                    continue
                    
                print(f"   Processing chunk entity: {name} ({typ})")
                
                # Outgoing relationships
                result = session.run(
                    """
                    MATCH (e)
                    WHERE e.name = $name AND $type IN labels(e)
                    MATCH (e)-[rel]->(n)
                    WHERE NOT 'Chunk' IN labels(n) AND NOT 'Document' IN labels(n)
                    RETURN e.name AS entity_name, labels(e) AS entity_type,
                           type(rel) AS rel_type, n.name AS neighbor_name, labels(n) AS neighbor_type
                    """,
                    name=name,
                    type=typ,
                )
                for row in result:
                    rel = row["rel_type"]
                    nbr = row["neighbor_name"]
                    if rel and nbr:
                        et = row["entity_type"][0] if row["entity_type"] else ""
                        nt = row["neighbor_type"][0] if row["neighbor_type"] else ""
                        adjacent_nodes.append({
                            "source": f"{row['entity_name']} ({et})",
                            "relationship": rel,
                            "target": f"{nbr} ({nt})",
                            "direction": "outgoing",
                            "source_entity": (row['entity_name'], et),
                            "target_entity": (nbr, nt)
                        })
                
                # Incoming relationships
                result = session.run(
                    """
                    MATCH (e)
                    WHERE e.name = $name AND $type IN labels(e)
                    MATCH (n)-[rel]->(e)
                    WHERE NOT 'Chunk' IN labels(n) AND NOT 'Document' IN labels(n)
                    RETURN e.name AS entity_name, labels(e) AS entity_type,
                           type(rel) AS rel_type, n.name AS neighbor_name, labels(n) AS neighbor_type
                    """,
                    name=name,
                    type=typ,
                )
                for row in result:
                    rel = row["rel_type"]
                    nbr = row["neighbor_name"]
                    if rel and nbr:
                        et = row["entity_type"][0] if row["entity_type"] else ""
                        nt = row["neighbor_type"][0] if row["neighbor_type"] else ""
                        adjacent_nodes.append({
                            "source": f"{nbr} ({nt})",
                            "relationship": rel,
                            "target": f"{row['entity_name']} ({et})",
                            "direction": "incoming",
                            "source_entity": (nbr, nt),
                            "target_entity": (row['entity_name'], et)
                        })
        
        # Remove duplicates based on source-target-relationship combination
        unique_nodes = []
        seen = set()
        for node in adjacent_nodes:
            # Create a more specific key that includes direction to avoid duplicates
            key = (node["source"], node["target"], node["relationship"], node["direction"])
            if key not in seen:
                seen.add(key)
                unique_nodes.append(node)
        
        # Count how many relationships each entity has to the selected chunks
        entity_relationship_counts = {}
        for node in unique_nodes:
            # Count source entity relationships
            source_entity = node["source_entity"]
            if source_entity not in entity_relationship_counts:
                entity_relationship_counts[source_entity] = 0
            entity_relationship_counts[source_entity] += 1
            
            # Count target entity relationships
            target_entity = node["target_entity"]
            if target_entity not in entity_relationship_counts:
                entity_relationship_counts[target_entity] = 0
            entity_relationship_counts[target_entity] += 1
        
        # Sort entities by relationship count (descending)
        sorted_entities = sorted(entity_relationship_counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"📊 Entity relationship counts (top 10):")
        for entity, count in sorted_entities[:10]:
            print(f"   {entity[0]} ({entity[1]}): {count} relationships")
        
        # Get the top N most connected entities
        top_entities = set(entity for entity, _ in sorted_entities[:max_relationships])
        
        # Filter relationships to only include the most connected entities
        filtered_nodes = []
        for node in unique_nodes:
            if (node["source_entity"] in top_entities or 
                node["target_entity"] in top_entities):
                filtered_nodes.append(node)
        
        # Remove redundant relationships (same source-target with different relationship types)
        # Keep the most specific relationship type
        relationship_priority = {
            "belongs_to_brand": 1,
            "BELONGS_TO_BRAND": 1,
            "developed_by": 2,
            "published_by": 2,
            "compares_with": 3,
            "has_insight": 4,
            "measured_by": 5,
            "measured_for": 5,
            "covers_period": 6,
            "collected_in": 6,
            "documented_in": 6,
            "targets_segment": 7,
            "is_type_of": 8,
            "is_feature_of": 8,
            "drives": 9,
            "appeals_to": 9,
            "includes": 9,
            "performs_on": 9,
            "targets": 9,
            "has_minister": 10,
            "start_date": 10,
            "analyzes": 11,
            "is_synonym_of": 12
        }
        
        # Group by source-target pairs and keep the highest priority relationship
        source_target_groups = {}
        for node in filtered_nodes:
            source_target = (node["source"], node["target"])
            if source_target not in source_target_groups:
                source_target_groups[source_target] = []
            source_target_groups[source_target].append(node)
        
        # For each source-target pair, keep the highest priority relationship
        deduplicated_nodes = []
        for source_target, nodes in source_target_groups.items():
            if len(nodes) == 1:
                deduplicated_nodes.append(nodes[0])
            else:
                # Multiple relationships between same entities, keep the highest priority
                best_node = min(nodes, key=lambda x: relationship_priority.get(x["relationship"], 999))
                deduplicated_nodes.append(best_node)
        
        # Sort by entity connection count (entities with more connections get priority)
        # Create a scoring system based on entity connection counts
        def get_relationship_score(node):
            source_count = entity_relationship_counts.get(node["source_entity"], 0)
            target_count = entity_relationship_counts.get(node["target_entity"], 0)
            return source_count + target_count
        
        # Sort by relationship score (descending) and take top relationships
        scored_nodes = [(node, get_relationship_score(node)) for node in deduplicated_nodes]
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        final_nodes = [node[0] for node in scored_nodes[:max_relationships]]
        
        # Convert back to string format
        formatted_nodes = []
        for node in final_nodes:
            if node["direction"] == "outgoing":
                formatted_nodes.append(f"{node['source']} --[{node['relationship']}]--> {node['target']}")
            else:
                formatted_nodes.append(f"{node['source']} --[{node['relationship']}]--> {node['target']}")
        
        print(f"📊 Found {len(unique_nodes)} unique relationships, deduplicated to {len(deduplicated_nodes)}, showing top {len(final_nodes)} most relevant relationships")
        
        return formatted_nodes

    def get_related_wikis(self, entity_ids):
        """Get related wiki pages for given entities."""
        wikis = set()
        with self.driver.session() as session:
            for name, typ in entity_ids:
                result = session.run(
                    """
                    MATCH (e)
                    WHERE e.name = $name AND $type IN labels(e)
                    RETURN labels(e) AS labels
                    """,
                    name=name,
                    type=typ,
                )
                for row in result:
                    labels = row["labels"]
                    for label in labels:
                        if label not in ["Chunk", "Document"]:
                            wikis.add(label)
        return list(wikis)

    def run_query(self, question):
        # Step 0: Resolve synonyms in the question
        expanded_question, synonym_mapping = self.glossary_resolver.resolve_query(question)
        if synonym_mapping:
            print(f"🔍 Resolved synonyms: {synonym_mapping}")
        
        # Step 1: dense retrieval using expanded question
        chunks = self.get_relevant_chunks(expanded_question)
        if not chunks:
            return {
                "answer": "🕵️ No relevant information found in the knowledge graph.",
                "references": [],
                "adjacent_nodes": [],
                "documents_accessed": []
            }
        
        chunk_ids = [c["id"] for c in chunks]

        # Step 2: collect mentions for each chunk
        chunk_to_mentions = {}
        entity_ids = set()
        with self.driver.session() as session:
            for cid in chunk_ids:
                mentions = []
                for row in session.run(
                    """
                    MATCH (c:Chunk {chunk_id: $cid})<-[:MENTIONED_IN_CHUNK]-(n)
                    WHERE NOT 'Chunk' IN labels(n)
                    RETURN n.name AS name, labels(n) AS labels
                    """,
                    cid=cid,
                ):
                    name = row["name"]
                    labels = row["labels"]
                    if name:
                        label = labels[0] if labels else ""
                        mentions.append(f"{name} ({label})")
                        entity_ids.add((name, label))
                chunk_to_mentions[cid] = mentions
        
        print(f"📊 Found {len(entity_ids)} unique entities across {len(chunk_ids)} chunks")
        print(f"📋 Entity types found: {set(typ for _, typ in entity_ids)}")
        
        # Show top entities by frequency
        entity_counts = {}
        for name, typ in entity_ids:
            entity_counts[(name, typ)] = entity_counts.get((name, typ), 0) + 1
        
        top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"🔝 Top entities by frequency:")
        for (name, typ), count in top_entities:
            print(f"   {name} ({typ}): {count} mentions")

        # Step 3: Get adjacent nodes (both incoming and outgoing)
        # Get all entities mentioned in the selected chunks
        chunk_entities = []
        for cid in chunk_ids:
            chunk_entities.extend(chunk_to_mentions.get(cid, []))
        
        # Convert mentions to entity tuples
        chunk_entity_tuples = []
        for mention in chunk_entities:
            if " (" in mention and mention.endswith(")"):
                name = mention.split(" (")[0]
                typ = mention.split(" (")[1][:-1]  # Remove closing parenthesis
                chunk_entity_tuples.append((name, typ))
        
        adjacent_nodes = self.get_adjacent_nodes(entity_ids, chunk_entity_tuples)

        # Step 4: compose context
        context_sections = []
        for c in chunks:
            section = c["text"] + "\nMentions in this chunk:"
            for m in (chunk_to_mentions.get(c["id"], []) or ["(none)"]):
                section += f"\n- {m}"
            context_sections.append(section)

        if adjacent_nodes:
            context_sections.append(
                "Additional facts from knowledge graph:\n" + "\n".join(adjacent_nodes)
            )

        context = "\n\n---\n\n".join(context_sections)
        prompt = f"""Answer the following question using only the information in the context below.

Question: {question}

Context:
{context}

Answer:"""

        answer = self.llm.invoke(prompt)

        # Step 5: Gather citation information
        # Get document information
        documents_accessed = self.get_document_info(chunk_ids)
        chunk_docs = self.get_chunk_document_info(chunk_ids)
        
        # Get related wikis
        related_wikis = self.get_related_wikis(entity_ids)
        
        # Format references (cited chunks) - just first 5-10 words + doc name
        references = []
        for chunk in chunks:
            # Get first 5-10 words
            words = chunk["text"].split()[:10]
            preview = " ".join(words)
            if len(chunk["text"].split()) > 10:
                preview += "..."
            
            doc_name = chunk_docs.get(chunk["id"], "Unknown")
            references.append({
                "chunk_id": chunk["id"],
                "text_preview": preview,
                "document": doc_name,
                "mentions": chunk_to_mentions.get(chunk["id"], [])
            })

        # Format documents accessed
        documents_list = []
        for doc_name, doc_info in documents_accessed.items():
            documents_list.append({
                "name": doc_name,
                "path": doc_info["path"],
                "source_id": doc_info["source_id"],
                "chunk_count": len(doc_info["chunks"])
            })

        # Create structured response
        response = {
            "answer": answer,
            "references": references,
            "adjacent_nodes": adjacent_nodes,
            "documents_accessed": documents_list,
            "related_wikis": related_wikis
        }

        # Format the response with citations at the end
        formatted_response = answer + "\n\n"
        formatted_response += "--references (cited chunks)\n"
        for i, ref in enumerate(references, 1):
            formatted_response += f"{i}. {ref['text_preview']} [from: {ref['document']}]\n"
            if ref['mentions']:
                formatted_response += f"   Mentions: {', '.join(ref['mentions'])}\n"
            formatted_response += "\n"

        formatted_response += "--adjacent nodes retrieved\n"
        for i, node in enumerate(adjacent_nodes, 1):
            formatted_response += f"{i}. {node}\n"

        formatted_response += "\n--documents accessed\n"
        for i, doc in enumerate(documents_list, 1):
            formatted_response += f"{i}. {doc['name']} (ID: {doc['source_id']})\n"

        if related_wikis:
            formatted_response += "\n--related wikis for context\n"
            for i, wiki in enumerate(related_wikis, 1):
                formatted_response += f"{i}. {wiki}\n"

        return formatted_response

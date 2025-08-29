import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from litellm_wrapper import LiteLLMChat, LiteLLMEmbeddings

load_dotenv()


class SemanticAgent:
    def __init__(self):
        # Use LiteLLM wrappers
        self.llm = LiteLLMChat(model=os.getenv("LITELLM_MODEL", "gpt-4o"))
        self.embeddings = LiteLLMEmbeddings(model=os.getenv("LITELLM_EMBED_MODEL", "text-embedding-3-large"))

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

    def run_query(self, question):
        # Step 1: dense retrieval
        chunks = self.get_relevant_chunks(question)
        if not chunks:
            return "🕵️ No relevant information found in the knowledge graph."
        chunk_ids = [c["id"] for c in chunks]

        # Step 2: collect mentions for each chunk
        chunk_to_mentions = {}
        entity_ids = set()
        with self.driver.session() as session:
            for cid in chunk_ids:
                mentions = []
                for row in session.run(
                    """
                    MATCH (c:Chunk {chunk_id: $cid})-[:MENTIONS]->(n)
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

        # Step 3: +1-hop expansion
        expansion_triples = []
        with self.driver.session() as session:
            for name, typ in entity_ids:
                for row in session.run(
                    f"""
                    MATCH (e:{typ} {{name: $name}})-[rel]->(n)
                    WHERE NOT 'Chunk' IN labels(n)
                    RETURN e.name AS entity_name, labels(e) AS entity_type,
                           type(rel) AS rel_type, n.name AS neighbor_name, labels(n) AS neighbor_type
                    """,
                    name=name,
                ):
                    rel = row["rel_type"]
                    nbr = row["neighbor_name"]
                    if rel and nbr:
                        et = row["entity_type"][0] if row["entity_type"] else ""
                        nt = row["neighbor_type"][0] if row["neighbor_type"] else ""
                        expansion_triples.append(
                            f"{row['entity_name']} ({et}) --[{rel}]--> {nbr} ({nt})"
                        )
        expansion_triples = list(dict.fromkeys(expansion_triples))  # dedupe

        # Step 4: compose context
        context_sections = []
        for c in chunks:
            section = c["text"] + "\nMentions in this chunk:"
            for m in (chunk_to_mentions.get(c["id"], []) or ["(none)"]):
                section += f"\n- {m}"
            context_sections.append(section)

        if expansion_triples:
            context_sections.append(
                "Additional facts from knowledge graph:\n" + "\n".join(expansion_triples)
            )

        context = "\n\n---\n\n".join(context_sections)
        prompt = f"""Answer the following question using only the information in the context below.

Question: {question}

Context:
{context}

Answer:"""

        return self.llm.invoke(prompt)

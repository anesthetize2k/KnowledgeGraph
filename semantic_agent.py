from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from neo4j import GraphDatabase
import os


class SemanticAgent:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
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
            chunks = [{"id": row["cid"], "text": row["chunk"]} for row in result]
            return chunks

    def get_neighbor_triples(self, chunk_ids):
        with self.driver.session() as session:
            result = session.run(
                """
                UNWIND $chunk_ids AS cid
                MATCH (c:Chunk {chunk_id: cid})-[rel]->(n)
                RETURN cid, c.text AS chunk_text, type(rel) AS rel_type, n.name AS neighbor_name, labels(n) AS neighbor_type
                """,
                chunk_ids=chunk_ids,
            )
            triples = []
            entity_ids = set()
            for row in result:
                # Only collect triples where neighbor is not a Chunk
                if row["neighbor_type"] and "Chunk" not in row["neighbor_type"]:
                    triples.append(
                        f"Chunk '{row['cid']}' ({row['chunk_text'][:40]}...) --[{row['rel_type']}]--> {row['neighbor_name']} ({row['neighbor_type'][0]})"
                    )
                    # Track mentioned entity node IDs for +1 hop
                    entity_ids.add((row["neighbor_name"], row["neighbor_type"][0]))
            return triples, entity_ids

    def get_entity_expansion_triples(self, entities):
        with self.driver.session() as session:
            expansion_triples = []
            for name, typ in entities:
                # Expand from the mentioned entity (excluding Chunks)
                result = session.run(
                    f"""
                    MATCH (e:{typ} {{name: $name}})-[rel]->(n)
                    WHERE NOT 'Chunk' IN labels(n)
                    RETURN e.name AS entity_name, labels(e) AS entity_type,
                           type(rel) AS rel_type, n.name AS neighbor_name, labels(n) AS neighbor_type
                    """,
                    name=name,
                )
                for row in result:
                    if row["rel_type"] and row["neighbor_name"]:
                        expansion_triples.append(
                            f"{row['entity_name']} ({row['entity_type'][0] if row['entity_type'] else ''}) --[{row['rel_type']}]--> {row['neighbor_name']} ({row['neighbor_type'][0] if row['neighbor_type'] else ''})"
                        )
            return expansion_triples

    def run_query(self, question):
        chunks = self.get_relevant_chunks(question)
        if not chunks:
            return "🕵️ No relevant information found in the knowledge graph."
        chunk_ids = [c["id"] for c in chunks]

        # Step 1: Gather mentions for each chunk (not Chunks)
        with self.driver.session() as session:
            chunk_to_mentions = {}
            entity_ids = set()
            for cid in chunk_ids:
                mentions = []
                result = session.run(
                    """
                    MATCH (c:Chunk {chunk_id: $cid})-[:MENTIONS]->(n)
                    WHERE NOT 'Chunk' IN labels(n)
                    RETURN n.name AS name, labels(n) AS labels
                    """,
                    cid=cid,
                )
                for row in result:
                    if row["name"]:
                        label = row["labels"][0] if row["labels"] else ""
                        mentions.append(f"{row['name']} ({label})")
                        entity_ids.add((row["name"], label))
                chunk_to_mentions[cid] = mentions

        # Step 2: Gather +1-hop facts for all mentioned entities
        expansion_triples = []
        with self.driver.session() as session:
            for name, typ in entity_ids:
                result = session.run(
                    f"""
                    MATCH (e:{typ} {{name: $name}})-[rel]->(n)
                    WHERE NOT 'Chunk' IN labels(n)
                    RETURN e.name AS entity_name, labels(e) AS entity_type,
                        type(rel) AS rel_type, n.name AS neighbor_name, labels(n) AS neighbor_type
                    """,
                    name=name,
                )
                for row in result:
                    if row["rel_type"] and row["neighbor_name"]:
                        expansion_triples.append(
                            f"{row['entity_name']} ({row['entity_type'][0] if row['entity_type'] else ''}) --[{row['rel_type']}]--> {row['neighbor_name']} ({row['neighbor_type'][0] if row['neighbor_type'] else ''})"
                        )
        # Deduplicate
        expansion_triples = list(dict.fromkeys(expansion_triples))

        # Step 3: Compose context
        context_sections = []
        for c in chunks:
            section = c["text"] + "\nMentions in this chunk:"
            mentions = chunk_to_mentions.get(c["id"], [])
            if mentions:
                for m in mentions:
                    section += f"\n- {m}"
            else:
                section += "\n- (none)"
            context_sections.append(section)

        # At the end: All expanded facts
        if expansion_triples:
            context_sections.append(
                "Additional facts from knowledge graph:\n"
                + "\n".join(expansion_triples)
            )

        context = "\n\n---\n\n".join(context_sections)
        print("------\n" + context + "\n------")

        prompt = f"""Answer the following question using only the information in the context below.

    Question: {question}

    Context:
    {context}

    Answer:"""
        response = self.llm.invoke(prompt)
        return getattr(response, "content", str(response))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from neo4j import GraphDatabase
import os


class SemanticAgent:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
        )

    def get_relevant_chunks(self, question, top_k=5):
        embedding = self.embeddings.embed_query(question)
        with self.driver.session() as session:
            result = session.run(
                """
                CALL db.index.vector.queryNodes('vector', $topK, $embedding)
                YIELD node, score
                RETURN node.text AS chunk, score
                ORDER BY score DESC
                LIMIT $topK
                """,
                embedding=embedding,
                topK=top_k,
            )
            return [row["chunk"] for row in result]

    def run_query(self, question):
        chunks = self.get_relevant_chunks(question)
        if not chunks:
            return "🕵️ No relevant information found in the knowledge graph."
        # Compose a system prompt for the LLM
        context = "\n---\n".join(chunks)
        prompt = f"""Answer the following question using only the information in the context below.

Question: {question}

Context:
{context}

Answer:"""
        response = self.llm.invoke(prompt)
        return getattr(response, "content", str(response))

import os
import re
from neo4j import GraphDatabase
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

load_dotenv()


class TripletExtractor:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
        )
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

        self.prompt = PromptTemplate.from_template(
            """
        Extract factual (subject, predicate, object) triplets from the text below.
        Only include clear relationships (like 'has policy', 'reports to', 'established by').
        Avoid repeating facts that have no relation.
        Remove numbering and format each triplet like: (Subject, Predicate, Object)

        Text:
        {text}

        Triplets:
        """
        )

        self.chain = self.prompt | self.llm

    def parse_triplets(self, raw_output):
        lines = raw_output.strip().split("\n")
        triplets = []
        for line in lines:
            line = re.sub(r"^\s*\d+[\.\)]\s*", "", line)
            parts = line.strip("()").split(",")
            if len(parts) == 3:
                triplets.append(tuple(part.strip().strip("'\"") for part in parts))
        return triplets

    def insert_into_neo4j(self, triplets):
        with self.driver.session() as session:
            for s, p, o in triplets:
                session.run(
                    """
                    MERGE (a:Entity {name: $s})
                    MERGE (b:Entity {name: $o})
                    MERGE (a)-[:RELATION {type: $p}]->(b)
                """,
                    s=s,
                    p=p,
                    o=o,
                )

    def process_chunks(self, chunks):
        for i, chunk in enumerate(chunks):
            print(f"📄 Processing chunk {i}")
            try:
                response = self.chain.invoke({"text": chunk.page_content})
                print("📥 Raw triplet output:\n", response["text"])  # LLM output
                triplets = self.parse_triplets(response.content)
                self.insert_into_neo4j(triplets)
                print(f"  → ✅ {len(triplets)} triplets inserted.")
            except Exception as e:
                print(f"  ⚠️ Error on chunk {i}: {e}")

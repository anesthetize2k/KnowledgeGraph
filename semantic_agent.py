import os
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv

load_dotenv()


class SemanticAgent:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
        )
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=True,
            return_intermediate_steps=True,
            allow_dangerous_requests=True,  # ← This is required now
        )

    def ask(self, question: str):
        print(f"\n🤖 Answering: {question}")
        result = self.chain.run(question)
        print(f"\n✅ Result: {result}")
        return result

from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.chains.graph_qa.cypher import GraphCypherQAChain
from langchain.prompts import PromptTemplate
import os
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

        self.cypher_prompt = PromptTemplate.from_template(
            """
        You are an expert at writing Cypher for Neo4j.

        Write a Cypher query that:
        - Retrieves facts about the person mentioned in the question
        - Gets all nodes 1-2 hops away from the person
        - Returns their `name` and `labels`
        - Uses `labels()` to extract node types
        - Uses `type()` to get relationship names

        Example:
        Q: Who runs the finance ministry?
        Cypher:
        MATCH (p:Person)-[r]-(n)
        WHERE toLower(p.name) CONTAINS "nirmala"
        RETURN 
        p.name AS subject,
        labels(p) AS subject_type,
        type(r) AS relation,
        n.name AS object,
        labels(n) AS object_type
        LIMIT 25

        Now do this:
        Q: {question}
        Cypher:
        """
        )

        self.chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            cypher_prompt=self.cypher_prompt,
            verbose=True,
            return_intermediate_steps=True,
            allow_dangerous_requests=True,
        )

    def run_query(self, question: str) -> str:
        return self.chain.run(question)

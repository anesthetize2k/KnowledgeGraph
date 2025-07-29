import os
import ast
from typing import List, Tuple
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import ast
from typing import List, Tuple

load_dotenv()


class TripletExtractor:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
        )

        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

        self.prompt = PromptTemplate.from_template(
            """
You are an expert in extracting structured information from Indian government documents.

Given a prompt delimited by triple backticks, do the following:
- Identify factual relationships between entities in the text
- Output a list of triplets in the format:
  [ENTITY 1, TYPE of ENTITY 1, RELATION, ENTITY 2, TYPE of ENTITY 2]

### Allowed Entity Types:
- ministry, department, organization, policy, scheme, person, title, budget item, location, date

### Allowed Relation Types:
- has department, has minister, has title, announced, allocates budget to, has budget,
  located in, effective from, governs, implemented by

### Example:
prompt: “The Ministry of Finance, headed by Nirmala Sitharaman, announced the Fiscal Responsibility Act on 1 Feb 2024. ₹1 lakh crore was allocated to the PMJDY scheme, which is implemented by the Department of Financial Services.”

output:
[
["Ministry of Finance", "ministry", "has minister", "Nirmala Sitharaman", "person"],
["Nirmala Sitharaman", "person", "has title", "Finance Minister", "title"],
["Ministry of Finance", "ministry", "announced", "Fiscal Responsibility Act", "policy"],
["Fiscal Responsibility Act", "policy", "effective from", "1 Feb 2024", "date"],
["Ministry of Finance", "ministry", "allocates budget to", "PMJDY", "scheme"],
["PMJDY", "scheme", "has budget", "₹1 lakh crore", "budget item"],
["PMJDY", "scheme", "implemented by", "Department of Financial Services", "department"]
]

Now process this:
```{text}```
output:
            """
        )

        self.chain = self.prompt | self.llm

    def parse_triplets(self, raw_output: str) -> List[Tuple[str, str, str, str, str]]:
        try:
            raw_output = (
                raw_output.replace("```json", "")
                .replace("```", "")
                .replace("\n", "")
                .strip()
            )

            # Debug print to inspect input before parsing
            print("📥 Cleaned raw input to eval:\n", raw_output)

            # Use safe evaluation
            parsed = ast.literal_eval(raw_output)
            if not isinstance(parsed, list):
                raise ValueError("Parsed output is not a list of triplets")

            # Filter and clean
            cleaned = []
            for triplet in parsed:
                if isinstance(triplet, list) and len(triplet) == 5:
                    s, s_type, p, o, o_type = (x.strip(" \"'[]") for x in triplet)
                    cleaned.append((s, s_type, p, o, o_type))

            return cleaned

        except Exception as e:
            print("⚠️ Failed to parse structured triplets:", e)
            return []

    def insert_into_neo4j(self, triplets):
        with self.driver.session() as session:
            for s, s_type, p, o, o_type in triplets:
                # Sanitize labels (no spaces, title case)
                s_label = s_type.strip().title().replace(" ", "")
                o_label = o_type.strip().title().replace(" ", "")
                rel_type = p.strip().upper().replace(" ", "_")

                query = f"""
                MERGE (a:{s_label} {{name: $s}})
                MERGE (b:{o_label} {{name: $o}})
                MERGE (a)-[r:{rel_type}]->(b)
                """

                session.run(query, s=s.strip(), o=o.strip())

    def process_chunks(self, chunks):
        for i, chunk in enumerate(chunks):
            print(f"\n📄 Processing chunk {i + 1}/{len(chunks)}")
            try:
                response = self.chain.invoke({"text": chunk.page_content})
                print("📥 Raw triplet output:\n", response.content)

                triplets = self.parse_triplets(response.content)
                if triplets:
                    self.insert_into_neo4j(triplets)
                    print(f"✅ Inserted {len(triplets)} triplets into Neo4j")
                else:
                    print("⚠️ No valid triplets found.")
            except Exception as e:
                print(f"❌ Error processing chunk {i + 1}: {e}")
